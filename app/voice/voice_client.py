"""
app/voice/voice_client.py
Real-time Voice Assistant
Pipeline: Mic -> Deepgram (WS) -> LLM (SSE) -> Cartesia (WS) -> Speakers

Fixes in this version:
  1. Text prints to console immediately when LLM chunk arrives (before TTS queuing)
  2. Sentence splitter uses \s* not \s+ — no longer waits for a space after final
     punctuation, so the last sentence of a response is spoken without delay
  3. VAD barge-in now uses the stricter consecutive-frame is_speech() from stt.py
  4. Persistent OutputStream (no per-sentence open/close)
  5. asyncio.get_running_loop() (not deprecated get_event_loop())
  6. Clean barge-in: awaits task cancellation before starting next turn
  7. STUTTER FIX: out_stream.write() called directly — no run_in_executor overhead
  8. STUTTER FIX: blocksize raised 2048 -> 4096 for more buffer headroom
"""

import os
import asyncio
import json
import re
import numpy as np
import sounddevice as sd
import httpx
from dotenv import load_dotenv

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

from app.voice.stt import (
    initialize_vad, is_speech,
    SAMPLE_RATE, FRAME_SIZE,
    VAD_THRESHOLD_WHILE_BOT_SPEAKING,
)
from app.voice.tts import CartesiaTTS

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SENTENCE_END = re.compile(r'(?<=[.!?])\s*')

BASE_URL = os.getenv("VOICE_API_BASE", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
mic_queue: asyncio.Queue = asyncio.Queue()
interrupted = asyncio.Event()
bot_speaking = asyncio.Event()
current_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# Mic callback  (sounddevice thread → non-blocking)
# ---------------------------------------------------------------------------
def mic_callback(indata, frames, time_info, status):
    if status:
        print(f"[Mic] {status}")
    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    mic_queue.put_nowait(pcm)


# ---------------------------------------------------------------------------
# LLM — SSE streaming
# ---------------------------------------------------------------------------
async def ask_sse(query: str, session_id: str):
    """Yield raw text chunks from the /ask SSE endpoint."""
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            async with client.stream(
                "GET", f"{BASE_URL}/ask",
                params={"q": query, "session_id": session_id},
                headers={"accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if interrupted.is_set():
                        return
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            yield data.get("text", "")
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"[LLM Error] {e}")


# ---------------------------------------------------------------------------
# TTS + Playback
# ---------------------------------------------------------------------------
async def play_sentence(text: str, tts: CartesiaTTS, out_stream: sd.OutputStream):
    """Synthesise one sentence and write PCM into the already-open OutputStream."""
    if not text.strip() or interrupted.is_set():
        return

    async def on_chunk(chunk: bytes):
        if interrupted.is_set():
            return
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        # FIX 7: write directly — run_in_executor adds thread-scheduling latency
        # that causes gaps between chunks → stutter
        out_stream.write(audio)

    await tts.stream_synthesize(text, on_chunk)


async def speak_stream(text_iter, tts: CartesiaTTS):
    """
    Consume LLM text chunks, print them immediately, buffer into sentences,
    then feed each sentence to TTS as soon as it's complete.
    """
    out_stream = sd.OutputStream(
        samplerate=tts.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=4096,   # FIX 8: was 2048 — larger buffer prevents underruns
    )
    out_stream.start()
    bot_speaking.set()

    try:
        buffer = ""
        async for chunk in text_iter:
            if interrupted.is_set():
                break

            print(chunk, end="", flush=True)

            buffer += chunk

            parts = _SENTENCE_END.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    if interrupted.is_set():
                        break
                    await play_sentence(sentence.strip(), tts, out_stream)
                buffer = parts[-1]

        # Speak any remaining text (e.g. a sentence without trailing punctuation)
        if buffer.strip() and not interrupted.is_set():
            await play_sentence(buffer.strip(), tts, out_stream)

    finally:
        out_stream.stop()
        out_stream.close()
        bot_speaking.clear()


# ---------------------------------------------------------------------------
# Turn orchestration
# ---------------------------------------------------------------------------
async def handle_turn(query: str, tts: CartesiaTTS, session_id: str):
    interrupted.clear()
    print(f"\nYou: {query}")
    print("Bot: ", end="", flush=True)

    try:
        await speak_stream(ask_sse(query, session_id), tts)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\n[Turn Error] {e}")
    finally:
        bot_speaking.clear()
        print()


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------
async def run(session_id: str):
    global current_task

    initialize_vad()
    tts = CartesiaTTS(os.getenv("CARTESIA_API_KEY", ""))

    dg_client = DeepgramClient()
    dg_conn = dg_client.listen.asynclive.v("1")

    async def on_transcript(self, result, **kwargs):
        global current_task
        if not result.channel.alternatives:
            return

        transcript = result.channel.alternatives[0].transcript
        if not transcript:
            return

        if result.is_final and getattr(result, "speech_final", False):
            if current_task and not current_task.done():
                interrupted.set()
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass

            current_task = asyncio.create_task(
                handle_turn(transcript, tts, session_id)
            )

    dg_conn.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_conn.on(
        LiveTranscriptionEvents.Error,
        lambda self, error, **kwargs: print(f"[Deepgram Error] {error}"),
    )

    options = LiveOptions(
        model="nova-2",
        language="en",
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        channels=1,
        interim_results=True,
        endpointing=300,
        vad_events=True,
    )

    print("Connecting to Deepgram...")
    if not await dg_conn.start(options):
        print("Failed to connect to Deepgram.")
        return

    async def audio_pump():
        while True:
            try:
                data = await mic_queue.get()

                if bot_speaking.is_set() and is_speech(data, VAD_THRESHOLD_WHILE_BOT_SPEAKING):
                    if not interrupted.is_set():
                        interrupted.set()
                        if current_task and not current_task.done():
                            current_task.cancel()
                        print("\n[Barge-in]")

                await dg_conn.send(data)
                await asyncio.sleep(0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Audio Pump Error] {e}")
                await asyncio.sleep(0.1)

    pump_task = asyncio.create_task(audio_pump())

    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=mic_callback,
        blocksize=FRAME_SIZE,
        dtype="float32",
    )

    with mic_stream:
        print("Session active. Speak to InsightStream!\n")
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nShutting down...")
        finally:
            pump_task.cancel()
            if current_task and not current_task.done():
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            await dg_conn.finish()
            await tts.close()


if __name__ == "__main__":
    asyncio.run(run("default_session"))