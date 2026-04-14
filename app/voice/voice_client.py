"""
app/voice/voice_client.py
Full voice pipeline: Mic → VAD → Deepgram STT → /ask SSE → Cartesia TTS
"""

import os
import io
import wave
import asyncio
import threading
import time
import re
import json
import numpy as np
import sounddevice as sd
import httpx
from collections import deque
from dotenv import load_dotenv

load_dotenv()

from deepgram import DeepgramClient
from app.voice.stt import (
    initialize_vad, is_speech,
    SAMPLE_RATE, FRAME_SIZE, BYTES_PER_FRAME,
    SILENCE_THRESHOLD, VAD_THRESHOLD, VAD_THRESHOLD_WHILE_BOT_SPEAKING,
)
from app.voice.tts import CartesiaTTS

BASE_URL = os.getenv("VOICE_API_BASE", "http://localhost:8000")

_SENTENCE_END  = re.compile(r'(?<=[.!?,])\s+')
_MIN_TTS_WORDS = 4


# ── Shared state ───────────────────────────────────────────────────────────────

mic_frames    = deque()
bot_speaking  = threading.Event()
interrupted   = threading.Event()
bot_stop_time = 0.0
running       = True


# ── Mic callback ───────────────────────────────────────────────────────────────

def mic_callback(indata, frames, time_info, status):
    mic_frames.append((indata[:, 0] * 32767).astype(np.int16).tobytes())


# ── Playback ───────────────────────────────────────────────────────────────────

def play_audio(pcm_bytes: bytes, sample_rate: int) -> None:
    global bot_stop_time
    audio      = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_size = sample_rate // 20  # 50 ms chunks → fast interrupt response
    offset     = 0
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while offset < len(audio) and not interrupted.is_set():
            end   = min(offset + chunk_size, len(audio))
            chunk = audio[offset:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            stream.write(chunk)
            offset = end
    bot_speaking.clear()
    bot_stop_time = time.time()


# ── STT ────────────────────────────────────────────────────────────────────────

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def transcribe(pcm_bytes: bytes) -> str:
    dg = DeepgramClient()
    try:
        resp = await asyncio.to_thread(
            dg.listen.v1.media.transcribe_file,
            request=_pcm_to_wav(pcm_bytes),
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
        )
        return resp.results.channels[0].alternatives[0].transcript.strip()
    except Exception as e:
        print(f"[STT error] {e}")
        return ""


# ── /ask SSE ───────────────────────────────────────────────────────────────────

async def ask_sse(query: str, session_id: str) -> str:
    full = []
    print("\nBot: ", end="", flush=True)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "GET", f"{BASE_URL}/ask",
                params={"q": query, "session_id": session_id},
                headers={"accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            text = json.loads(line[6:]).get("text", "")
                            if text:
                                print(text, end="", flush=True)
                                full.append(text)
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        print(f"[ask error] {e}")
    print()
    return "".join(full)


# ── TTS ────────────────────────────────────────────────────────────────────────

async def speak(text: str, tts: CartesiaTTS) -> None:
    global bot_stop_time
    sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]

    for i, sentence in enumerate(sentences):
        if interrupted.is_set():
            break
        if len(sentence.split()) < _MIN_TTS_WORDS and i < len(sentences) - 1:
            sentences[i + 1] = sentence + " " + sentences[i + 1]
            continue
        try:
            audio_bytes = await tts.synthesize(sentence)
        except Exception as e:
            print(f"[TTS error] {e}")
            continue
        if interrupted.is_set():
            break

        interrupted.clear()
        bot_stop_time = time.time() + len(audio_bytes) / 2 / tts.sample_rate + 0.8
        bot_speaking.set()

        t = threading.Thread(target=play_audio, args=(audio_bytes, tts.sample_rate), daemon=True)
        t.start()
        t.join()

    bot_speaking.clear()


# ── VAD loop ───────────────────────────────────────────────────────────────────

async def vad_loop(on_utterance) -> None:
    accumulated   = b""
    last_speech   = time.time()
    user_speaking = False

    while running:
        await asyncio.sleep(0.005)
        while mic_frames:
            raw = mic_frames.popleft()
            for pos in range(0, len(raw) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
                frame     = raw[pos: pos + BYTES_PER_FRAME]
                threshold = VAD_THRESHOLD_WHILE_BOT_SPEAKING if time.time() < bot_stop_time else VAD_THRESHOLD

                if is_speech(frame, threshold):
                    last_speech    = time.time()
                    accumulated   += frame
                    if not user_speaking:
                        user_speaking = True
                    if bot_speaking.is_set():
                        interrupted.set()
                        bot_speaking.clear()
                        print("\n[interrupted]", flush=True)

                elif user_speaking:
                    accumulated += frame
                    if time.time() - last_speech >= SILENCE_THRESHOLD:
                        user_speaking = False
                        utterance, accumulated = accumulated, b""
                        await on_utterance(utterance)


# ── Turn handler ───────────────────────────────────────────────────────────────

async def handle_turn(pcm: bytes, tts: CartesiaTTS, session_id: str) -> None:
    global running
    print("Transcribing…", end=" ", flush=True)
    transcript = await transcribe(pcm)
    if not transcript:
        print("(nothing heard)")
        return
    print(f"\nYou: {transcript}")
    if transcript.lower().strip() in {"exit", "quit", "goodbye", "bye"}:
        print("Goodbye!")
        running = False
        return
    interrupted.clear()
    reply = await ask_sse(transcript, session_id)
    if reply and not interrupted.is_set():
        await speak(reply, tts)


# ── Entry ──────────────────────────────────────────────────────────────────────

async def run(session_id: str) -> None:
    global running
    initialize_vad()
    tts = CartesiaTTS(api_key=os.getenv("CARTESIA_API_KEY", ""))

    print("Listening… speak now. 1.25 s silence sends your turn. Say 'exit' to quit.\n")

    async def on_utterance(pcm: bytes):
        await handle_turn(pcm, tts, session_id)

    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=FRAME_SIZE, callback=mic_callback,
    )
    try:
        with mic_stream:
            await vad_loop(on_utterance)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        running = False
        interrupted.set()
        await tts.close()