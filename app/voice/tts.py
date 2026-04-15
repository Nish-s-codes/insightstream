"""
app/voice/tts.py
Cartesia WebSocket TTS — correct for SDK 3.x (verified against source)

The SDK 3.x flow is:
    connection = await client.tts.websocket_connect().enter()
    ctx = connection.context(model_id=..., voice=..., output_format=...)
    await ctx.send(model_id=..., transcript=..., voice=..., output_format=...)
    async for response in ctx.receive():   # per-context queue
        if response.audio: ...

Key fixes vs previous version:
  - REMOVED continue_=False — the context manages this internally; passing it
    explicitly caused "multiple values for keyword argument 'continue_'"
  - REMOVED ws.receive(context_id) — doesn't exist; correct call is ctx.receive()
  - Use websocket_connect().enter() instead of the removed tts.websocket()
  - connection.context() creates an AsyncWebSocketContext per sentence
"""

import asyncio
from cartesia import AsyncCartesia


class CartesiaTTS:
    def __init__(self, api_key: str):
        self.client = AsyncCartesia(api_key=api_key)
        self.model_id = "sonic-english"
        self.voice: dict = {"mode": "id", "id": "a0e99841-438c-4a64-b679-ae501e7d6091"}
        self.sample_rate = 22050
        self.output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }
        self._conn = None          # AsyncTTSResourceConnection — reused across sentences
        self._lock = asyncio.Lock()

    async def _get_conn(self):
        """Return a live connection, opening one if needed."""
        if self._conn is None:
            self._conn = await self.client.tts.websocket_connect().enter()
        return self._conn

    async def _reset_conn(self):
        conn, self._conn = self._conn, None
        if conn:
            try:
                await conn.close()
            except Exception:
                pass

    async def stream_synthesize(self, text: str, on_chunk_cb):
        """
        Synthesise `text` and call on_chunk_cb(bytes) for every raw PCM packet.
        One WebSocket context is used per sentence.
        """
        if not text.strip():
            return

        async with self._lock:
            try:
                conn = await self._get_conn()

                # context() creates a fresh AsyncWebSocketContext with its own context_id
                ctx = conn.context(
                    model_id=self.model_id,
                    voice=self.voice,
                    output_format=self.output_format,
                )

                # DO NOT pass continue_= or context_id — the context owns those
                await ctx.send(
                    model_id=self.model_id,
                    transcript=text,
                    voice=self.voice,
                    output_format=self.output_format,
                )

                # ctx.receive() is an async-generator that stops on "done"/"error"
                async for response in ctx.receive():
                    if hasattr(response, "audio") and response.audio:
                        await on_chunk_cb(response.audio)

            except Exception as e:
                print(f"[TTS Error] {e}")
                await self._reset_conn()

    async def close(self):
        await self._reset_conn()
        try:
            await self.client.close()
        except Exception:
            pass