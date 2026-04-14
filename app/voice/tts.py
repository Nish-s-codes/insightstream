"""
app/voice/tts.py
Cartesia TTS wrapper.
"""

from cartesia import AsyncCartesia


class CartesiaTTS:
    def __init__(self, api_key: str):
        self.client      = AsyncCartesia(api_key=api_key)
        self.model_id    = "sonic-english"
        self.voice_id    = "a0e99841-438c-4a64-b679-ae501e7d6091"
        self.sample_rate = 22050

    def _output_format(self) -> dict:
        return {
            "container":   "raw",
            "encoding":    "pcm_s16le",
            "sample_rate": self.sample_rate,
        }

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        buf = b""
        response = await self.client.tts.bytes(
            model_id=self.model_id,
            transcript=text,
            voice={"mode": "id", "id": self.voice_id},
            output_format=self._output_format(),
        )
        if isinstance(response, bytes):
            return response
        # iterable of chunks
        async for chunk in response:
            buf += chunk
        return buf

    async def stream_synthesize(self, text: str, on_chunk) -> None:
        if not text.strip():
            return
        response = await self.client.tts.bytes(
            model_id=self.model_id,
            transcript=text,
            voice={"mode": "id", "id": self.voice_id},
            output_format=self._output_format(),
        )
        if isinstance(response, bytes):
            await on_chunk(response)
            return
        async for chunk in response:
            await on_chunk(chunk)

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass