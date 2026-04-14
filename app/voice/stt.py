"""
app/voice/stt.py
VAD (Silero) constants and helpers.

Deepgram is used for transcription in voice_client.py.
This module only owns:
  - Audio constants shared across voice components
  - Silero VAD model loading + inference
"""

import torch

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000   # Mic capture rate (Deepgram + Silero both want 16 kHz)
FRAME_SIZE       = 512     # Samples per VAD frame (~32 ms at 16 kHz)
BYTES_PER_FRAME  = FRAME_SIZE * 2  # 16-bit PCM → 2 bytes per sample

# Silence detector: if no speech for this many seconds → send turn to LLM
SILENCE_THRESHOLD = 1.25   # seconds (same as reference)

# VAD confidence thresholds
VAD_THRESHOLD                    = 0.30  # Normal listening
VAD_THRESHOLD_WHILE_BOT_SPEAKING = 0.98  # Much stricter — avoids bot's own audio triggering VAD

# ── Model singleton ────────────────────────────────────────────────────────────
_vad_model = None


def initialize_vad() -> None:
    """Load Silero VAD once. Safe to call multiple times."""
    global _vad_model
    if _vad_model is not None:
        return
    print("Loading Silero VAD...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    _vad_model = model
    print("VAD ready.")


def is_speech(frame_bytes: bytes, threshold: float = VAD_THRESHOLD) -> bool:
    """
    Return True if the given PCM frame (int16, 16 kHz, mono) contains speech.

    Args:
        frame_bytes: Raw bytes — exactly BYTES_PER_FRAME bytes (FRAME_SIZE * 2).
        threshold:   VAD confidence cutoff.
    """
    if len(frame_bytes) != BYTES_PER_FRAME:
        return False

    audio = torch.frombuffer(bytearray(frame_bytes), dtype=torch.int16).clone()
    audio = audio.float() / 32768.0
    prob  = _vad_model(audio, SAMPLE_RATE).item()
    return prob > threshold