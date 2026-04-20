"""
app/voice/stt.py

Fixes vs original:
  1. VAD_THRESHOLD raised 0.4 → 0.7  (0.4 fires on keyboard clicks, chair noise)
  2. VAD_THRESHOLD_WHILE_BOT_SPEAKING raised 0.6 → 0.85
  3. Consecutive-frame gate: a single loud frame is NOT speech.
     Require VAD_CONFIRMATION_FRAMES consecutive frames above threshold.
     At FRAME_SIZE=512 / SAMPLE_RATE=16000 each frame ≈ 32 ms.
     3 frames = ~96 ms of sustained voice → kills transient pops/clicks.
"""

import torch
import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_SIZE = 512
BYTES_PER_FRAME = FRAME_SIZE * 2          # 16-bit PCM = 2 bytes/sample

# ---------------------------------------------------------------------------
# VAD tuning
# ---------------------------------------------------------------------------
VAD_THRESHOLD = 0.70                      # was 0.4 — too sensitive
VAD_THRESHOLD_WHILE_BOT_SPEAKING = 0.85  # was 0.6 — echo from speaker was triggering barge-in
SILENCE_THRESHOLD = 0.8                   # seconds of silence → end of turn (used by Deepgram endpointing)

# Number of consecutive frames that must ALL be above threshold before
# is_speech() returns True.  Eliminates single-frame transient noise.
VAD_CONFIRMATION_FRAMES = 3

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model = None
_frame_history: deque[bool] = deque(maxlen=VAD_CONFIRMATION_FRAMES)


def initialize_vad():
    global _model
    if _model is None:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _model = model


def is_speech(frame_bytes: bytes, threshold: float = VAD_THRESHOLD) -> bool:
    """
    Returns True only when the last VAD_CONFIRMATION_FRAMES frames ALL scored
    above `threshold`.  A single noisy frame never triggers a barge-in.
    """
    if _model is None:
        return False

    audio_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    with torch.no_grad():
        confidence = _model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()

    _frame_history.append(confidence > threshold)

    # Require the full window to be filled AND every frame to be True
    return len(_frame_history) == VAD_CONFIRMATION_FRAMES and all(_frame_history)