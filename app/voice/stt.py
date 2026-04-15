import torch
import numpy as np

# Audio Constants
SAMPLE_RATE = 16000
FRAME_SIZE = 512
BYTES_PER_FRAME = FRAME_SIZE * 2  # 16-bit PCM

# VAD Constants
VAD_THRESHOLD = 0.4
VAD_THRESHOLD_WHILE_BOT_SPEAKING = 0.6  # Higher threshold to ignore bot echo
SILENCE_THRESHOLD = 0.8  # Seconds of silence to trigger end of turn

_model = None

def initialize_vad():
    global _model
    if _model is None:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        _model = model

def is_speech(frame_bytes: bytes, threshold: float = VAD_THRESHOLD) -> bool:
    if _model is None:
        return False
    audio_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    with torch.no_grad():
        confidence = _model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
    return confidence > threshold