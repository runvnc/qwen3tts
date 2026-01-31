"""
Audio transcription utilities using Whisper.
"""

import io
import logging
import os
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
    _whisper_model = None
except ImportError:
    logger.warning("whisper not available, auto-transcription disabled")
    WHISPER_AVAILABLE = False
    whisper = None


def get_whisper_model():
    """Get or load the Whisper model for transcription."""
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return None
    if _whisper_model is None:
        model_size = os.environ.get('WHISPER_MODEL', 'base')
        logger.info(f"Loading Whisper model: {model_size}")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe_audio(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio using Whisper."""
    model = get_whisper_model()
    if model is None:
        return ""

    # Whisper expects 16kHz audio
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    result = model.transcribe(audio, fp16=False)
    return result.get('text', '').strip()


def decode_audio_input(audio_b64: str) -> Tuple[np.ndarray, int]:
    """Decode base64 audio to numpy array."""
    import base64
    import soundfile as sf

    # Handle data URL format
    if audio_b64.startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_b64)

    with io.BytesIO(audio_bytes) as f:
        audio, sr = sf.read(f, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    return audio.astype(np.float32), int(sr)
