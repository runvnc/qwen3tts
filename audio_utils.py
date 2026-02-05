"""
Audio conversion utilities for Qwen3-TTS server.

Uses audioop for resampling and ulaw conversion (same as mr_pocket_tts).
"""

import numpy as np
import audioop
from typing import List


def float32_to_ulaw(audio: np.ndarray, sample_rate: int = 24000, target_rate: int = 8000) -> bytes:
    """Convert float32 audio to ulaw at target sample rate.
    
    Uses audioop.ratecv for resampling (same approach as mr_pocket_tts).
    
    Args:
        audio: Float32 audio samples in range [-1.0, 1.0]
        sample_rate: Input sample rate (default 24000)
        target_rate: Output sample rate (default 8000 for telephony)
    
    Returns:
        u-law encoded bytes at target sample rate
    """
    # Convert float32 to 16-bit PCM bytes
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    pcm_bytes = pcm16.tobytes()
    
    # Resample using audioop.ratecv (same as mr_pocket_tts)
    if sample_rate != target_rate:
        # audioop.ratecv(fragment, width, nchannels, inrate, outrate, state)
        pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, sample_rate, target_rate, None)
    
    # Convert to ulaw
    return audioop.lin2ulaw(pcm_bytes, 2)


def chunk_audio(audio_bytes: bytes, chunk_size: int = 160) -> List[bytes]:
    """Split audio into chunks (160 bytes = 20ms at 8kHz ulaw).
    
    Args:
        audio_bytes: u-law encoded audio bytes
        chunk_size: Size of each chunk in bytes (default 160 = 20ms)
    
    Returns:
        List of audio chunks, last chunk padded with silence if needed
    """
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        if len(chunk) < chunk_size:
            # Pad last chunk with silence (ulaw silence = 0xFF)
            chunk = chunk + bytes([0xFF] * (chunk_size - len(chunk)))
        chunks.append(chunk)
    return chunks
