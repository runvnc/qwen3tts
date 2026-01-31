"""
Audio conversion utilities for Qwen3-TTS server.
"""

import numpy as np
from typing import List


def float32_to_ulaw(audio: np.ndarray, sample_rate: int = 24000, target_rate: int = 8000) -> bytes:
    """Convert float32 audio to ulaw 8kHz."""
    # Resample if needed
    if sample_rate != target_rate:
        duration = len(audio) / sample_rate
        target_samples = int(duration * target_rate)
        indices = np.linspace(0, len(audio) - 1, target_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio)

    # Normalize and convert to 16-bit PCM
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)

    # Convert to ulaw
    return pcm16_to_ulaw(pcm16)


def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
    """Convert 16-bit PCM to u-law using numpy (no audioop dependency)."""
    BIAS = 0x84
    CLIP = 32635

    sign = (pcm16 < 0).astype(np.uint8) * 0x80
    pcm16 = np.abs(pcm16).clip(0, CLIP)
    pcm16 = pcm16 + BIAS

    exponent = np.floor(np.log2(pcm16)).astype(np.uint8) - 7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (pcm16 >> (exponent + 3)) & 0x0F

    ulaw_bytes = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw_bytes.astype(np.uint8).tobytes()


def chunk_audio(audio_bytes: bytes, chunk_size: int = 160) -> List[bytes]:
    """Split audio into chunks (160 bytes = 20ms at 8kHz ulaw)."""
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        if len(chunk) < chunk_size:
            # Pad last chunk with silence (ulaw silence = 0xFF)
            chunk = chunk + bytes([0xFF] * (chunk_size - len(chunk)))
        chunks.append(chunk)
    return chunks
