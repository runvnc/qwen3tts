"""
Audio conversion utilities for Qwen3-TTS server.

Fixed version with:
1. Proper anti-aliased resampling (scipy.signal.resample_poly)
2. Verified u-law encoding using standard audioop or correct numpy implementation
"""

import numpy as np
from typing import List

# Try to use scipy for proper resampling
try:
    from scipy.signal import resample_poly
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to use audioop for verified ulaw encoding
try:
    import audioop
    AUDIOOP_AVAILABLE = True
except ImportError:
    AUDIOOP_AVAILABLE = False


def resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio with proper anti-aliasing filter.
    
    Args:
        audio: Input audio as float32 numpy array
        sample_rate: Original sample rate
        target_rate: Target sample rate
    
    Returns:
        Resampled audio as float32 numpy array
    """
    if sample_rate == target_rate:
        return audio
    
    if SCIPY_AVAILABLE:
        # Use scipy's polyphase resampling with anti-aliasing
        # Find GCD to get integer up/down factors
        from math import gcd
        g = gcd(sample_rate, target_rate)
        up = target_rate // g
        down = sample_rate // g
        
        # resample_poly applies a low-pass filter automatically
        return resample_poly(audio, up, down).astype(np.float32)
    else:
        # Fallback: simple linear interpolation (not ideal but works)
        # Apply a simple low-pass filter first to reduce aliasing
        if target_rate < sample_rate:
            # Downsample case - apply crude low-pass by averaging
            ratio = sample_rate // target_rate
            if ratio > 1:
                # Simple box filter
                kernel_size = ratio
                kernel = np.ones(kernel_size) / kernel_size
                audio = np.convolve(audio, kernel, mode='same')
        
        duration = len(audio) / sample_rate
        target_samples = int(duration * target_rate)
        indices = np.linspace(0, len(audio) - 1, target_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def pcm16_to_ulaw_audioop(pcm16: np.ndarray) -> bytes:
    """Convert 16-bit PCM to u-law using audioop (verified implementation)."""
    pcm_bytes = pcm16.astype('<i2').tobytes()  # Little-endian 16-bit
    return audioop.lin2ulaw(pcm_bytes, 2)


def pcm16_to_ulaw_numpy(pcm16: np.ndarray) -> bytes:
    """Convert 16-bit PCM to u-law using numpy.
    
    This is the standard ITU-T G.711 u-law encoding algorithm.
    """
    BIAS = 0x84  # 132
    CLIP = 32635  # Max value before clipping
    
    # Work with int32 to avoid overflow issues
    samples = pcm16.astype(np.int32)
    
    # Get sign and work with absolute values
    sign = np.where(samples < 0, 0x80, 0).astype(np.uint8)
    samples = np.abs(samples)
    
    # Clip to valid range
    samples = np.clip(samples, 0, CLIP)
    
    # Add bias
    samples = samples + BIAS
    
    # Find the exponent using log2
    exponent = np.floor(np.log2(np.maximum(samples, 1))).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7).astype(np.uint8)
    
    # Extract mantissa (4 bits)
    mantissa = ((samples >> (exponent + 3)) & 0x0F).astype(np.uint8)
    
    # Combine: sign(1) + exponent(3) + mantissa(4), then invert
    ulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    
    return ulaw.astype(np.uint8).tobytes()


def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
    """Convert 16-bit PCM to u-law.
    
    Uses audioop if available (verified), otherwise numpy implementation.
    """
    if AUDIOOP_AVAILABLE:
        return pcm16_to_ulaw_audioop(pcm16)
    else:
        return pcm16_to_ulaw_numpy(pcm16)


def float32_to_ulaw(audio: np.ndarray, sample_rate: int = 24000, target_rate: int = 8000) -> bytes:
    """Convert float32 audio to ulaw at target sample rate.
    
    Args:
        audio: Float32 audio samples in range [-1.0, 1.0]
        sample_rate: Input sample rate (default 24000)
        target_rate: Output sample rate (default 8000 for telephony)
    
    Returns:
        u-law encoded bytes at target sample rate
    """
    # Resample with anti-aliasing
    if sample_rate != target_rate:
        audio = resample_audio(audio, sample_rate, target_rate)
    
    # Normalize and convert to 16-bit PCM
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    
    # Convert to ulaw
    return pcm16_to_ulaw(pcm16)


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
