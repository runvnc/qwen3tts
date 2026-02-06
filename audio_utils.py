"""
Audio conversion utilities for Qwen3-TTS server.

Includes:
- Proper anti-aliasing low-pass filter before downsampling (kills click energy)
- Boundary click detection and repair
- Stateful streaming resampler (maintains ratecv state across chunks)
- Standard ulaw conversion

The key insight: clicks are broadband impulses. Since we downsample to 8kHz
(Nyquist = 4kHz), a proper LPF at 24kHz removes click energy that audioop.ratecv's
mediocre built-in filter misses.
"""

import numpy as np
import audioop
from typing import List, Optional, Tuple
from scipy.signal import butter, sosfilt, sosfiltfilt

# Pre-compute the anti-aliasing filter coefficients (done once at import time)
# Cutoff at 3600Hz for 24kHz source -> 8kHz target (Nyquist=4000, leave margin)
_AA_FILTER_ORDER = 5
_AA_FILTER_CUTOFF = 3600.0
_AA_FILTER_SOURCE_SR = 24000
_AA_SOS = butter(_AA_FILTER_ORDER, _AA_FILTER_CUTOFF, btype='low',
                 fs=_AA_FILTER_SOURCE_SR, output='sos')


class StreamingAntiAliasFilter:
    """Streaming-compatible anti-aliasing filter using cascaded biquads.
    
    Maintains filter state between chunks so there are no discontinuities
    at chunk boundaries from the filter itself.
    """
    
    def __init__(self, sos=None):
        self.sos = sos if sos is not None else _AA_SOS
        # zi shape: (n_sections, 2) for sosfilt
        self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)
    
    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a chunk, maintaining state across calls."""
        if len(chunk) == 0:
            return chunk
        filtered, self._zi = sosfilt(self.sos, chunk.astype(np.float64),
                                     zi=self._zi)
        return filtered.astype(np.float32)
    
    def reset(self):
        """Reset filter state (call between utterances)."""
        self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)


class BoundaryClickRepair:
    """Detects and repairs clicks at chunk boundaries.
    
    Keeps the last few samples of the previous chunk. When a new chunk arrives,
    checks for a discontinuity at the junction and interpolates if found.
    """
    
    def __init__(self, overlap: int = 4, threshold: float = 0.15):
        self.overlap = overlap
        self.threshold = threshold  # amplitude jump threshold for click detection
        self._prev_tail: Optional[np.ndarray] = None
    
    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process a chunk, repairing any click at the boundary with previous chunk."""
        if len(chunk) == 0:
            return chunk
        
        chunk = chunk.copy()
        
        if self._prev_tail is not None and len(chunk) >= self.overlap:
            # Check for discontinuity at boundary
            last_val = self._prev_tail[-1]
            first_val = chunk[0]
            jump = abs(first_val - last_val)
            
            # Also check the derivative context - what was the trend?
            if len(self._prev_tail) >= 2:
                prev_slope = self._prev_tail[-1] - self._prev_tail[-2]
                expected_next = last_val + prev_slope
                deviation = abs(first_val - expected_next)
            else:
                deviation = jump
            
            if deviation > self.threshold:
                # Interpolate across the boundary
                # Use last 2 samples of prev + first 2 samples after repair zone
                n_repair = min(self.overlap, len(chunk))
                # Linear interpolation from prev tail end to chunk[n_repair]
                if len(chunk) > n_repair:
                    target = chunk[n_repair]
                else:
                    target = chunk[-1]
                
                interp = np.linspace(last_val, target, n_repair + 2,
                                     dtype=np.float32)
                chunk[:n_repair] = interp[1:n_repair + 1]
        
        # Save tail for next call
        self._prev_tail = chunk[-self.overlap:].copy()
        return chunk
    
    def reset(self):
        """Reset state between utterances."""
        self._prev_tail = None


class StreamingResampler:
    """Stateful resampler that maintains audioop.ratecv state across chunks.
    
    This prevents click artifacts at chunk boundaries during resampling
    by carrying the internal filter state from one chunk to the next.
    """
    
    def __init__(self, source_rate: int = 24000, target_rate: int = 8000):
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._ratecv_state = None
    
    def process(self, audio: np.ndarray) -> bytes:
        """Resample a float32 audio chunk to ulaw bytes, maintaining state.
        
        Args:
            audio: Float32 audio samples in range [-1.0, 1.0]
        
        Returns:
            u-law encoded bytes at target sample rate
        """
        if len(audio) == 0:
            return b''
        
        # Convert float32 to 16-bit PCM bytes
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767).astype(np.int16)
        pcm_bytes = pcm16.tobytes()
        
        # Resample with state
        if self.source_rate != self.target_rate:
            pcm_bytes, self._ratecv_state = audioop.ratecv(
                pcm_bytes, 2, 1,
                self.source_rate, self.target_rate,
                self._ratecv_state
            )
        
        # Convert to ulaw
        return audioop.lin2ulaw(pcm_bytes, 2)
    
    def reset(self):
        """Reset resampler state (call between utterances)."""
        self._ratecv_state = None


def float32_to_ulaw(audio: np.ndarray, sample_rate: int = 24000,
                    target_rate: int = 8000) -> bytes:
    """Convert float32 audio to ulaw at target sample rate.
    
    NOTE: This is stateless - each call resamples independently.
    For streaming use, prefer StreamingResampler which maintains state.
    
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
    
    # Resample using audioop.ratecv (stateless)
    if sample_rate != target_rate:
        pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, sample_rate, target_rate, None)
    
    # Convert to ulaw
    return audioop.lin2ulaw(pcm_bytes, 2)


def float32_to_ulaw_filtered(audio: np.ndarray, sample_rate: int = 24000,
                              target_rate: int = 8000) -> bytes:
    """Convert float32 audio to ulaw with proper anti-aliasing filter.
    
    Applies a Butterworth low-pass filter before downsampling to remove
    high-frequency content (including click energy) that would alias.
    This is a non-streaming version for one-shot conversion.
    
    Args:
        audio: Float32 audio samples in range [-1.0, 1.0]
        sample_rate: Input sample rate (default 24000)
        target_rate: Output sample rate (default 8000 for telephony)
    
    Returns:
        u-law encoded bytes at target sample rate
    """
    audio = np.clip(audio, -1.0, 1.0)
    
    # Apply anti-aliasing filter if downsampling
    if sample_rate > target_rate and len(audio) > _AA_FILTER_ORDER * 3:
        # Use sosfiltfilt for zero-phase filtering (no group delay)
        # Only works for non-streaming / complete chunks
        audio = sosfiltfilt(_AA_SOS, audio.astype(np.float64)).astype(np.float32)
    
    # Convert to 16-bit PCM
    pcm16 = (audio * 32767).astype(np.int16)
    pcm_bytes = pcm16.tobytes()
    
    # Resample
    if sample_rate != target_rate:
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
