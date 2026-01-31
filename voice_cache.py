"""
Voice caching for Qwen3-TTS server.

Caches voice clone prompts and pre-computed embeddings to avoid
redundant computation on each generation request.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CachedVoice:
    """Cached voice data including prompt and pre-computed embeddings."""
    voice_id: str
    prompt_items: List[Any]
    created_at: float = field(default_factory=time.time)
    
    # Pre-computed embeddings (populated lazily)
    speaker_embed: Optional[torch.Tensor] = None
    ref_code: Optional[torch.Tensor] = None
    tts_const_embeds: Optional[tuple] = None  # (bos, eos, pad)
    codec_input_embedding: Optional[torch.Tensor] = None
    
    # Metadata
    ref_text: str = ""
    x_vector_only: bool = False


class VoiceCache:
    """Manages cached voice prompts and embeddings."""
    
    def __init__(self, max_voices: int = 50):
        self.max_voices = max_voices
        self._cache: Dict[str, CachedVoice] = {}
        self._access_times: Dict[str, float] = {}
    
    def compute_voice_id(self, audio_b64: str, ref_text: str, x_vector_only: bool) -> str:
        """Compute a unique ID for a voice based on audio hash."""
        hasher = hashlib.sha256()
        hasher.update(audio_b64.encode('utf-8')[:10000])  # First 10KB of base64
        hasher.update(ref_text.encode('utf-8'))
        hasher.update(str(x_vector_only).encode('utf-8'))
        return hasher.hexdigest()[:16]
    
    def get(self, voice_id: str) -> Optional[CachedVoice]:
        """Get a cached voice by ID."""
        if voice_id in self._cache:
            self._access_times[voice_id] = time.time()
            return self._cache[voice_id]
        return None
    
    def put(self, voice_id: str, prompt_items: List[Any], ref_text: str = "", x_vector_only: bool = False) -> CachedVoice:
        """Cache a voice prompt."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_voices:
            self._evict_oldest()
        
        cached = CachedVoice(
            voice_id=voice_id,
            prompt_items=prompt_items,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
        )
        self._cache[voice_id] = cached
        self._access_times[voice_id] = time.time()
        logger.info(f"Cached voice {voice_id}, total cached: {len(self._cache)}")
        return cached
    
    def update_embeddings(
        self,
        voice_id: str,
        speaker_embed: torch.Tensor = None,
        ref_code: torch.Tensor = None,
        tts_const_embeds: tuple = None,
        codec_input_embedding: torch.Tensor = None,
    ):
        """Update pre-computed embeddings for a cached voice."""
        cached = self._cache.get(voice_id)
        if cached:
            if speaker_embed is not None:
                cached.speaker_embed = speaker_embed
            if ref_code is not None:
                cached.ref_code = ref_code
            if tts_const_embeds is not None:
                cached.tts_const_embeds = tts_const_embeds
            if codec_input_embedding is not None:
                cached.codec_input_embedding = codec_input_embedding
            logger.debug(f"Updated embeddings for voice {voice_id}")
    
    def has_embeddings(self, voice_id: str) -> bool:
        """Check if a voice has pre-computed embeddings."""
        cached = self._cache.get(voice_id)
        if cached:
            return cached.speaker_embed is not None
        return False
    
    def _evict_oldest(self):
        """Evict the least recently accessed voice."""
        if not self._access_times:
            return
        oldest_id = min(self._access_times, key=self._access_times.get)
        del self._cache[oldest_id]
        del self._access_times[oldest_id]
        logger.info(f"Evicted voice {oldest_id} from cache")
    
    def clear(self):
        """Clear all cached voices."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Voice cache cleared")
    
    def __len__(self):
        return len(self._cache)
    
    def __contains__(self, voice_id: str):
        return voice_id in self._cache
