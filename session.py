"""
Session management for Qwen3-TTS server.
"""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class VoiceSession:
    """Holds voice clone state for a WebSocket connection."""
    voice_prompt: Optional[List[Any]] = None
    sample_rate: int = 24000
    text_buffer: str = ""
    is_generating: bool = False
    cancel_requested: bool = False
    voice_id: Optional[str] = None  # Reference to cached voice
