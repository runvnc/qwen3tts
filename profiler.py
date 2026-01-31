"""
Simple profiler for tracking timing of generation steps.
"""

import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """Record of a single timed operation."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None


@dataclass
class GenerationProfile:
    """Profile for a single generation request."""
    request_id: str
    start_time: float = field(default_factory=time.time)
    timings: Dict[str, float] = field(default_factory=dict)
    markers: List[tuple] = field(default_factory=list)  # (name, timestamp)
    
    def mark(self, name: str):
        """Mark a point in time."""
        elapsed = (time.time() - self.start_time) * 1000
        self.markers.append((name, elapsed))
        logger.info(f"[{self.request_id}] {name}: +{elapsed:.0f}ms")
    
    def record(self, name: str, duration_ms: float):
        """Record a duration."""
        self.timings[name] = duration_ms
    
    def summary(self) -> str:
        """Get a summary of all timings."""
        total = (time.time() - self.start_time) * 1000
        lines = [f"Generation profile for {self.request_id}:"]
        
        for name, elapsed in self.markers:
            lines.append(f"  {name}: +{elapsed:.0f}ms")
        
        for name, duration in self.timings.items():
            lines.append(f"  {name}: {duration:.1f}ms")
        
        lines.append(f"  TOTAL: {total:.0f}ms")
        return "\n".join(lines)


class Profiler:
    """Manages profiling for generation requests."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._profiles: Dict[str, GenerationProfile] = {}
        self._current: Optional[GenerationProfile] = None
    
    def start(self, request_id: str) -> GenerationProfile:
        """Start profiling a new request."""
        if not self.enabled:
            return None
        
        profile = GenerationProfile(request_id=request_id)
        self._profiles[request_id] = profile
        self._current = profile
        profile.mark("start")
        return profile
    
    def mark(self, name: str, request_id: str = None):
        """Mark a point in the current or specified profile."""
        if not self.enabled:
            return
        
        profile = self._profiles.get(request_id) if request_id else self._current
        if profile:
            profile.mark(name)
    
    @contextmanager
    def measure(self, name: str, request_id: str = None):
        """Context manager to measure duration of a block."""
        if not self.enabled:
            yield
            return
        
        profile = self._profiles.get(request_id) if request_id else self._current
        start = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start) * 1000
            if profile:
                profile.record(name, duration)
                logger.info(f"[{profile.request_id}] {name}: {duration:.1f}ms")
    
    def finish(self, request_id: str = None) -> Optional[str]:
        """Finish profiling and return summary."""
        if not self.enabled:
            return None
        
        profile = self._profiles.get(request_id) if request_id else self._current
        if profile:
            profile.mark("finish")
            summary = profile.summary()
            logger.info(summary)
            
            # Cleanup
            if request_id and request_id in self._profiles:
                del self._profiles[request_id]
            if self._current == profile:
                self._current = None
            
            return summary
        return None


# Global profiler instance
profiler = Profiler(enabled=True)
