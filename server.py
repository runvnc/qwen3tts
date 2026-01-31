#!/usr/bin/env python3
"""
Qwen3-TTS WebSocket Streaming Server

This server provides:
1. Voice cloning with reference audio
2. Streaming text input (buffered to word boundaries)
3. Streaming audio output (chunked ulaw 8kHz for SIP compatibility)

Protocol:
- Connect to ws://host:port/ws
- Send JSON messages for control
- Receive binary audio chunks (ulaw 8kHz) or JSON status messages

Message Types (client -> server):
- {"type": "init", "ref_audio_base64": "...", "ref_text": "...", "language": "Auto"}
  Initialize voice clone with reference audio
  
- {"type": "text", "text": "Hello world", "final": false}
  Send text chunk. Set final=true when done with utterance.
  
- {"type": "generate", "text": "Full text here"}
  Generate audio for complete text (non-streaming input)

- {"type": "cancel"}
  Cancel current generation

Message Types (server -> client):
- {"type": "ready", "voice_loaded": true}
  Server ready, voice clone initialized
  
- {"type": "audio_start"}
  Audio generation starting
  
- Binary data: raw ulaw 8kHz audio chunks (~20ms each, 160 bytes)
  
- {"type": "audio_end"}
  Audio generation complete
  
- {"type": "error", "message": "..."}
  Error occurred
"""

import asyncio
import base64
import io
import json
import hashlib
import tempfile
import logging
import os
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import qwen_tts
try:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    QWEN_TTS_AVAILABLE = True
except ImportError:
    logger.warning("qwen_tts not available, running in mock mode")
    QWEN_TTS_AVAILABLE = False
    Qwen3TTSModel = None
    VoiceClonePromptItem = None

# Try to import whisper for auto-transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
    _whisper_model = None
except ImportError:
    logger.warning("whisper not available, auto-transcription disabled")
    WHISPER_AVAILABLE = False
    whisper = None

# Audio conversion utilities
def float32_to_ulaw(audio: np.ndarray, sample_rate: int = 24000, target_rate: int = 8000) -> bytes:
    """Convert float32 audio to ulaw 8kHz for SIP/telephony."""
    import audioop
    
    # Resample if needed
    if sample_rate != target_rate:
        # Simple linear interpolation resampling
        duration = len(audio) / sample_rate
        target_samples = int(duration * target_rate)
        indices = np.linspace(0, len(audio) - 1, target_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    
    # Normalize and convert to 16-bit PCM
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    pcm_bytes = pcm16.tobytes()
    
    # Convert to ulaw
    ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
    return ulaw_bytes


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


@dataclass
class VoiceSession:
    """Holds voice clone state for a connection."""
    voice_prompt: Optional[List[Any]] = None
    sample_rate: int = 24000
    text_buffer: str = ""
    is_generating: bool = False
    cancel_requested: bool = False
    voice_id: Optional[str] = None  # Reference to cached voice


class Qwen3TTSServer:
    """WebSocket server for Qwen3-TTS streaming."""
    
    def __init__(
        self,
        model_path: str = None,  # Auto-detect based on VRAM
        fallback_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        host: str = "0.0.0.0",
        port: int = 8765,
    ):
        self.model_path = model_path
        self.fallback_model = fallback_model
        self.device = device
        self.dtype = dtype
        self.host = host
        self.port = port
        
        self.model: Optional[Qwen3TTSModel] = None
        self.sessions: Dict[str, VoiceSession] = {}
        
        # Persistent voice cache (survives across connections)
        self.voice_cache: Dict[str, List[Any]] = {}
        
    def _get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.dtype.lower(), torch.bfloat16)
    
    def _detect_model(self) -> str:
        """Auto-detect which model to use based on available VRAM."""
        if self.model_path:
            return self.model_path
        
        # Default models
        large_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        small_model = self.fallback_model
        
        try:
            if torch.cuda.is_available():
                # Get available VRAM in GB
                device_idx = 0
                if ":" in self.device:
                    device_idx = int(self.device.split(":")[1])
                
                total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
                logger.info(f"Detected {total_vram:.1f}GB VRAM on {self.device}")
                
                # Use 1.7B if we have >= 12GB VRAM, otherwise 0.6B
                if total_vram >= 12:
                    logger.info(f"Using 1.7B model (sufficient VRAM)")
                    return large_model
                else:
                    logger.info(f"Using 0.6B model (limited VRAM)")
                    return small_model
            else:
                logger.info("No CUDA available, using 0.6B model")
                return small_model
        except Exception as e:
            logger.warning(f"VRAM detection failed: {e}, defaulting to 1.7B")
            return large_model
    
    async def load_model(self):
        """Load the Qwen3-TTS model."""
        if not QWEN_TTS_AVAILABLE:
            logger.warning("Running in mock mode - no model loaded")
            return
        
        model_to_load = self._detect_model()
        logger.info(f"Loading model from {model_to_load}...")
        start = time.time()
        
        self.model = Qwen3TTSModel.from_pretrained(
            model_to_load,
            device_map=self.device,
            dtype=self._get_torch_dtype(),
            attn_implementation="flash_attention_2",
        )
        
        logger.info(f"Model loaded in {time.time() - start:.2f}s")
        self.model_path = model_to_load  # Update for status reporting
    
    def _decode_audio_input(self, audio_b64: str) -> Tuple[np.ndarray, int]:
        """Decode base64 audio to numpy array."""
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
    
    def _compute_voice_id(self, audio_b64: str, ref_text: str, x_vector_only: bool) -> str:
        """Compute a unique ID for a voice based on audio and text."""
        # Hash the audio and text to create a unique ID
        hasher = hashlib.sha256()
        hasher.update(audio_b64.encode('utf-8')[:10000])  # First 10KB of base64
        hasher.update(ref_text.encode('utf-8'))
        hasher.update(str(x_vector_only).encode('utf-8'))
        return hasher.hexdigest()[:16]
    
    def get_cached_voice(self, voice_id: str) -> Optional[List[Any]]:
        """Get a cached voice prompt by ID."""
        return self.voice_cache.get(voice_id)
    
    def cache_voice(self, voice_id: str, prompt: List[Any]):
        """Cache a voice prompt."""
        self.voice_cache[voice_id] = prompt
        logger.info(f"Cached voice {voice_id}, total cached: {len(self.voice_cache)}")
    
    async def handle_init(
        self,
        websocket: WebSocketServerProtocol,
        session: VoiceSession,
        data: Dict[str, Any]
    ):
        """Initialize voice clone from reference audio."""
        try:
            ref_audio_b64 = data.get("ref_audio_base64")
            voice_id = data.get("voice_id")  # Optional pre-computed ID
            ref_text = data.get("ref_text", "")
            auto_transcribe = data.get("auto_transcribe", False)
            x_vector_only = data.get("x_vector_only", False)

            # Allow init with ONLY voice_id (no audio upload) to re-bind a cached voice prompt
            # to this *connection's* session after reconnects.
            if not ref_audio_b64 and voice_id:
                cached = self.get_cached_voice(voice_id)
                if cached:
                    logger.info(f"Using cached voice {voice_id} (voice_id-only init)")
                    session.voice_prompt = cached
                    session.voice_id = voice_id
                    await websocket.send(json.dumps({
                        "type": "ready",
                        "voice_loaded": True,
                        "voice_id": voice_id,
                        "ref_text": ref_text,
                        "cached": True
                    }))
                    return
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"voice_id not found in cache: {voice_id}"
                }))
                return
            
            if not ref_audio_b64:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "ref_audio_base64 is required (or provide cached voice_id)"
                }))
                return
            
            # Decode audio first (needed for transcription and voice ID)
            audio, sr = self._decode_audio_input(ref_audio_b64)
            logger.info(f"Reference audio: {len(audio)/sr:.2f}s at {sr}Hz")
            
            # Auto-transcribe if requested and no ref_text provided
            if auto_transcribe and not ref_text:
                if WHISPER_AVAILABLE:
                    logger.info("Auto-transcribing reference audio...")
                    ref_text = transcribe_audio(audio, sr)
                    logger.info(f"Transcription: {ref_text}")
                else:
                    logger.warning("Auto-transcribe requested but Whisper not available")
                    x_vector_only = True  # Fall back to x-vector only mode
            
            # Compute voice ID if not provided
            if not voice_id:
                voice_id = self._compute_voice_id(ref_audio_b64, ref_text, x_vector_only)
            
            # Check if already cached
            cached = self.get_cached_voice(voice_id)
            if cached:
                logger.info(f"Using cached voice {voice_id}")
                session.voice_prompt = cached
                session.voice_id = voice_id
                await websocket.send(json.dumps({
                    "type": "ready",
                    "voice_loaded": True,
                    "voice_id": voice_id,
                    "ref_text": ref_text,
                    "cached": True
                }))
                return
            
            if not x_vector_only and not ref_text:
                await websocket.send(json.dumps({
                    "type": "error", 
                    "message": "ref_text is required when x_vector_only is false"
                }))
                return
            
            if self.model:
                # Create voice clone prompt
                prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio=(audio, sr),
                    ref_text=ref_text if not x_vector_only else None,
                    x_vector_only_mode=x_vector_only,
                )
                session.voice_prompt = prompt_items
                session.sample_rate = 24000  # Qwen3-TTS output rate
                
                # Cache the voice prompt
                self.cache_voice(voice_id, prompt_items)
                session.voice_id = voice_id
                logger.info("Voice clone prompt created successfully")
            else:
                # Mock mode
                session.voice_prompt = [{"mock": True}]
                logger.info("Mock voice prompt created")
            
            await websocket.send(json.dumps({
                "type": "ready",
                "voice_loaded": True,
                "voice_id": voice_id,
                "ref_text": ref_text,
                "cached": False
            }))
            
        except Exception as e:
            logger.error(f"Error in handle_init: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def handle_generate(
        self,
        websocket: WebSocketServerProtocol,
        session: VoiceSession,
        data: Dict[str, Any]
    ):
        """Generate audio for text."""
        try:
            text = data.get("text", "")
            language = data.get("language", "Auto")
            
            if not text:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "text is required"
                }))
                return
            
            if not session.voice_prompt:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Voice not initialized. Send init message first."
                }))
                return
            
            session.is_generating = True
            session.cancel_requested = False
            
            await websocket.send(json.dumps({"type": "audio_start"}))
            
            start_time = time.time()
            
            if self.model:
                # Generate with Qwen3-TTS
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=session.voice_prompt,
                )
                
                audio = wavs[0]
                logger.info(f"Generated {len(audio)/sr:.2f}s audio in {time.time()-start_time:.2f}s")
                
            else:
                # Mock mode - generate silence
                sr = 24000
                duration = len(text) * 0.05  # ~50ms per character
                audio = np.zeros(int(sr * duration), dtype=np.float32)
                await asyncio.sleep(0.5)  # Simulate generation time
                logger.info(f"Mock generated {duration:.2f}s audio")
            
            # Convert to ulaw and chunk
            ulaw_audio = float32_to_ulaw(audio, sr, 8000)
            chunks = chunk_audio(ulaw_audio, 160)
            
            # Stream chunks with pacing
            chunk_duration = 0.020  # 20ms per chunk
            
            for i, chunk in enumerate(chunks):
                if session.cancel_requested:
                    logger.info("Generation cancelled")
                    break
                
                await websocket.send(chunk)
                
                # Pace output to roughly match real-time
                # (can be adjusted for faster-than-realtime streaming)
                if i < len(chunks) - 1:
                    await asyncio.sleep(chunk_duration * 0.5)  # 2x realtime
            
            await websocket.send(json.dumps({"type": "audio_end"}))
            
            session.is_generating = False
            
        except Exception as e:
            logger.error(f"Error in handle_generate: {e}")
            session.is_generating = False
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def handle_text_stream(
        self,
        websocket: WebSocketServerProtocol,
        session: VoiceSession,
        data: Dict[str, Any]
    ):
        """Handle streaming text input."""
        text = data.get("text", "")
        is_final = data.get("final", False)
        
        session.text_buffer += text
        
        if is_final and session.text_buffer:
            # Generate for accumulated text
            await self.handle_generate(
                websocket,
                session,
                {"text": session.text_buffer, "language": data.get("language", "Auto")}
            )
            session.text_buffer = ""
    
    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle a WebSocket connection."""
        session_id = str(id(websocket))
        session = VoiceSession()
        self.sessions[session_id] = session
        
        logger.info(f"New connection: {session_id}")
        
        try:
            await websocket.send(json.dumps({
                "type": "connected",
                "model": self.model_path,
                "mock_mode": not QWEN_TTS_AVAILABLE
            }))
            
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary messages not expected from client
                    continue
                
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "")
                    
                    if msg_type == "init":
                        await self.handle_init(websocket, session, data)
                    
                    elif msg_type == "generate":
                        await self.handle_generate(websocket, session, data)
                    
                    elif msg_type == "text":
                        await self.handle_text_stream(websocket, session, data)
                    
                    elif msg_type == "cancel":
                        session.cancel_requested = True
                        logger.info("Cancel requested")
                    
                    elif msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                    
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message[:100]}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {session_id}")
        finally:
            del self.sessions[session_id]
    
    async def run(self):
        """Start the WebSocket server."""
        await self.load_model()
        
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=50 * 1024 * 1024,  # 50MB max message size for audio
        ):
            await asyncio.Future()  # Run forever


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-TTS WebSocket Server")
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("QWEN3_TTS_MODEL", None),
        help="Model path or HuggingFace repo (default: auto-detect based on VRAM, "
             "1.7B for >=12GB, 0.6B otherwise)"
    )
    parser.add_argument(
        "--device", "-d",
        default=os.environ.get("QWEN3_TTS_DEVICE", "cuda:0"),
        help="Device (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("QWEN3_TTS_DTYPE", "bfloat16"),
        help="Model dtype (bfloat16, float16, float32)"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("QWEN3_TTS_HOST", "0.0.0.0"),
        help="Server host"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("QWEN3_TTS_PORT", "8765")),
        help="Server port"
    )
    
    args = parser.parse_args()
    
    server = Qwen3TTSServer(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        host=args.host,
        port=args.port,
    )
    
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
