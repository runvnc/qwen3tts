#!/usr/bin/env python3
"""
Qwen3-TTS WebSocket Streaming Server

Modular server with profiling for latency analysis.

Protocol:
- Connect to ws://host:port/ws
- Send JSON messages for control
- Receive binary audio chunks (ulaw 8kHz) or JSON status messages

Message Types (client -> server):
- {"type": "init", "ref_audio_base64": "...", "ref_text": "...", "language": "Auto"}
- {"type": "generate", "text": "Full text here"}
- {"type": "generate_stream", "text": "Full text here"}
- {"type": "cancel"}
- {"type": "ping"}

Message Types (server -> client):
- {"type": "connected", "model": "..."}
- {"type": "ready", "voice_loaded": true, "voice_id": "..."}
- {"type": "audio_start"}
- Binary data: raw ulaw 8kHz audio chunks
- {"type": "audio_end"}
- {"type": "error", "message": "..."}
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import websockets
from websockets.server import WebSocketServerProtocol

# Local imports
from audio_utils import float32_to_ulaw, chunk_audio
from voice_cache import VoiceCache, CachedVoice
from session import VoiceSession
from transcription import (
    WHISPER_AVAILABLE, 
    transcribe_audio, 
    decode_audio_input
)
from profiler import profiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noisy upstream logs
for _name in ("qwen_tts", "transformers", "transformers.generation"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Try to import qwen_tts
try:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    QWEN_TTS_AVAILABLE = True
except ImportError:
    logger.warning("qwen_tts not available, running in mock mode")
    QWEN_TTS_AVAILABLE = False
    Qwen3TTSModel = None

# Try to import streaming engine
try:
    from streaming_engine import Qwen3StreamingEngine
    STREAMING_ENGINE_AVAILABLE = True
except ImportError:
    logger.warning("streaming_engine not available, true streaming disabled")
    STREAMING_ENGINE_AVAILABLE = False


class Qwen3TTSServer:
    """WebSocket server for Qwen3-TTS streaming."""

    def __init__(
        self,
        model_path: str = None,
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
        self.voice_cache = VoiceCache(max_voices=50)
        self.streaming_engine = None

    def _get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype.lower(), torch.bfloat16)

    def _detect_model(self) -> str:
        """Auto-detect which model to use based on available VRAM."""
        if self.model_path:
            return self.model_path

        large_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        small_model = self.fallback_model

        try:
            if torch.cuda.is_available():
                device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
                total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
                logger.info(f"Detected {total_vram:.1f}GB VRAM on {self.device}")
                return large_model if total_vram >= 12 else small_model
        except Exception as e:
            logger.warning(f"VRAM detection failed: {e}")
        
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

        # Silence pad_token_id warning
        try:
            inner = getattr(self.model, "model", None)
            gc = getattr(inner, "generation_config", None)
            cfg = getattr(inner, "config", None)
            eos_id = getattr(cfg, "eos_token_id", None)
            if gc and getattr(gc, "pad_token_id", None) is None and eos_id:
                gc.pad_token_id = eos_id
        except Exception:
            pass

        logger.info(f"Model loaded in {time.time() - start:.2f}s")
        self.model_path = model_to_load

        # Initialize streaming engine
        if STREAMING_ENGINE_AVAILABLE and self.model:
            self.streaming_engine = Qwen3StreamingEngine(self.model)
            logger.info("Streaming engine initialized")

    async def handle_init(
        self,
        websocket: WebSocketServerProtocol,
        session: VoiceSession,
        data: Dict[str, Any]
    ):
        """Initialize voice clone from reference audio."""
        profile = profiler.start(f"init_{id(websocket)}")
        
        try:
            ref_audio_b64 = data.get("ref_audio_base64")
            voice_id = data.get("voice_id")
            ref_text = data.get("ref_text", "")
            auto_transcribe = data.get("auto_transcribe", False)
            x_vector_only = data.get("x_vector_only", False)

            profile.mark("params_parsed")

            # Quick cache check using audio hash (before decoding)
            if ref_audio_b64 and not voice_id:
                quick_voice_id = self.voice_cache.compute_voice_id(ref_audio_b64, "", x_vector_only)
                cached = self.voice_cache.get(quick_voice_id)
                if cached:
                    profile.mark("cache_hit")
                    logger.info(f"Voice found in cache: {quick_voice_id}")
                    session.voice_prompt = cached.prompt_items
                    session.voice_id = quick_voice_id
                    await websocket.send(json.dumps({
                        "type": "ready",
                        "voice_loaded": True,
                        "voice_id": quick_voice_id,
                        "cached": True
                    }))
                    profiler.finish()
                    return

            # Handle voice_id-only init (rebind cached voice to this session)
            if not ref_audio_b64 and voice_id:
                cached = self.voice_cache.get(voice_id)
                if cached:
                    profile.mark("voice_id_rebind")
                    session.voice_prompt = cached.prompt_items
                    session.voice_id = voice_id
                    await websocket.send(json.dumps({
                        "type": "ready",
                        "voice_loaded": True,
                        "voice_id": voice_id,
                        "cached": True
                    }))
                    profiler.finish()
                    return
                
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"voice_id not found: {voice_id}"
                }))
                return

            if not ref_audio_b64:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "ref_audio_base64 is required"
                }))
                return

            # Decode audio
            profile.mark("decoding_audio")
            audio, sr = decode_audio_input(ref_audio_b64)
            profile.mark("audio_decoded")
            logger.info(f"Reference audio: {len(audio)/sr:.2f}s at {sr}Hz")

            # Auto-transcribe if needed
            if auto_transcribe and not ref_text:
                if WHISPER_AVAILABLE:
                    profile.mark("transcribing")
                    ref_text = transcribe_audio(audio, sr)
                    profile.mark("transcribed")
                    logger.info(f"Transcription: {ref_text}")
                else:
                    x_vector_only = True

            # Compute voice_id
            if not voice_id:
                voice_id = self.voice_cache.compute_voice_id(ref_audio_b64, "", x_vector_only)

            # Check cache again (with computed voice_id)
            cached = self.voice_cache.get(voice_id)
            if cached:
                profile.mark("cache_hit_after_decode")
                session.voice_prompt = cached.prompt_items
                session.voice_id = voice_id
                await websocket.send(json.dumps({
                    "type": "ready",
                    "voice_loaded": True,
                    "voice_id": voice_id,
                    "ref_text": ref_text,
                    "cached": True
                }))
                profiler.finish()
                return

            if not x_vector_only and not ref_text:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "ref_text required when x_vector_only is false"
                }))
                return

            # Create voice clone prompt
            profile.mark("creating_voice_prompt")
            if self.model:
                prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio=(audio, sr),
                    ref_text=ref_text if not x_vector_only else None,
                    x_vector_only_mode=x_vector_only,
                )
                session.voice_prompt = prompt_items
                session.sample_rate = 24000
                profile.mark("voice_prompt_created")

                # Cache the voice
                self.voice_cache.put(voice_id, prompt_items, ref_text, x_vector_only)
                session.voice_id = voice_id
            else:
                session.voice_prompt = [{"mock": True}]

            await websocket.send(json.dumps({
                "type": "ready",
                "voice_loaded": True,
                "voice_id": voice_id,
                "ref_text": ref_text,
                "cached": False
            }))
            
            profiler.finish()

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
        """Generate audio for text (non-streaming)."""
        profile = profiler.start(f"gen_{id(websocket)}")
        
        try:
            text = data.get("text", "")
            language = data.get("language", "Auto")

            profile.mark("params_parsed")
            logger.info(f"generate: text='{text[:50]}...' voice_id={session.voice_id}")

            if not text:
                await websocket.send(json.dumps({"type": "error", "message": "text required"}))
                return

            if not session.voice_prompt:
                await websocket.send(json.dumps({"type": "error", "message": "Voice not initialized"}))
                return

            session.is_generating = True
            session.cancel_requested = False

            if self.model:
                profile.mark("generating")
                with profiler.measure("model_generate"):
                    wavs, sr = self.model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=session.voice_prompt,
                    )
                profile.mark("generated")

                audio = wavs[0]
                logger.info(f"Generated {len(audio)/sr:.2f}s audio")
            else:
                sr = 24000
                duration = len(text) * 0.05
                audio = np.zeros(int(sr * duration), dtype=np.float32)
                await asyncio.sleep(0.5)

            # Convert and stream
            profile.mark("converting")
            ulaw_audio = float32_to_ulaw(audio, sr, 8000)
            chunks = chunk_audio(ulaw_audio, 160)
            profile.mark("converted")

            for i, chunk in enumerate(chunks):
                if session.cancel_requested:
                    break
                await websocket.send(chunk)
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.010)

            await websocket.send(json.dumps({"type": "audio_end"}))
            session.is_generating = False
            profiler.finish()

        except Exception as e:
            logger.error(f"Error in handle_generate: {e}")
            session.is_generating = False
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    async def handle_generate_stream(
        self,
        websocket: WebSocketServerProtocol,
        session: VoiceSession,
        data: Dict[str, Any]
    ):
        """Generate audio with true streaming."""
        profile = profiler.start(f"stream_{id(websocket)}")
        
        try:
            text = data.get("text", "")
            language = data.get("language", "Auto")
            initial_chunk_tokens = data.get("initial_chunk_tokens", 8)
            stream_chunk_tokens = data.get("stream_chunk_tokens", 8)

            profile.mark("params_parsed")
            logger.info(f"generate_stream: text='{text[:50]}...' initial={initial_chunk_tokens} stream={stream_chunk_tokens}")

            if not text:
                await websocket.send(json.dumps({"type": "error", "message": "text required"}))
                return

            if not session.voice_prompt:
                await websocket.send(json.dumps({"type": "error", "message": "Voice not initialized"}))
                return

            if not self.streaming_engine:
                logger.warning("Streaming engine not available, falling back")
                await self.handle_generate(websocket, session, data)
                return

            session.is_generating = True
            session.cancel_requested = False

            await websocket.send(json.dumps({"type": "audio_start"}))
            profile.mark("audio_start_sent")

            chunk_count = 0
            total_bytes = 0
            first_chunk_time = None

            profile.mark("streaming_start")
            async for pcm_chunk in self.streaming_engine.generate_stream(
                text=text,
                voice_clone_prompt=session.voice_prompt,
                language=language,
                initial_chunk_tokens=initial_chunk_tokens,
                stream_chunk_tokens=stream_chunk_tokens,
            ):
                if session.cancel_requested:
                    logger.info("Generation cancelled")
                    break

                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    profile.mark("first_audio_chunk")
                    logger.info(f"First audio chunk at +{profile.timings.get('first_audio_chunk', 0):.0f}ms")

                # Convert to ulaw
                audio = np.frombuffer(pcm_chunk, dtype=np.float32)
                ulaw_audio = float32_to_ulaw(audio, 24000, 8000)
                ulaw_chunks = chunk_audio(ulaw_audio, 160)

                for ulaw_chunk in ulaw_chunks:
                    await websocket.send(ulaw_chunk)
                    chunk_count += 1
                    total_bytes += len(ulaw_chunk)

                await asyncio.sleep(0)

            await websocket.send(json.dumps({"type": "audio_end"}))
            profile.mark("audio_end_sent")

            audio_duration = total_bytes / 8000
            logger.info(f"Streaming complete: {chunk_count} chunks, {audio_duration:.2f}s audio")

            session.is_generating = False
            profiler.finish()

        except Exception as e:
            logger.error(f"Error in handle_generate_stream: {e}")
            session.is_generating = False
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))

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
                    continue

                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "init":
                        await self.handle_init(websocket, session, data)
                    elif msg_type == "generate":
                        await self.handle_generate(websocket, session, data)
                    elif msg_type == "generate_stream":
                        await self.handle_generate_stream(websocket, session, data)
                    elif msg_type == "cancel":
                        session.cancel_requested = True
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
            max_size=50 * 1024 * 1024,
        ):
            await asyncio.Future()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS WebSocket Server")
    parser.add_argument("--model", "-m", default=os.environ.get("QWEN3_TTS_MODEL"))
    parser.add_argument("--device", "-d", default=os.environ.get("QWEN3_TTS_DEVICE", "cuda:0"))
    parser.add_argument("--dtype", default=os.environ.get("QWEN3_TTS_DTYPE", "bfloat16"))
    parser.add_argument("--host", default=os.environ.get("QWEN3_TTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("QWEN3_TTS_PORT", "8765")))

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
