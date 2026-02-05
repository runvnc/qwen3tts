#!/usr/bin/env python3
"""
Qwen3-TTS WebSocket Streaming Server - Optimized Version

Uses the dffdeeq fork's optimizations for ~3x speedup.
Requires: pip install -e /path/to/Qwen3-TTS-streaming
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

import websockets
from websockets.server import WebSocketServerProtocol

# Local imports
from audio_utils import float32_to_ulaw, chunk_audio
from voice_cache import VoiceCache
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

# Default chunk tokens
DEFAULT_EMIT_EVERY = int(os.environ.get('QWEN3_EMIT_EVERY', '8'))
DEFAULT_DECODE_WINDOW = int(os.environ.get('QWEN3_DECODE_WINDOW', '80'))
# PCM buffer size in samples (at 24kHz). 8 frames = ~13333 samples = ~555ms
DEFAULT_PCM_BUFFER_SAMPLES = int(os.environ.get('QWEN3_PCM_BUFFER_SAMPLES', '13333'))
DEFAULT_CROSSFADE_SAMPLES = int(os.environ.get('QWEN3_CROSSFADE_SAMPLES', '480'))  # 20ms at 24kHz

# Try to import qwen_tts (should be the fork)
try:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    QWEN_TTS_AVAILABLE = True
    logger.info("qwen_tts imported successfully")
except ImportError as e:
    logger.warning(f"qwen_tts import failed: {e}")
    import traceback
    traceback.print_exc()
    logger.warning("Running in mock mode")
    QWEN_TTS_AVAILABLE = False
    Qwen3TTSModel = None


class PCMCrossfadeBuffer:
    """Buffer that accumulates PCM chunks with crossfade at boundaries.
    
    The crossfade works by blending the END of one chunk with the START of the next.
    This creates a smooth transition without losing or duplicating audio.
    
    For sequential (non-overlapping) chunks:
    - Chunk 1: [A A A A T T]  (T = tail to be faded out)
    - Chunk 2: [H H B B B B]  (H = head to be faded in)
    - Output:  [A A A A X X B B B B]  (X = crossfaded region)
    
    The crossfade region replaces T and H with a blend.
    """
    
    def __init__(self, crossfade_samples: int = 480):
        self.crossfade_samples = crossfade_samples
        self.buffer = np.array([], dtype=np.float32)
        self.prev_tail = None  # Just the tail of the previous chunk for blending
    
    def add_chunk(self, chunk: np.ndarray):
        """Add a chunk, applying crossfade with previous chunk's tail."""
        if self.crossfade_samples <= 0:
            # No crossfade, just accumulate
            self.buffer = np.concatenate([self.buffer, chunk])
            return
        
        if len(chunk) < self.crossfade_samples:
            # Chunk too small for crossfade, just accumulate
            if self.prev_tail is not None:
                self.buffer = np.concatenate([self.buffer, self.prev_tail])
                self.prev_tail = None
            self.buffer = np.concatenate([self.buffer, chunk])
            return
        
        if self.prev_tail is None:
            # First chunk - output everything except tail, save tail for blending
            self.buffer = np.concatenate([self.buffer, chunk[:-self.crossfade_samples]])
            self.prev_tail = chunk[-self.crossfade_samples:].copy()
            return
        
        # We have a previous tail and a new chunk - crossfade them
        # Create crossfade curves
        fade_out = np.linspace(1.0, 0.0, self.crossfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, self.crossfade_samples, dtype=np.float32)
        
        # Blend: prev_tail fades out, chunk head fades in
        blended = self.prev_tail * fade_out + chunk[:self.crossfade_samples] * fade_in
        
        # Output: blended region + chunk middle (excluding head and tail)
        if len(chunk) > 2 * self.crossfade_samples:
            # Chunk has: head (blended) + middle + tail (save for next)
            middle = chunk[self.crossfade_samples:-self.crossfade_samples]
            self.buffer = np.concatenate([self.buffer, blended, middle])
            self.prev_tail = chunk[-self.crossfade_samples:].copy()
        else:
            # Chunk is small - head overlaps with tail region
            # Just output the blend and save what's left as tail
            remaining = len(chunk) - self.crossfade_samples
            if remaining > 0:
                self.buffer = np.concatenate([self.buffer, blended])
                self.prev_tail = chunk[-remaining:].copy() if remaining >= self.crossfade_samples else chunk[self.crossfade_samples:].copy()
            else:
                self.buffer = np.concatenate([self.buffer, blended])
                self.prev_tail = None
    
    def flush(self) -> np.ndarray:
        """Flush all remaining audio and return it."""
        if self.prev_tail is not None and len(self.prev_tail) > 0:
            self.buffer = np.concatenate([self.buffer, self.prev_tail])
            self.prev_tail = None
        
        result = self.buffer
        self.buffer = np.array([], dtype=np.float32)
        return result
    
    def get_and_clear(self, min_samples: int) -> Optional[np.ndarray]:
        """Get buffer contents if >= min_samples, clearing the buffer."""
        if len(self.buffer) >= min_samples:
            result = self.buffer
            self.buffer = np.array([], dtype=np.float32)
            return result
        return None


class Qwen3TTSServer:
    """WebSocket server for Qwen3-TTS streaming with fork optimizations."""

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
        
        logger.info(f"Server config: emit_every={DEFAULT_EMIT_EVERY}, decode_window={DEFAULT_DECODE_WINDOW}")

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
        """Load the Qwen3-TTS model with optimizations."""
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
            attn_implementation="sdpa",
        )

        logger.info(f"Model loaded in {time.time() - start:.2f}s")
        self.model_path = model_to_load

        # Enable streaming optimizations from the fork
        if hasattr(self.model, 'enable_streaming_optimizations'):
            logger.info("Enabling streaming optimizations...")
            start = time.time()
            self.model.enable_streaming_optimizations(
                decode_window_frames=DEFAULT_DECODE_WINDOW,
                use_compile=True,
                use_cuda_graphs=True,  # Enable for better TTFT
                compile_mode="reduce-overhead",
                use_fast_codebook=False,  # Disabled in fork too
                compile_codebook_predictor=True,
            )
            logger.info(f"Optimizations enabled in {time.time() - start:.2f}s")
        else:
            logger.warning("enable_streaming_optimizations not available - using standard qwen_tts?")

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

            # Quick cache check
            if ref_audio_b64 and not voice_id:
                quick_voice_id = self.voice_cache.compute_voice_id(ref_audio_b64, "", x_vector_only)
                cached = self.voice_cache.get(quick_voice_id)
                if cached:
                    profile.mark("cache_hit")
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

            # Handle voice_id-only init
            if not ref_audio_b64 and voice_id:
                cached = self.voice_cache.get(voice_id)
                if cached:
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

            # Auto-transcribe if needed
            if auto_transcribe and not ref_text:
                if WHISPER_AVAILABLE:
                    profile.mark("transcribing")
                    ref_text = transcribe_audio(audio, sr)
                    profile.mark("transcribed")
                else:
                    x_vector_only = True

            # Compute voice_id
            if not voice_id:
                voice_id = self.voice_cache.compute_voice_id(ref_audio_b64, "", x_vector_only)

            # Check cache again
            cached = self.voice_cache.get(voice_id)
            if cached:
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
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    async def handle_generate_stream(
        self,
        websocket: WebSocketServerProtocol,
        session: VoiceSession,
        data: Dict[str, Any]
    ):
        """Generate audio with fork's optimized streaming."""
        profile = profiler.start(f"stream_{id(websocket)}")
        
        try:
            text = data.get("text", "")
            language = data.get("language", "Auto")
            pcm_buffer_samples = data.get("pcm_buffer_samples", DEFAULT_PCM_BUFFER_SAMPLES)
            crossfade_samples = data.get("crossfade_samples", DEFAULT_CROSSFADE_SAMPLES)
            emit_every = data.get("emit_every_frames", DEFAULT_EMIT_EVERY)
            decode_window = data.get("decode_window_frames", DEFAULT_DECODE_WINDOW)

            profile.mark("params_parsed")
            logger.info(f"generate_stream: text='{text[:50]}...' emit={emit_every} crossfade={crossfade_samples}")

            if not text:
                await websocket.send(json.dumps({"type": "error", "message": "text required"}))
                return

            if not session.voice_prompt:
                await websocket.send(json.dumps({"type": "error", "message": "Voice not initialized"}))
                return

            # Check if fork's streaming method is available
            if not hasattr(self.model, 'stream_generate_voice_clone'):
                logger.error("stream_generate_voice_clone not available - need fork's qwen_tts")
                await websocket.send(json.dumps({"type": "error", "message": "Optimized streaming not available"}))
                return

            session.is_generating = True
            session.cancel_requested = False

            await websocket.send(json.dumps({"type": "audio_start"}))
            profile.mark("audio_start_sent")

            chunk_count = 0
            total_bytes = 0
            first_chunk_time = None
            pcm_sr = 24000  # Will be updated from first chunk
            t_start = time.time()
            
            # Use crossfade buffer for smooth chunk boundaries
            pcm_buffer = PCMCrossfadeBuffer(crossfade_samples=crossfade_samples)

            # Use fork's optimized streaming
            for pcm_chunk, sr in self.model.stream_generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=session.voice_prompt,
                emit_every_frames=emit_every,
                decode_window_frames=decode_window,
                overlap_samples=512,
            ):
                if session.cancel_requested:
                    logger.info("Generation cancelled")
                    break

                if first_chunk_time is None:
                    first_chunk_time = time.time() - t_start
                    profile.mark("first_audio_chunk")
                    logger.info(f"First chunk at {first_chunk_time*1000:.0f}ms")

                # Update sample rate from chunk
                pcm_sr = sr
                
                # Add chunk to crossfade buffer
                pcm_buffer.add_chunk(pcm_chunk)
                
                # Convert and send when buffer is large enough
                ready_pcm = pcm_buffer.get_and_clear(pcm_buffer_samples)
                if ready_pcm is not None:
                    ulaw_audio = float32_to_ulaw(ready_pcm, pcm_sr, 8000)
                    ulaw_chunks = chunk_audio(ulaw_audio, 160)
                    
                    for ulaw_chunk in ulaw_chunks:
                        await websocket.send(ulaw_chunk)
                        chunk_count += 1
                        total_bytes += len(ulaw_chunk)

                await asyncio.sleep(0)

            # Flush remaining PCM buffer
            final_pcm = pcm_buffer.flush()
            if len(final_pcm) > 0:
                ulaw_audio = float32_to_ulaw(final_pcm, pcm_sr, 8000)
                ulaw_chunks = chunk_audio(ulaw_audio, 160)
                
                for ulaw_chunk in ulaw_chunks:
                    await websocket.send(ulaw_chunk)
                    chunk_count += 1
                    total_bytes += len(ulaw_chunk)

            await websocket.send(json.dumps({"type": "audio_end"}))
            profile.mark("audio_end_sent")

            total_time = time.time() - t_start
            audio_duration = total_bytes / 8000
            rtf = total_time / audio_duration if audio_duration > 0 else 0
            logger.info(f"Streaming complete: {chunk_count} chunks, {audio_duration:.2f}s audio, RTF={rtf:.2f}")

            session.is_generating = False
            profiler.finish()

        except Exception as e:
            logger.error(f"Error in handle_generate_stream: {e}")
            import traceback
            traceback.print_exc()
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
                "optimized": hasattr(self.model, 'stream_generate_voice_clone') if self.model else False
            }))

            async for message in websocket:
                if isinstance(message, bytes):
                    continue

                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "init":
                        await self.handle_init(websocket, session, data)
                    elif msg_type == "generate_stream":
                        await self.handle_generate_stream(websocket, session, data)
                    elif msg_type == "cancel":
                        session.cancel_requested = True
                    elif msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message[:100]}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed: {session_id} ({e})")
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
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS WebSocket Server (Optimized)")
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
