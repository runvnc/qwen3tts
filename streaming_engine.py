"""
Qwen3-TTS Streaming Engine with torch.compile and CUDA optimizations.

True streaming audio generation with detailed profiling.
"""

import time
import torch
import numpy as np
import logging
from typing import Any, Optional, AsyncGenerator, Dict, List

logger = logging.getLogger(__name__)

# Check for torch.compile availability
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')
logger.info(f"torch.compile available: {TORCH_COMPILE_AVAILABLE}")


class EmbeddingCache:
    """Cache for pre-computed embeddings per voice."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, voice_id: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(voice_id)
    
    def put(self, voice_id: str, embeddings: Dict[str, Any]):
        self._cache[voice_id] = embeddings
        logger.info(f"Cached embeddings for voice {voice_id}")
    
    def has(self, voice_id: str) -> bool:
        return voice_id in self._cache


class Qwen3StreamingEngine:
    """Streaming audio generation engine for Qwen3-TTS with optimizations."""
    
    def __init__(self, model_wrapper):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.talker = self.model.talker
        self.tokenizer = self.model.speech_tokenizer
        self.device = self.model.device
        
        # Embedding cache
        self.embedding_cache = EmbeddingCache()
        
        # Pre-compute constant embeddings
        self._tts_const_embeds = None
        self._precompute_constants()
        
        # Compile the talker for faster inference
        self._compiled_talker = None
        self._compile_model()
        
        try:
            self.upsample_rate = self.tokenizer.get_decode_upsample_rate()
        except:
            self.upsample_rate = 2000
        
        # Chunk sizes - can be tuned
        self.initial_chunk_tokens = 4
        self.stream_chunk_tokens = 4
        self.context_size = 38
        self.crossfade_samples = 240
        
        logger.info(f"StreamingEngine: initial={self.initial_chunk_tokens}, stream={self.stream_chunk_tokens}")
    
    def _compile_model(self):
        """Compile the talker model for faster inference."""
        if not TORCH_COMPILE_AVAILABLE:
            logger.warning("torch.compile not available, skipping compilation")
            return
        
        try:
            logger.info("Compiling talker model with torch.compile...")
            t0 = time.time()
            
            # Use reduce-overhead mode for best latency
            # This enables CUDA graphs internally where possible
            self._compiled_talker = torch.compile(
                self.talker,
                mode="reduce-overhead",  # Optimizes for latency
                fullgraph=False,  # Allow graph breaks for dynamic control flow
            )
            
            logger.info(f"Model compiled in {time.time()-t0:.2f}s")
            
            # Warmup the compiled model
            self._warmup_compiled_model()
            
        except Exception as e:
            logger.error(f"torch.compile failed: {e}")
            logger.info("Falling back to uncompiled model")
            self._compiled_talker = None
    
    def _warmup_compiled_model(self):
        """Warmup the compiled model to trigger JIT compilation."""
        if self._compiled_talker is None:
            return
        
        logger.info("Warming up compiled model...")
        t0 = time.time()
        
        try:
            with torch.no_grad():
                # Create dummy inputs matching expected shapes
                batch_size = 1
                seq_len = 10
                hidden_size = self.talker.config.hidden_size
                
                dummy_embeds = torch.randn(
                    batch_size, seq_len, hidden_size,
                    device=self.device, dtype=torch.bfloat16
                )
                
                # Get TTS embeds for warmup
                tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_tts_embeds()
                
                # Run a few forward passes to trigger compilation
                for i in range(3):
                    _ = self._compiled_talker.forward(
                        input_ids=None,
                        inputs_embeds=dummy_embeds,
                        past_key_values=None,
                        use_cache=True,
                        past_hidden=None,
                        trailing_text_hidden=dummy_embeds[:, :5, :],
                        tts_pad_embed=tts_pad_embed,
                        generation_step=0,
                        subtalker_dosample=True
                    )
                
                # Sync to ensure compilation is complete
                torch.cuda.synchronize()
                
            logger.info(f"Warmup complete in {time.time()-t0:.2f}s")
            
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self._compiled_talker = None
    
    def _precompute_constants(self):
        """Pre-compute TTS constant embeddings (BOS, EOS, PAD)."""
        t0 = time.time()
        with torch.no_grad():
            tts_token_ids = torch.tensor([
                [self.model.config.tts_bos_token_id, 
                 self.model.config.tts_eos_token_id, 
                 self.model.config.tts_pad_token_id]
            ], device=self.device)
            self._tts_const_embeds = self.talker.text_projection(
                self.talker.get_text_embeddings()(tts_token_ids)
            )
        logger.info(f"Pre-computed TTS constants in {(time.time()-t0)*1000:.1f}ms")
    
    def _get_tts_embeds(self):
        """Get cached TTS constant embeddings."""
        return self._tts_const_embeds.chunk(3, dim=1)
    
    def precompute_voice_embeddings(
        self,
        voice_id: str,
        voice_clone_prompt: list,
        language: str = "Auto"
    ) -> Dict[str, Any]:
        """Pre-compute and cache embeddings for a voice."""
        t0 = time.time()
        
        with torch.no_grad():
            voice_clone_prompt_dict = self.wrapper._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
            ref_code = voice_clone_prompt_dict["ref_code"][0].to(self.device)
            
            voice_clone_spk_embeds = self.model.generate_speaker_prompt(voice_clone_prompt_dict)
            speaker_embed = voice_clone_spk_embeds[0]
            
            language_id = self.model.config.talker_config.codec_language_id.get(language.lower())
            
            codec_prefill_ids = [
                self.model.config.talker_config.codec_think_id if language_id 
                else self.model.config.talker_config.codec_nothink_id,
                self.model.config.talker_config.codec_think_bos_id
            ]
            if language_id:
                codec_prefill_ids.append(language_id)
            codec_prefill_ids.append(self.model.config.talker_config.codec_think_eos_id)
            
            codec_input_embedding_0 = self.talker.get_input_embeddings()(
                torch.tensor([codec_prefill_ids], device=self.device)
            )
            codec_input_embedding_1 = self.talker.get_input_embeddings()(
                torch.tensor([[
                    self.model.config.talker_config.codec_pad_id, 
                    self.model.config.talker_config.codec_bos_id
                ]], device=self.device)
            )
            codec_input_embedding = torch.cat([
                codec_input_embedding_0, 
                speaker_embed.view(1, 1, -1), 
                codec_input_embedding_1
            ], dim=1)
            
            ref_texts = [it.ref_text for it in voice_clone_prompt]
            ref_ids = [
                self.wrapper._tokenize_texts([self.wrapper._build_ref_text(rt)])[0] if rt else None 
                for rt in ref_texts
            ]
            
            embeddings = {
                'voice_clone_prompt_dict': voice_clone_prompt_dict,
                'ref_code': ref_code,
                'speaker_embed': speaker_embed,
                'codec_input_embedding': codec_input_embedding,
                'language_id': language_id,
                'ref_ids': ref_ids,
                'icl_mode': voice_clone_prompt_dict["icl_mode"][0],
            }
            
            self.embedding_cache.put(voice_id, embeddings)
            
            logger.info(f"Pre-computed embeddings for {voice_id} in {(time.time()-t0)*1000:.1f}ms")
            
            return embeddings
    
    async def generate_stream(
        self,
        text: str,
        voice_clone_prompt: list,
        language: str = "Auto",
        voice_id: str = None,
        initial_chunk_tokens: Optional[int] = None,
        stream_chunk_tokens: Optional[int] = None,
        temperature: float = 0.8,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio in a streaming fashion."""
        
        initial_tokens = initial_chunk_tokens or self.initial_chunk_tokens
        stream_tokens = stream_chunk_tokens or self.stream_chunk_tokens
        
        t_start = time.time()
        token_times = []
        
        # Use compiled model if available
        talker = self._compiled_talker if self._compiled_talker is not None else self.talker
        using_compiled = self._compiled_talker is not None
        
        def mark(name):
            elapsed = (time.time() - t_start) * 1000
            logger.info(f"[streaming_engine] {name}: +{elapsed:.0f}ms")
            return elapsed
        
        mark("start")
        logger.info(f"[streaming_engine] using_compiled={using_compiled}")
        
        with torch.no_grad():
            # Check embedding cache
            cached_embeds = self.embedding_cache.get(voice_id) if voice_id else None
            
            if cached_embeds:
                mark("cache_hit")
                voice_clone_prompt_dict = cached_embeds['voice_clone_prompt_dict']
                ref_code = cached_embeds['ref_code']
                codec_input_embedding = cached_embeds['codec_input_embedding']
                language_id = cached_embeds['language_id']
                ref_ids = cached_embeds['ref_ids']
                icl_mode = cached_embeds['icl_mode']
            else:
                mark("cache_miss")
                voice_clone_prompt_dict = self.wrapper._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
                ref_code = voice_clone_prompt_dict["ref_code"][0].to(self.device)
                
                voice_clone_spk_embeds = self.model.generate_speaker_prompt(voice_clone_prompt_dict)
                speaker_embed = voice_clone_spk_embeds[0]
                
                language_id = self.model.config.talker_config.codec_language_id.get(language.lower())
                
                codec_prefill_ids = [
                    self.model.config.talker_config.codec_think_id if language_id 
                    else self.model.config.talker_config.codec_nothink_id,
                    self.model.config.talker_config.codec_think_bos_id
                ]
                if language_id:
                    codec_prefill_ids.append(language_id)
                codec_prefill_ids.append(self.model.config.talker_config.codec_think_eos_id)
                
                codec_input_embedding_0 = self.talker.get_input_embeddings()(
                    torch.tensor([codec_prefill_ids], device=self.device)
                )
                codec_input_embedding_1 = self.talker.get_input_embeddings()(
                    torch.tensor([[
                        self.model.config.talker_config.codec_pad_id, 
                        self.model.config.talker_config.codec_bos_id
                    ]], device=self.device)
                )
                codec_input_embedding = torch.cat([
                    codec_input_embedding_0, 
                    speaker_embed.view(1, 1, -1), 
                    codec_input_embedding_1
                ], dim=1)
                
                ref_texts = [it.ref_text for it in voice_clone_prompt]
                ref_ids = [
                    self.wrapper._tokenize_texts([self.wrapper._build_ref_text(rt)])[0] if rt else None 
                    for rt in ref_texts
                ]
                icl_mode = voice_clone_prompt_dict["icl_mode"][0]
                mark("embeddings_computed")
            
            # Tokenize input text
            input_id = self.wrapper._tokenize_texts([self.wrapper._build_assistant_text(text)])[0]
            mark("text_tokenized")
            
            # Get TTS constant embeddings
            tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_tts_embeds()
            
            # Build talker input embeddings
            role_start_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, :3])
            )
            _talker_input_embed = torch.cat((
                tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), 
                tts_bos_embed
            ), dim=1) + codec_input_embedding[:, :-1]
            talker_input_embed = torch.cat((role_start_embed, _talker_input_embed), dim=1)
            mark("talker_embed_built")
            
            # Handle ICL mode
            if icl_mode:
                icl_input_embed, trailing_text_hidden = self.model.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[0][:, 3:-2],
                    ref_code=ref_code,
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=False
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
                mark("icl_prompt_generated")
            else:
                talker_input_embed = torch.cat([
                    talker_input_embed,
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 3:4])
                    ) + codec_input_embedding[:, -1:]
                ], dim=1)
                trailing_text_hidden = torch.cat((
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 4:-5])
                    ),
                    tts_eos_embed
                ), dim=1)
                mark("non_icl_embed_built")
            
            # Generation loop state
            past_key_values = None
            past_hidden = None
            generation_step = 0
            current_input_ids = None
            current_inputs_embeds = talker_input_embed
            
            # Audio state
            history_codes = ref_code
            chunk_buffer = []
            prev_chunk_tail = None
            is_first_chunk = True
            chunk_count = 0
            
            eos_token_id = self.model.config.talker_config.codec_eos_token_id
            
            def sample_token(logits):
                temp = max(temperature, 1e-5)
                probs = torch.softmax(logits / temp, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            
            mark("ready_to_generate")
            
            # Sync before generation loop
            torch.cuda.synchronize()
            
            # Main generation loop
            for i in range(max_new_tokens):
                t_forward_start = time.time()
                
                outputs = talker.forward(
                    input_ids=current_input_ids,
                    inputs_embeds=current_inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    past_hidden=past_hidden,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    generation_step=generation_step,
                    subtalker_dosample=True
                )
                
                # Sync to get accurate timing
                torch.cuda.synchronize()
                
                t_forward_end = time.time()
                forward_ms = (t_forward_end - t_forward_start) * 1000
                
                if i == 0:
                    mark("first_forward_pass")
                    logger.info(f"[streaming_engine] first forward took {forward_ms:.1f}ms")
                
                if is_first_chunk:
                    token_times.append(forward_ms)
                    if len(token_times) <= initial_tokens + 2:
                        logger.info(f"[streaming_engine] token {i}: {forward_ms:.1f}ms (total: {sum(token_times):.1f}ms)")
                
                past_key_values = outputs.past_key_values
                past_hidden = outputs.past_hidden
                generation_step = outputs.generation_step
                current_inputs_embeds = None
                
                next_token = sample_token(outputs.logits[:, -1, :])
                current_input_ids = next_token
                
                if next_token.item() == eos_token_id:
                    break
                
                frame_codes = outputs.hidden_states[1] if len(outputs.hidden_states) > 1 else None
                if frame_codes is not None:
                    code_frame = frame_codes.squeeze(0)
                    chunk_buffer.append(code_frame)
                    
                    threshold = initial_tokens if is_first_chunk else stream_tokens
                    
                    if len(chunk_buffer) >= threshold:
                        context = history_codes[-self.context_size:] if len(history_codes) > self.context_size else history_codes
                        to_decode = torch.stack(chunk_buffer, dim=0)
                        
                        if is_first_chunk:
                            mark("decoding_first_chunk")
                            logger.info(f"[streaming_engine] token generation took {sum(token_times):.1f}ms for {len(token_times)} tokens")
                        
                        t_decode_start = time.time()
                        audio_chunk = self._decode_with_context(to_decode, context)
                        t_decode_end = time.time()
                        
                        if is_first_chunk:
                            mark("first_chunk_decoded")
                            logger.info(f"[streaming_engine] decode took {(t_decode_end-t_decode_start)*1000:.1f}ms")
                        
                        if prev_chunk_tail is not None:
                            audio_chunk = self._apply_crossfade(prev_chunk_tail, audio_chunk)
                        
                        fade_len = self.crossfade_samples
                        if len(audio_chunk) > fade_len:
                            prev_chunk_tail = audio_chunk[-fade_len:].copy()
                            yield audio_chunk[:-fade_len].tobytes()
                        else:
                            yield audio_chunk.tobytes()
                            prev_chunk_tail = None
                        
                        history_codes = torch.cat([history_codes, to_decode], dim=0)
                        chunk_buffer = []
                        is_first_chunk = False
                        chunk_count += 1
            
            # Flush remaining buffer
            if chunk_buffer:
                context = history_codes[-self.context_size:] if len(history_codes) > self.context_size else history_codes
                audio_chunk = self._decode_with_context(torch.stack(chunk_buffer, dim=0), context)
                if prev_chunk_tail is not None:
                    audio_chunk = self._apply_crossfade(prev_chunk_tail, audio_chunk)
                yield audio_chunk.tobytes()
                chunk_count += 1
            
            total_ms = (time.time() - t_start) * 1000
            mark("generation_complete")
            logger.info(f"[streaming_engine] Generated {chunk_count} chunks in {total_ms:.0f}ms")
    
    def _apply_crossfade(self, tail: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Apply linear crossfade between chunks."""
        fade_len = len(tail)
        if len(current) < fade_len:
            return current
        
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = 1 - fade_in
        current[:fade_len] = tail * fade_out + current[:fade_len] * fade_in
        return current
    
    def _decode_with_context(self, new_codes: torch.Tensor, context_codes: torch.Tensor) -> np.ndarray:
        """Decode audio codes with context for better quality."""
        with torch.no_grad():
            wav_ctx_all, _ = self.tokenizer.decode([{"audio_codes": context_codes}])
            trim_samples = len(wav_ctx_all[0])
            
            full_codes = torch.cat([context_codes, new_codes], dim=0)
            wavs, sr = self.tokenizer.decode([{"audio_codes": full_codes}])
            wav = wavs[0]
            
            if trim_samples < len(wav):
                return wav[trim_samples:].astype(np.float32)
            return wav.astype(np.float32)
