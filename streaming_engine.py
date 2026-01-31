"""
Qwen3-TTS Streaming Engine

Adapted from CloudWells/qwen3-tts-realtime-streaming for true streaming audio generation.
Uses manual KV-cache stepping to yield audio chunks as tokens are generated.

Key features:
- Step-by-step inference with KV-caching
- Configurable initial buffer (can be 0 for lowest latency)
- Context-aware decoding for voice stability
- Cross-fading between chunks to eliminate clicks
"""

import torch
import numpy as np
import logging
from typing import Any, Optional, AsyncGenerator, Dict

logger = logging.getLogger(__name__)


class Qwen3StreamingEngine:
    """Streaming audio generation engine for Qwen3-TTS."""
    
    def __init__(self, model_wrapper):
        """
        Initialize the streaming engine.
        
        Args:
            model_wrapper: The Qwen3TTSModel instance
        """
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.talker = self.model.talker
        self.tokenizer = self.model.speech_tokenizer
        self.device = self.model.device
        
        # Get upsample rate for audio conversion
        try:
            self.upsample_rate = self.tokenizer.get_decode_upsample_rate()
        except:
            self.upsample_rate = 2000
        
        # Default chunk sizes (in tokens, ~83ms per token at 12Hz)
        self.initial_chunk_tokens = 6  # ~0.5s - reduced for lower latency
        self.stream_chunk_tokens = 4   # ~0.33s per subsequent chunk (smaller = smoother)
        self.context_size = 38         # ~3s context for decoding stability
        self.crossfade_samples = 240   # ~10ms at 24kHz
        self.min_buffer_chunks = 2    # Minimum chunks to buffer before yielding (smooths playback)
        
        logger.info(f"StreamingEngine initialized: initial_chunk={self.initial_chunk_tokens}, "
                   f"stream_chunk={self.stream_chunk_tokens}, context={self.context_size}")
    
    async def generate_stream(
        self,
        text: str,
        voice_clone_prompt: list,
        language: str = "Auto",
        initial_chunk_tokens: Optional[int] = None,
        stream_chunk_tokens: Optional[int] = None,
        temperature: float = 0.8,
        max_new_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio in a streaming fashion, yielding chunks as they're produced.
        
        Args:
            text: Text to synthesize
            voice_clone_prompt: Pre-computed voice clone prompt items
            language: Target language
            initial_chunk_tokens: Tokens to buffer before first output (default: self.initial_chunk_tokens)
            stream_chunk_tokens: Tokens per subsequent chunk (default: self.stream_chunk_tokens)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            bytes: Float32 PCM audio chunks at 24kHz
        """
        initial_tokens = initial_chunk_tokens or self.initial_chunk_tokens
        stream_tokens = stream_chunk_tokens or self.stream_chunk_tokens
        
        with torch.no_grad():
            # Convert prompt items to internal format
            voice_clone_prompt_dict = self.wrapper._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
            ref_code = voice_clone_prompt_dict["ref_code"][0]
            
            # Tokenize input text
            input_id = self.wrapper._tokenize_texts([self.wrapper._build_assistant_text(text)])[0]
            
            # Get reference text IDs if available
            ref_texts = [it.ref_text for it in voice_clone_prompt]
            ref_ids = [
                self.wrapper._tokenize_texts([self.wrapper._build_ref_text(rt)])[0] if rt else None 
                for rt in ref_texts
            ]
            
            # Generate speaker embedding
            voice_clone_spk_embeds = self.model.generate_speaker_prompt(voice_clone_prompt_dict)
            speaker_embed = voice_clone_spk_embeds[0]
            
            # Get language ID
            language_id = self.model.config.talker_config.codec_language_id.get(language.lower())
            
            # Build TTS token embeddings
            tts_token_ids = torch.tensor([
                [self.model.config.tts_bos_token_id, 
                 self.model.config.tts_eos_token_id, 
                 self.model.config.tts_pad_token_id]
            ], device=self.device)
            tts_const_embeds = self.talker.text_projection(
                self.talker.get_text_embeddings()(tts_token_ids)
            )
            tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_const_embeds.chunk(3, dim=1)
            
            # Build codec prefill sequence
            codec_prefill_ids = [
                self.model.config.talker_config.codec_think_id if language_id 
                else self.model.config.talker_config.codec_nothink_id,
                self.model.config.talker_config.codec_think_bos_id
            ]
            if language_id:
                codec_prefill_ids.append(language_id)
            codec_prefill_ids.append(self.model.config.talker_config.codec_think_eos_id)
            
            # Build input embeddings
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
            
            # Build talker input embeddings
            role_start_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, :3])
            )
            _talker_input_embed = torch.cat((
                tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), 
                tts_bos_embed
            ), dim=1) + codec_input_embedding[:, :-1]
            talker_input_embed = torch.cat((role_start_embed, _talker_input_embed), dim=1)
            
            # Handle ICL mode vs non-ICL mode
            if voice_clone_prompt_dict["icl_mode"][0]:
                icl_input_embed, trailing_text_hidden = self.model.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[0][:, 3:-2],
                    ref_code=ref_code.to(self.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=False
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
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
            
            # Generation loop state
            past_key_values = None
            past_hidden = None
            generation_step = 0
            current_input_ids = None
            current_inputs_embeds = talker_input_embed
            
            # Audio state
            history_codes = ref_code.to(self.device)
            chunk_buffer = []
            prev_chunk_tail = None
            is_first_chunk = True
            
            eos_token_id = self.model.config.talker_config.codec_eos_token_id
            
            def sample_token(logits):
                """Sample next token with temperature."""
                temp = max(temperature, 1e-5)
                probs = torch.softmax(logits / temp, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            
            # Main generation loop
            for i in range(max_new_tokens):
                # Forward pass
                outputs = self.talker.forward(
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
                
                # Update state
                past_key_values = outputs.past_key_values
                past_hidden = outputs.past_hidden
                generation_step = outputs.generation_step
                current_inputs_embeds = None
                
                # Sample next token
                next_token = sample_token(outputs.logits[:, -1, :])
                current_input_ids = next_token
                
                # Check for end of sequence
                if next_token.item() == eos_token_id:
                    break
                
                # Extract audio frame codes
                frame_codes = outputs.hidden_states[1] if len(outputs.hidden_states) > 1 else None
                if frame_codes is not None:
                    code_frame = frame_codes.squeeze(0)
                    chunk_buffer.append(code_frame)
                    
                    # Determine chunk threshold
                    threshold = initial_tokens if is_first_chunk else stream_tokens
                    
                    if len(chunk_buffer) >= threshold:
                        # Get context for decoding
                        context = history_codes[-self.context_size:] if len(history_codes) > self.context_size else history_codes
                        
                        # Decode chunk with context
                        to_decode = torch.stack(chunk_buffer, dim=0)
                        audio_chunk = self._decode_with_context(to_decode, context)
                        
                        # Apply crossfade if not first chunk
                        if prev_chunk_tail is not None:
                            audio_chunk = self._apply_crossfade(prev_chunk_tail, audio_chunk)
                        
                        # Save tail for next crossfade
                        fade_len = self.crossfade_samples
                        if len(audio_chunk) > fade_len:
                            prev_chunk_tail = audio_chunk[-fade_len:].copy()
                            yield audio_chunk[:-fade_len].tobytes()
                        else:
                            yield audio_chunk.tobytes()
                            prev_chunk_tail = None
                        
                        # Update history
                        history_codes = torch.cat([history_codes, to_decode], dim=0)
                        chunk_buffer = []
                        is_first_chunk = False
            
            # Flush remaining buffer
            if chunk_buffer:
                context = history_codes[-self.context_size:] if len(history_codes) > self.context_size else history_codes
                audio_chunk = self._decode_with_context(torch.stack(chunk_buffer, dim=0), context)
                if prev_chunk_tail is not None:
                    audio_chunk = self._apply_crossfade(prev_chunk_tail, audio_chunk)
                yield audio_chunk.tobytes()
    
    def _apply_crossfade(self, tail: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Apply linear crossfade between chunks to eliminate clicks."""
        fade_len = len(tail)
        if len(current) < fade_len:
            return current
        
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = 1 - fade_in
        current[:fade_len] = tail * fade_out + current[:fade_len] * fade_in
        return current
    
    def _decode_with_context(self, new_codes: torch.Tensor, context_codes: torch.Tensor) -> np.ndarray:
        """
        Decode audio codes with context for better quality.
        
        The decoder needs context to produce stable output, so we decode
        context + new_codes together, then trim the context portion.
        """
        with torch.no_grad():
            # Decode context to get trim length
            wav_ctx_all, _ = self.tokenizer.decode([{"audio_codes": context_codes}])
            trim_samples = len(wav_ctx_all[0])
            
            # Decode full sequence
            full_codes = torch.cat([context_codes, new_codes], dim=0)
            wavs, sr = self.tokenizer.decode([{"audio_codes": full_codes}])
            wav = wavs[0]
            
            # Trim context portion
            if trim_samples < len(wav):
                return wav[trim_samples:].astype(np.float32)
            return wav.astype(np.float32)
