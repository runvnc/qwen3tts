# Qwen3-TTS Engineering Handoff - Session 2 (2026-01-31)

## Current State

### What's Working
1. **Voice caching** - Server now checks cache BEFORE decoding audio/running Whisper
2. **Streaming engine** - Manual KV-cache stepping yields audio as tokens generate
3. **Plugin integration** - Uses `generate_stream` message type

### What's Broken
1. **Audio hiccupping** - Streaming chunks arrive with gaps, causing choppy playback
2. **Connection errors** - "received 1000 (OK)" when client closes before server finishes
3. **~1 second first-audio latency** - Seems too high, may be redoing work unnecessarily

## Key Files

### Server (RunPod)
- `/workspace/qwen3tts/server.py` - WebSocket server
- `/workspace/qwen3tts/streaming_engine.py` - True streaming generation

### Plugin (MindRoot)
- `/xfiles/plugins_ah/mr_qwen3tts/src/mr_qwen3tts/mod.py` - Main plugin
- `/xfiles/plugins_ah/mr_qwen3tts/src/mr_qwen3tts/realtime_stream.py` - Partial text handling

### Local copies for editing
- `/files/qwen3tts/server.py`
- `/files/qwen3tts/streaming_engine.py`

## Latency Analysis

From logs:
```
connection open
New connection: XXX
Voice found in cache via quick hash: XXX (skipping audio decode/transcription)
Using cached voice XXX (voice_id-only init)
First audio chunk in 1.017s
```

### Breakdown (estimated)
- Connection open -> voice cache check: ~0.5s (network + hash)
- Voice init -> first audio: ~1.0s (THIS IS THE PROBLEM)

### Where is the 1 second going?

Looking at `streaming_engine.py`, on EVERY generation:

```python
# These are computed fresh each time:
voice_clone_prompt_dict = self.wrapper._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
ref_code = voice_clone_prompt_dict["ref_code"][0]
input_id = self.wrapper._tokenize_texts([self.wrapper._build_assistant_text(text)])[0]
voice_clone_spk_embeds = self.model.generate_speaker_prompt(voice_clone_prompt_dict)
speaker_embed = voice_clone_spk_embeds[0]

# Build embeddings (expensive!):
tts_const_embeds = self.talker.text_projection(self.talker.get_text_embeddings()(tts_token_ids))
codec_input_embedding_0 = self.talker.get_input_embeddings()(...)
codec_input_embedding_1 = self.talker.get_input_embeddings()(...)
role_start_embed = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, :3]))

# ICL prompt generation:
icl_input_embed, trailing_text_hidden = self.model.generate_icl_prompt(...)
```

**HYPOTHESIS**: Many of these embeddings are CONSTANT for a given voice and could be cached:
- `tts_const_embeds` (BOS, EOS, PAD tokens) - CONSTANT
- `speaker_embed` - CONSTANT per voice
- `codec_input_embedding_0`, `codec_input_embedding_1` - CONSTANT per voice
- `ref_code` - CONSTANT per voice

Only these should be computed per-request:
- `input_id` (text tokenization)
- `trailing_text_hidden` (depends on text)
- `talker_input_embed` (depends on text)

## Proposed Fix: Cache Voice Embeddings

### Server-side voice cache should store:
```python
voice_cache[voice_id] = {
    "prompt_items": prompt_items,  # Current
    "speaker_embed": speaker_embed,  # NEW
    "ref_code": ref_code,  # NEW
    "tts_const_embeds": (tts_bos_embed, tts_eos_embed, tts_pad_embed),  # NEW
    "codec_input_embedding": codec_input_embedding,  # NEW
}
```

### Streaming engine should:
1. Accept pre-computed embeddings
2. Only compute text-dependent parts per request
3. Skip redundant embedding computations

## Hiccupping Issue

### Cause
Streaming engine yields chunks as tokens are generated. Token generation has variable timing:
- Some tokens fast (~20ms)
- Some tokens slow (~100ms+)
- Gaps between chunks cause audio hiccups

### Solutions
1. **Server-side buffering** - Buffer 2-3 chunks before sending, pace output
2. **Client-side buffering** - AudioPacer buffers more before playing
3. **Non-streaming fallback** - Generate all audio first, then stream with pacing

## Connection Issues

### Why new connection each time?
The plugin creates a new WebSocket connection for each `generate_stream` call because:
1. Previous connection may have timed out
2. Error handling closes connection
3. Lock release allows reconnection

### Fix
Keep connection alive between requests. The plugin already tries to reuse connections via `connect()` ping check, but something is causing disconnects.

## Quick Wins

### 1. Cache constant embeddings (HIGH IMPACT)
Modify `streaming_engine.py` to cache:
- `tts_const_embeds`
- `speaker_embed` per voice
- `codec_input_embedding` per voice

Estimated savings: 200-500ms per request

### 2. Reduce initial_chunk_tokens (MEDIUM IMPACT)
Currently 6 tokens (~0.5s). Could try 4 tokens (~0.33s).
Tradeoff: Lower quality first chunk.

### 3. Add server-side pacing (FIXES HICCUPPING)
Instead of sending chunks immediately:
```python
buffer = []
for chunk in generate_stream(...):
    buffer.append(chunk)
    if len(buffer) >= 2:  # Buffer 2 chunks
        await websocket.send(buffer.pop(0))
        await asyncio.sleep(0.02)  # Pace at 20ms
```

### 4. Pre-warm model (LOW IMPACT)
Run a dummy generation on startup to warm CUDA kernels.

## Code That Was Attempted But Had Issues

The following changes were attempted but had indentation issues:

1. **Debug logging** - Added timing logs to server.py and streaming_engine.py
2. **ulaw fix** - Changed `_pcm16_to_ulaw_numpy` to return bytes properly

The server.py was reset to original via `git checkout`. The streaming_engine.py changes may still be in place.

## Next Steps

1. **Add timing instrumentation** to identify exact bottleneck
2. **Implement embedding caching** in streaming_engine.py
3. **Add server-side buffering** to fix hiccupping
4. **Test with vLLM** for faster inference (Qwen claims 97ms with vLLM)

## Reference: CloudWells Implementation

Cloned to `/files/qwen3-tts-realtime-streaming/`

Their approach:
- 3-second initial buffer for voice stability
- 8-token subsequent chunks
- Context-aware decoding with 38-token sliding window
- Cross-fading between chunks

We adapted this but removed the 3-second buffer for lower latency.

## Reference: Qwen3-TTS Architecture

From technical report (arXiv:2601.15621):
- 12Hz tokenizer = 83ms per token
- Dual-track architecture for streaming
- First-packet latency: 97ms (with vLLM + CUDA graphs)
- The 97ms is NOT achievable with basic SDK

## Environment

- RunPod GPU server running server.py
- MindRoot with mr_qwen3tts plugin
- WebSocket connection between them
- Audio output to SIP via AudioPacer
