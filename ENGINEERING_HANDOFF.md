# Qwen3-TTS Streaming Server: Engineering Handoff (Updated 2026-01-31)

## TL;DR

**Major Update**: Server now supports TRUE STREAMING audio generation via `generate_stream` message type.

- Uses manual KV-cache stepping to yield audio chunks as tokens are generated
- First audio chunk in ~0.67s (8 tokens) instead of waiting for full generation
- Adapted from [CloudWells/qwen3-tts-realtime-streaming](https://github.com/CloudWells/qwen3-tts-realtime-streaming)

---

## Architecture Overview

### Server Components

```
/files/qwen3tts/
├── server.py              # WebSocket server with streaming support
├── streaming_engine.py    # True streaming generation engine (NEW)
├── Dockerfile
├── requirements.txt
└── test_client.py
```

### MindRoot Plugin

```
/xfiles/plugins_ah/mr_qwen3tts/src/mr_qwen3tts/
├── mod.py                 # Main plugin, stream_tts service, speak command
├── realtime_stream.py     # Handles partial_command pipe for incremental text
└── audio_pacer.py         # Paces audio output to SIP
```

---

## Message Protocol

### Client → Server

| Type | Description | Key Fields |
|------|-------------|------------|
| `init` | Initialize voice clone | `ref_audio_base64`, `ref_text`, `voice_id` |
| `generate` | Non-streaming generation (legacy) | `text`, `language` |
| `generate_stream` | **TRUE STREAMING** generation | `text`, `language`, `initial_chunk_tokens`, `stream_chunk_tokens` |
| `text` | Streaming text input | `text`, `final` |
| `cancel` | Cancel current generation | - |

### Server → Client

| Type | Description |
|------|-------------|
| `connected` | Connection established |
| `ready` | Voice initialized, includes `voice_id` |
| `audio_start` | Audio generation starting |
| Binary | Raw ulaw 8kHz audio chunks (160 bytes = 20ms) |
| `audio_end` | Audio generation complete |
| `error` | Error with `message` field |

---

## Streaming Engine Details

### How It Works

The `streaming_engine.py` implements step-by-step inference:

1. **Prepare embeddings** - Build input embeddings from text and voice clone prompt
2. **Token-by-token generation** - Manual forward pass with KV-cache
3. **Extract frame codes** - Get audio codes from `outputs.hidden_states[1]`
4. **Buffer and decode** - Accumulate tokens, decode with context when threshold reached
5. **Yield audio** - Send float32 PCM chunks as they're ready

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_chunk_tokens` | 8 | Tokens before first audio (~0.67s at 12Hz) |
| `stream_chunk_tokens` | 8 | Tokens per subsequent chunk (~0.67s) |
| `context_size` | 38 | Context tokens for decoding (~3s) |
| `crossfade_samples` | 240 | Crossfade length (~10ms at 24kHz) |

### Latency Expectations

- **First audio**: ~0.67s (8 tokens × 83ms/token)
- **Subsequent chunks**: ~0.67s each
- **Can be tuned lower** by reducing `initial_chunk_tokens` (may affect quality)

---

## Voice Caching

Voice prompts are cached server-side by `voice_id` (hash of audio + ref_text).

**Important**: Voice prompt is per-connection session state. After reconnect:
1. Client can send `init` with just `voice_id` (no audio upload)
2. Server re-binds cached prompt to new session
3. Much faster than re-uploading and re-processing audio

---

## Plugin Modes

### Standard Mode

```python
async for chunk in stream_tts(text="Hello", context=context):
    # chunks arrive as server generates them
```

### Realtime Mode (`MR_QWEN3TTS_REALTIME_STREAM=1`)

1. `partial_command` pipe captures text deltas from LLM
2. Text buffered in `RealtimeSpeakSession`
3. On `speak()` command completion, `finish()` sends accumulated text
4. Audio streams back via WebSocket

---

## Environment Variables

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_TTS_MODEL` | auto-detect | Model path or HF repo |
| `QWEN3_TTS_DEVICE` | `cuda:0` | PyTorch device |
| `QWEN3_TTS_DTYPE` | `bfloat16` | Model dtype |
| `QWEN3_TTS_HOST` | `0.0.0.0` | Server bind host |
| `QWEN3_TTS_PORT` | `8765` | Server port |
| `WHISPER_MODEL` | `base` | Whisper model for auto-transcription |

### Plugin

| Variable | Default | Description |
|----------|---------|-------------|
| `MR_QWEN3TTS_WS_URL` | `ws://localhost:8765` | Server WebSocket URL |
| `MR_QWEN3TTS_REF_AUDIO` | - | Fallback reference audio path |
| `MR_QWEN3TTS_REF_TEXT` | - | Fallback reference text |
| `MR_QWEN3TTS_REALTIME_STREAM` | `0` | Enable realtime streaming mode |

---

## Testing

### Quick Test

```bash
# Start server
cd /files/qwen3tts
python server.py

# In another terminal, test with client
python test_client.py
```

### Verify Streaming

Look for log messages like:
```
First audio chunk in 0.XXXs
Streaming complete: N chunks, M bytes in X.XXs
```

If you see "First audio chunk" time close to total time, streaming isn't working.

---

## Troubleshooting

### "no voice prompt for session"

Client sent `generate_stream` before `init`. Always init voice first.

### High first-audio latency

1. Check `initial_chunk_tokens` - lower = faster but potentially less stable
2. Verify streaming engine loaded: look for "Streaming engine initialized" in logs
3. If falling back to non-streaming, check for import errors

### Voice quality issues

1. Increase `initial_chunk_tokens` (e.g., 16 or 24)
2. Ensure reference audio is good quality (3-10s, clear speech)
3. Check `context_size` - larger = better quality but more memory

---

## Files Changed

### New Files
- `/files/qwen3tts/streaming_engine.py` - True streaming generation

### Modified Files
- `/files/qwen3tts/server.py` - Added `generate_stream` handler
- `/xfiles/plugins_ah/mr_qwen3tts/src/mr_qwen3tts/mod.py` - Use `generate_stream`
- `/xfiles/plugins_ah/mr_qwen3tts/src/mr_qwen3tts/realtime_stream.py` - Use `generate_stream`

---

## References

- [CloudWells/qwen3-tts-realtime-streaming](https://github.com/CloudWells/qwen3-tts-realtime-streaming) - Streaming implementation reference
- [Qwen3-TTS HuggingFace Discussion](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/discussions/4) - Streaming capability discussion
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - Official repo
