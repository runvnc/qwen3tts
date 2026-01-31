# Qwen3-TTS WebSocket Server

Streaming TTS server using Qwen3-TTS with voice cloning support.

## Features

- **Voice cloning** from 3-second reference audio
- **Streaming audio output** (ulaw 8kHz for SIP/telephony)
- **Auto model selection** based on available VRAM:
  - ≥12GB VRAM → 1.7B model (better quality)
  - <12GB VRAM → 0.6B model (lower VRAM requirement)
- **~100ms first packet latency** for real-time applications
- WebSocket protocol for real-time communication
- RunPod/Docker deployment ready

## Model Sizes & Requirements

| Model | Parameters | VRAM | First Packet Latency |
|-------|-----------|------|---------------------|
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | ~12-16GB | ~100ms |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | ~4-6GB | ~97ms |

## Quick Start

### Local Development

```bash
pip install -r requirements.txt

# Auto-detect model based on VRAM
python server.py

# Or explicitly specify model
python server.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
python server.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

### Docker

```bash
docker build -t qwen3tts-server .
docker run --gpus all -p 8765:8765 qwen3tts-server
```

### RunPod Deployment

1. Build and push the Docker image
2. Create a RunPod serverless endpoint or GPU pod
3. Set environment variables as needed

## WebSocket Protocol

Connect to `ws://host:8765`

### Initialize Voice Clone

```json
{
  "type": "init",
  "ref_audio_base64": "<base64 encoded wav/mp3>",
  "ref_text": "The transcript of the reference audio",
  "x_vector_only": false
}
```

Response:
```json
{"type": "ready", "voice_loaded": true}
```

### Generate Audio

```json
{
  "type": "generate",
  "text": "Hello, this is a test.",
  "language": "Auto"
}
```

Response sequence:
1. `{"type": "audio_start"}`
2. Binary chunks (160 bytes each, ulaw 8kHz, 20ms per chunk)
3. `{"type": "audio_end"}`

### Streaming Text Input

```json
{"type": "text", "text": "Hello ", "final": false}
{"type": "text", "text": "world!", "final": true}
```

### Cancel Generation

```json
{"type": "cancel"}
```

## Command Line Options

```
--model, -m     Model path (default: auto-detect based on VRAM)
--device, -d    Device: cuda:0, cpu, etc. (default: cuda:0)
--dtype         Model dtype: bfloat16, float16, float32 (default: bfloat16)
--host          Server host (default: 0.0.0.0)
--port, -p      Server port (default: 8765)
```

## Environment Variables

- `QWEN3_TTS_MODEL`: Model path (default: auto-detect)
- `QWEN3_TTS_DEVICE`: Device (default: `cuda:0`)
- `QWEN3_TTS_DTYPE`: Model dtype (default: `bfloat16`)
- `QWEN3_TTS_HOST`: Server host (default: `0.0.0.0`)
- `QWEN3_TTS_PORT`: Server port (default: `8765`)

## Audio Format

Output audio is:
- Format: μ-law (ulaw)
- Sample rate: 8000 Hz
- Channels: Mono
- Chunk size: 160 bytes (20ms)

This format is compatible with SIP/VoIP telephony systems.

## Testing

```bash
# Test with reference audio
python test_client.py --ref-audio voice.wav --ref-text "Hello world" --text "Testing the voice clone"

# Test without voice clone (mock mode if no GPU)
python test_client.py --text "Hello world"
```

## Latency Optimization

For lowest latency:
1. Use the 12Hz tokenizer models (already default)
2. Use GPU with sufficient VRAM for the 1.7B model
3. Keep WebSocket connection alive between requests
4. Pre-initialize voice clone before first generation
