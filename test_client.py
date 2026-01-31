#!/usr/bin/env python3
"""Test client for Qwen3-TTS WebSocket server."""

import asyncio
import base64
import json
import sys
import wave
import io

import websockets


async def test_server(
    ws_url: str = "ws://localhost:8765",
    ref_audio_path: str = None,
    ref_text: str = None,
    test_text: str = "Hello, this is a test of the Qwen3 text to speech system.",
    output_path: str = "output.wav",
):
    """Test the Qwen3-TTS WebSocket server."""
    print(f"Connecting to {ws_url}...")
    
    async with websockets.connect(ws_url, max_size=50*1024*1024) as ws:
        # Wait for connected message
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"Connected: {data}")
        
        # Initialize voice if reference audio provided
        if ref_audio_path:
            print(f"Loading reference audio from {ref_audio_path}...")
            with open(ref_audio_path, 'rb') as f:
                audio_bytes = f.read()
            ref_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            await ws.send(json.dumps({
                "type": "init",
                "ref_audio_base64": ref_audio_b64,
                "ref_text": ref_text or "",
                "x_vector_only": not bool(ref_text),
            }))
            
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Voice init response: {data}")
            
            if data.get("type") == "error":
                print(f"Error: {data.get('message')}")
                return
        
        # Generate audio
        print(f"Generating audio for: {test_text}")
        await ws.send(json.dumps({
            "type": "generate",
            "text": test_text,
            "language": "Auto",
        }))
        
        # Collect audio chunks
        audio_chunks = []
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                audio_chunks.append(msg)
                print(f"Received audio chunk: {len(msg)} bytes")
            else:
                data = json.loads(msg)
                print(f"Message: {data}")
                
                if data.get("type") == "audio_end":
                    break
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
                    return
        
        # Combine and save audio
        if audio_chunks:
            all_audio = b"".join(audio_chunks)
            print(f"Total audio: {len(all_audio)} bytes ({len(all_audio)/8000:.2f}s at 8kHz)")
            
            # Save as WAV (ulaw 8kHz)
            with wave.open(output_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(1)  # 8-bit for ulaw
                wav.setframerate(8000)
                wav.setcomptype('ULAW', 'CCITT G.711 u-law')
                wav.writeframes(all_audio)
            
            print(f"Saved to {output_path}")
        else:
            print("No audio received")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS WebSocket server")
    parser.add_argument("--url", default="ws://localhost:8765", help="WebSocket URL")
    parser.add_argument("--ref-audio", help="Reference audio file for voice cloning")
    parser.add_argument("--ref-text", help="Transcript of reference audio")
    parser.add_argument("--text", default="Hello, this is a test of the Qwen3 text to speech system.",
                        help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    
    args = parser.parse_args()
    
    asyncio.run(test_server(
        ws_url=args.url,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        test_text=args.text,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
