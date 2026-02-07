#!/bin/bash
pm2 start server_optimized.py --name qwen3tts --interpreter python3 -- --host 0.0.0.0 --port 3030
