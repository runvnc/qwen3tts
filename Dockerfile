FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .
COPY reference_demo.py .

# Expose WebSocket port
EXPOSE 8765

# Environment variables
ENV QWEN3_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV QWEN3_TTS_DEVICE=cuda:0
ENV QWEN3_TTS_DTYPE=bfloat16
ENV QWEN3_TTS_HOST=0.0.0.0
ENV QWEN3_TTS_PORT=8765

# Run server
CMD ["python", "server.py"]
