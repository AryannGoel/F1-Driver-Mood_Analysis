# Pit Wall — runs the full FastAPI app + Hugging Face models in one container.
# Works on Hugging Face Spaces (SDK: docker), Render, Railway, Fly.io, etc.
FROM python:3.12-slim

# ffmpeg decodes the mp3/webm radio clips for Whisper + wav2vec2
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (smaller, no CUDA), then the rest
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# make cache + clip dirs writable no matter which user the Space runs the container as
RUN mkdir -p /app/.hfcache /app/.fastf1cache /app/clips && chmod -R 777 /app

# writable cache locations inside the container
ENV HF_HOME=/app/.hfcache \
    PORT=7860
# models (~1 GB) download lazily on the first analysis; that request is slow, then cached.

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
