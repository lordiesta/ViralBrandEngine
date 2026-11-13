# ================================================================
#  ViralBrandEngine — Dockerfile
# ================================================================
FROM python:3.12-slim

# System dependencies (FFmpeg + ImageMagick for MoviePy text rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policy for MoviePy TextClip
RUN sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@*"/' \
    /etc/ImageMagick-6/policy.xml 2>/dev/null || true

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create media directories
RUN mkdir -p media/renders media/audio media/broll

# Default: run the FastAPI server
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
