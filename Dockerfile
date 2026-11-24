# Dockerfile for AI Lecture Video Processing & Summarization System
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement definition and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and directories
COPY . /app

# Create required upload and output directories
RUN mkdir -p uploads outputs cache/audio cache/transcripts outputs/subtitles

# Expose web application port
EXPOSE 5000

# Entry point command
CMD ["python", "app.py"]
