"""
===============================================================================
Module: Video Stream Processor (services/video_processor.py)
===============================================================================
1. Purpose:
   Inspects and validates uploaded lecture video files, extracts stream metadata,
   and verifies container formats, codecs, resolution, and audio availability.
2. Input:
   Path to uploaded lecture video file.
3. Processing:
   - Validates file existence, format extension, size limit
   - Extracts duration, resolution, frame rate, audio sample rate via ffprobe
   - Returns structured metadata dictionary
4. Algorithm/Model:
   FFmpeg / ffprobe stream probing.
5. Output:
   Metadata dictionary containing duration, resolution, FPS, codecs, audio presence.
6. Data Stored:
   filename, duration, resolution, FPS, audio information, upload status.
7. Failure Cases:
   Rejects missing files, empty files, unsupported extensions (.exe, .txt), or corrupted videos.
8. Evaluation:
   Verified against standard lecture video test samples (MP4, AVI, MOV, MKV, WebM).
9. Integration:
   First stage in pipeline; provides video metadata to job manager and audio extractor.
===============================================================================
"""

import os
from typing import Dict, Any
from utils.ffmpeg_utils import get_media_metadata
from config import ALLOWED_EXTENSIONS, MAX_VIDEO_SIZE_MB, MAX_LECTURE_DURATION_SECONDS

def validate_and_inspect_video(video_path: str, max_size_mb: float = MAX_VIDEO_SIZE_MB) -> Dict[str, Any]:
    """Validates lecture video file format and extracts complete stream metadata."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Allowed extensions: {list(ALLOWED_EXTENSIONS)}")

    size_bytes = os.path.getsize(video_path)
    if size_bytes == 0:
        raise ValueError("Uploaded video file is empty (0 bytes).")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Video file size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb:.0f} MB.")

    metadata = get_media_metadata(video_path)

    if metadata["duration_seconds"] <= 0.0:
        raise ValueError("Invalid video file: duration is 0 seconds.")

    if metadata["duration_seconds"] > MAX_LECTURE_DURATION_SECONDS:
        hours = metadata["duration_seconds"] / 3600.0
        max_h = MAX_LECTURE_DURATION_SECONDS / 3600.0
        raise ValueError(
            f"Lecture video duration ({hours:.1f} hours) exceeds the maximum allowed limit of "
            f"{max_h:.0f} hours. Please upload a standard lecture video."
        )

    if not metadata["has_audio"]:
        raise ValueError("Video validation failed: No audio stream detected in lecture video.")

    print(f"[VIDEO PROCESSOR] Validated '{metadata['filename']}': Duration {metadata['duration_seconds']}s, Res: {metadata['resolution']}, FPS: {metadata['fps']}")
    return metadata
