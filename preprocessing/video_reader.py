import os
import cv2
from typing import Dict, Any

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def validate_video_file(video_path: str) -> None:
    """Check if the video file exists and has a supported extension."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")
    
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported video extension '{ext}'. Allowed formats: {ALLOWED_EXTENSIONS}")

def format_duration(seconds: float) -> str:
    """Format total seconds into MM:SS.mmm format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{mins:02d}:{secs:02d}.{millis:03d}"

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts essential video metadata from a video file path.
    
    Returns a dictionary with:
    - filepath: Absolute path
    - filename: File basename
    - fps: Frames per second (float)
    - total_frames: Total number of frames (int)
    - width: Video width in pixels (int)
    - height: Video height in pixels (int)
    - resolution: Width x Height string (e.g. '1280x720')
    - duration_seconds: Duration in seconds (float)
    - duration_formatted: Formatted time string (MM:SS.mmm)
    """
    validate_video_file(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file using OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Unable to read valid stream properties from {video_path} (FPS: {fps}, frames: {total_frames}).")

    duration_seconds = total_frames / fps
    resolution = f"{width}x{height}"

    return {
        "filepath": os.path.abspath(video_path),
        "filename": os.path.basename(video_path),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "resolution": resolution,
        "duration_seconds": round(duration_seconds, 2),
        "duration_formatted": format_duration(duration_seconds),
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            meta = get_video_metadata(path)
            print("Video Metadata:")
            for k, v in meta.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python preprocessing/video_reader.py <path_to_video>")
