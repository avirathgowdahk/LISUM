"""
===============================================================================
Module: FFmpeg Utilities (utils/ffmpeg_utils.py)
===============================================================================
1. Purpose:
   Provides low-level FFmpeg and ffprobe wrappers for stream validation, metadata
   extraction, audio isolation, clip slicing, video composition, and subtitle burning.
2. Input:
   Video file paths, timestamps, target codecs, output configuration.
3. Processing:
   - Validates video/audio stream integrity via ffprobe
   - Extracts mono 16kHz speech WAV audio
   - Cuts video clips matching exact timestamps
   - Normalizes and concatenates clips into final summarized lecture video
4. Output:
   Metadata dictionary, extracted audio/video files.
===============================================================================
"""

import os
import sys
import subprocess
import json
from typing import Dict, Any, List, Tuple

def get_media_metadata(video_path: str) -> Dict[str, Any]:
    """
    Executes ffprobe to extract full stream metadata from input lecture video.
    Validates file existence, container format, video codec, audio codec, resolution, FPS, duration.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise ValueError(f"Failed to inspect video metadata via ffprobe: {res.stderr}")

    probe = json.loads(res.stdout)
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])

    duration_sec = float(fmt.get("duration", 0.0))
    size_bytes = int(fmt.get("size", 0))

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0

    # Calculate FPS
    fps = 30.0
    if video_stream and "r_frame_rate" in video_stream:
        r_fps = video_stream["r_frame_rate"].split("/")
        if len(r_fps) == 2 and float(r_fps[1]) > 0:
            fps = float(r_fps[0]) / float(r_fps[1])

    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    sample_rate = int(audio_stream.get("sample_rate", 0)) if audio_stream else 0
    channels = int(audio_stream.get("channels", 0)) if audio_stream else 0

    return {
        "filename": os.path.basename(video_path),
        "file_path": os.path.abspath(video_path),
        "duration_seconds": round(duration_sec, 2),
        "file_size_mb": round(size_bytes / (1024 * 1024), 2),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "video_codec": video_stream.get("codec_name") if video_stream else None,
        "has_audio": has_audio,
        "audio_codec": audio_codec,
        "sample_rate": sample_rate,
        "channels": channels
    }

def extract_normalized_audio(video_path: str, output_wav_path: str) -> str:
    """Extracts 16kHz mono WAV audio normalized for speech recognition."""
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")

    return output_wav_path

def cut_video_segment(video_path: str, start_time: float, end_time: float, output_clip_path: str) -> str:
    """Slices precise video segment from source video."""
    os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
    duration = max(0.1, end_time - start_time)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-to", f"{end_time:.3f}",
        "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        output_clip_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.exists(output_clip_path):
        raise RuntimeError(f"FFmpeg clip cutting failed for interval [{start_time:.2f}, {end_time:.2f}]: {res.stderr}")

    return output_clip_path

def concatenate_video_segments(clip_paths: List[str], output_video_path: str) -> str:
    """Concatenates video clips into one output video using FFmpeg concat filter."""
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    if not clip_paths:
        raise ValueError("No video clip paths provided for concatenation.")

    if len(clip_paths) == 1:
        # Simple copy
        cmd = ["ffmpeg", "-y", "-i", clip_paths[0], "-c", "copy", output_video_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_video_path

    # Build FFmpeg concat list file
    concat_txt = output_video_path + ".txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in clip_paths:
            escaped = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        output_video_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if os.path.exists(concat_txt):
        os.remove(concat_txt)

    if res.returncode != 0 or not os.path.exists(output_video_path):
        # Fallback to re-encoding filter_complex concat
        inputs = []
        filter_str = []
        for idx, cp in enumerate(clip_paths):
            inputs.extend(["-i", cp])
            filter_str.append(f"[{idx}:v][{idx}:a]")
        filter_str.append(f"concat=n={len(clip_paths)}:v=1:a=1[v][a]")

        cmd_fallback = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", "".join(filter_str),
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            output_video_path
        ]
        res_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_fb.returncode != 0:
            raise RuntimeError(f"FFmpeg concatenation failed: {res_fb.stderr}")

    return output_video_path
