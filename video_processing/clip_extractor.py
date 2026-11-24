import os
import subprocess
from typing import List, Tuple

def extract_video_clip(
    input_video: str,
    start_sec: float,
    end_sec: float,
    output_clip_path: str,
    fast_copy: bool = True
) -> str:
    """
    Extracts a sub-clip from input_video between start_sec and end_sec using FFmpeg.
    Preserves original video & audio streams using ultra-fast stream copy or ultrafast encoding.
    """
    os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
    duration = end_sec - start_sec

    if fast_copy:
        # Fast stream copy without frame re-encoding
        cmd_copy = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-to", f"{end_sec:.3f}",
            "-i", input_video,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_clip_path
        ]
        res_copy = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_copy.returncode == 0 and os.path.exists(output_clip_path) and os.path.getsize(output_clip_path) > 0:
            return output_clip_path

    # Fallback: Frame-accurate sub-clip extraction with ultrafast re-encoding
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", input_video,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        output_clip_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg clip extraction failed for interval {start_sec}-{end_sec}:\n{result.stderr}")

    return output_clip_path

def extract_multiple_clips(
    input_video: str,
    segments: List[Tuple[float, float]],
    temp_dir: str,
    fast_copy: bool = True
) -> List[str]:
    """Extracts multiple video clips corresponding to time segment tuples."""
    os.makedirs(temp_dir, exist_ok=True)
    clip_paths = []

    for idx, (start_sec, end_sec) in enumerate(segments):
        clip_name = f"clip_{idx:03d}_{start_sec:.2f}_{end_sec:.2f}.mp4"
        clip_path = os.path.join(temp_dir, clip_name)
        extract_video_clip(input_video, start_sec, end_sec, clip_path, fast_copy=fast_copy)
        clip_paths.append(clip_path)

    return clip_paths
