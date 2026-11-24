"""
===============================================================================
Module: Video Composer (services/video_composer.py)
===============================================================================
1. Purpose:
   Concatenates extracted lecture video clips while preserving the ORIGINAL SPEAKER'S VOICE
   and burning synchronized SRT subtitles into the final short lecture video.
2. Input:
   Source video path, list of selected segment (start, end) tuples, SRT subtitle path (optional), audio_mode (default "original").
3. Processing:
   - Preserves original lecturer voice audio and video streams
   - Maintains pristine 1:1 audio-video synchronization
   - Burns synchronized SRT subtitles onto the output video stream
===============================================================================
"""

import os
from typing import List, Tuple, Callable, Optional
from video_processing.video_merger import generate_summary_video
from utils.ffmpeg_utils import get_media_metadata

def compose_final_lecture_video(
    input_video: str,
    selected_segments: List[Tuple[float, float]],
    output_path: str,
    narration_audio_path: str = None,
    srt_subtitle_path: str = None,
    audio_mode: str = "original",
    burn_subtitles: bool = False,
    progress_callback: Optional[Callable[[str, int, str], None]] = None
) -> str:
    """Composes selected lecture clips preserving original lecturer audio and video."""
    if not selected_segments:
        raise ValueError("Cannot compose final video: No video segments provided.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[VIDEO COMPOSER] Composing final video ({len(selected_segments)} clips, Audio Mode: {audio_mode.upper()}, Subtitles: {'BURNED' if burn_subtitles else 'EXTERNAL'}) -> '{output_path}'...")

    final_file = generate_summary_video(
        input_video=input_video,
        selected_segments=selected_segments,
        output_summary_path=output_path,
        narration_audio_path=narration_audio_path if audio_mode != "original" else None,
        srt_subtitle_path=srt_subtitle_path,
        audio_mode=audio_mode,
        burn_subtitles=burn_subtitles,
        progress_callback=progress_callback
    )

    if not os.path.exists(final_file) or os.path.getsize(final_file) == 0:
        raise RuntimeError(f"Video composition failed: '{output_path}' was not generated.")

    meta = get_media_metadata(final_file)
    print(f"[VIDEO COMPOSER] Video composition complete: '{final_file}' (Duration: {meta['duration_seconds']}s, Audio Stream: {meta['has_audio']})")

    return final_file
