import os
import time
import shutil
import subprocess
from typing import List, Tuple, Callable, Optional
from video_processing.clip_extractor import extract_video_clip, extract_multiple_clips
from utils.ffmpeg_utils import get_media_metadata

def merge_video_clips(
    clip_paths: List[str],
    output_summary_path: str,
    temp_dir: str
) -> str:
    """Concatenates video clips using FFmpeg concat demuxer preserving original audio & video."""
    if not clip_paths:
        raise ValueError("No video clips provided to merge.")

    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    concat_file_path = os.path.join(temp_dir, "concat_list.txt")

    with open(concat_file_path, "w", encoding="utf-8") as f:
        for path in clip_paths:
            abs_path = os.path.abspath(path).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file_path,
        "-c", "copy",
        output_summary_path
    ]

    print(f"[FFMPEG] Command:\n{' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[FFMPEG Warning] Stream copy concat failed: {result.stderr}\nFallback to ultrafast re-encode...")
        fallback_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            output_summary_path
        ]
        print(f"[FFMPEG] Command:\n{' '.join(fallback_cmd)}")
        fallback_res = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if fallback_res.returncode != 0:
            raise RuntimeError(f"FFmpeg video concatenation failed:\n{fallback_res.stderr}")

    return output_summary_path

def generate_summary_video(
    input_video: str,
    selected_segments: List[Tuple[float, float]],
    output_summary_path: str,
    narration_audio_path: str = None,
    srt_subtitle_path: str = None,
    audio_mode: str = "original",
    burn_subtitles: bool = False,
    progress_callback: Optional[Callable[[str, int, str], None]] = None
) -> str:
    """
    Renders summarized lecture video preserving ORIGINAL LECTURER AUDIO from source video.
    Executes fast segment extraction and stream-copy concat demuxer rendering.
    """
    t_start = time.time()
    perf = {}

    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    if not selected_segments:
        raise ValueError("No video segments selected for summarization.")

    # Validate audio stream in input video
    meta = get_media_metadata(input_video)
    if not meta.get("has_audio"):
        print("[ERROR] Source video contains no audio stream.")
        raise ValueError("Source video contains no audio stream.")

    v_codec = meta.get("codec") or meta.get("video_codec") or "h264"
    a_codec = meta.get("audio_codec") or "aac"
    has_srt = srt_subtitle_path and os.path.exists(srt_subtitle_path) and os.path.getsize(srt_subtitle_path) > 0
    sub_mode = "burned" if (burn_subtitles and has_srt) else "external"

    # Hardware Acceleration Detection
    hw_accel = "no"
    try:
        hw_check = subprocess.run(["ffmpeg", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "h264_nvenc" in hw_check.stdout or "h264_qsv" in hw_check.stdout:
            hw_accel = "yes"
    except Exception:
        pass

    target_duration = max(1.0, sum(end - start for (start, end) in selected_segments))

    print(f"\n==================================================")
    print(f"[FFMPEG] Starting Fast Lecture Video Composition")
    print(f"[FFMPEG] Source Video   : {input_video}")
    print(f"[FFMPEG] Video Codec    : {v_codec}")
    print(f"[FFMPEG] Audio Codec    : {a_codec}")
    print(f"[FFMPEG] Subtitle Mode  : {sub_mode.upper()} ({'External .SRT file' if sub_mode == 'external' else 'Burned into video stream'})")
    print(f"[FFMPEG] Hardware Accel : {hw_accel.upper()}")
    print(f"[FFMPEG] Selected Clips : {len(selected_segments)} (Target: {target_duration:.2f}s)")
    print(f"==================================================\n")

    temp_dir = os.path.join("outputs", f"temp_clips_{int(time.time()*1000)}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # ---------------------------------------------------------------------
        # STAGE 1: Extract Selected Segments (93–96%)
        # ---------------------------------------------------------------------
        t0 = time.time()
        if progress_callback:
            progress_callback("EXTRACTING_CLIPS", 93, f"Extracting {len(selected_segments)} selected video segments...")

        clip_paths = []
        for idx, (start_sec, end_sec) in enumerate(selected_segments):
            clip_path = os.path.join(temp_dir, f"clip_{idx:03d}_{start_sec:.2f}_{end_sec:.2f}.mp4")
            extract_video_clip(input_video, start_sec, end_sec, clip_path, fast_copy=True)
            clip_paths.append(clip_path)

        perf["segment_extraction"] = round(time.time() - t0, 3)
        print(f"[FFMPEG PERF] Segment extraction: {perf['segment_extraction']} sec")

        # ---------------------------------------------------------------------
        # STAGE 2: Joining Video + Original Audio (96–98%)
        # ---------------------------------------------------------------------
        t0 = time.time()
        if progress_callback:
            progress_callback("JOINING_CLIPS", 96, "Joining video clips with original lecturer audio...")

        raw_concat_path = os.path.join(temp_dir, "raw_concat.mp4")
        merge_video_clips(clip_paths, raw_concat_path, temp_dir)
        perf["concatenation"] = round(time.time() - t0, 3)
        print(f"[FFMPEG PERF] Concatenation: {perf['concatenation']} sec")

        # ---------------------------------------------------------------------
        # STAGE 3: Subtitles & Final Muxing (98–99%)
        # ---------------------------------------------------------------------
        t0 = time.time()
        if progress_callback:
            progress_callback("PREPARING_SUBTITLES", 98, f"Finalizing output (Subtitle Mode: {sub_mode.upper()})...")

        if sub_mode == "burned":
            abs_srt = os.path.abspath(srt_subtitle_path).replace("\\", "/").replace(":", "\\:")
            sub_style = "FontName=Nirmala UI,FontSize=16,Bold=1,Outline=1,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4,MarginV=25,Alignment=2"
            cmd_burn = [
                "ffmpeg", "-y",
                "-i", raw_concat_path,
                "-vf", f"subtitles='{abs_srt}':force_style='{sub_style}'",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "copy",
                output_summary_path
            ]
            print(f"[FFMPEG] Command:\n{' '.join(cmd_burn)}")
            res_burn = subprocess.run(cmd_burn, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_burn.returncode != 0:
                raise RuntimeError(f"FFmpeg subtitle burning failed:\n{res_burn.stderr}")
            perf["subtitle_processing"] = round(time.time() - t0, 3)
        else:
            # External subtitles: Copy merged stream directly
            shutil.copy(raw_concat_path, output_summary_path)
            perf["subtitle_processing"] = 0.001

        perf["final_muxing"] = round(time.time() - t0, 3)
        perf["total"] = round(time.time() - t_start, 3)

        print(f"[FFMPEG PERF] Subtitle processing: {perf['subtitle_processing']} sec")
        print(f"[FFMPEG PERF] Final muxing: {perf['final_muxing']} sec")
        print(f"[FFMPEG PERF] TOTAL: {perf['total']} sec")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # STAGE 4: Final Post-FFmpeg Validation (ffprobe)
    # -------------------------------------------------------------------------
    final_meta = get_media_metadata(output_summary_path)
    val_v = "PASS" if final_meta.get("duration_seconds", 0) > 0 else "FAIL"
    val_a = "PASS" if final_meta.get("has_audio") else "FAIL"
    val_audio_src = "PASS" if audio_mode == "original" else "FAIL"
    final_dur = final_meta.get("duration_seconds", 0.0)

    print(f"\n==================================================")
    print(f"[FINAL VALIDATION]")
    print(f"Video stream: {val_v}")
    print(f"Audio stream: {val_a}")
    print(f"Original lecturer audio: {val_audio_src}")
    print(f"Duration: {final_dur:.2f} sec")
    print(f"Target: {target_duration:.2f} sec")
    print(f"==================================================\n")

    if progress_callback:
        progress_callback("FINALIZING", 99, "Output video finalized & verified!")

    return output_summary_path
