"""
Benchmark script for FFmpeg video composition performance.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_processing.video_merger import generate_summary_video
from utils.ffmpeg_utils import get_media_metadata

def run_benchmark():
    video_dir = "uploads"
    test_video = None

    for fname in os.listdir(video_dir):
        if fname.endswith(".mp4") and "Composite" in fname:
            test_video = os.path.join(video_dir, fname)
            break

    if not test_video:
        for fname in os.listdir(video_dir):
            if fname.endswith(".mp4") and os.path.getsize(os.path.join(video_dir, fname)) > 5000000:
                test_video = os.path.join(video_dir, fname)
                break

    if not test_video:
        print("[ERROR] No test video found in uploads/")
        return

    meta = get_media_metadata(test_video)
    duration = meta["duration_seconds"]

    print(f"\n==================================================")
    print(f"   STARTING FFMPEG PERFORMANCE BENCHMARK          ")
    print(f"   Input Video: {test_video}")
    print(f"   Original Duration: {duration:.2f}s ({int(duration//60)}m {int(duration%60)}s)")
    print(f"   Resolution: {meta.get('resolution', 'unknown')}")
    print(f"==================================================\n")

    # Selected 5 clips totaling ~120s from a 15-minute video
    selected_segments = [
        (14.0, 45.0),
        (108.0, 150.0),
        (250.0, 290.0),
        (400.0, 430.0),
        (600.0, 620.0)
    ]

    out_path = os.path.join("outputs", "benchmark_summary_fast.mp4")
    if os.path.exists(out_path):
        os.remove(out_path)

    t_before = time.time()

    def cb(stage, pct, msg):
        print(f"[{pct:3d}%] [{stage}] {msg}")

    result_path = generate_summary_video(
        input_video=test_video,
        selected_segments=selected_segments,
        output_summary_path=out_path,
        audio_mode="original",
        burn_subtitles=False,
        progress_callback=cb
    )

    t_after = time.time()
    ffmpeg_sec = round(t_after - t_before, 2)

    final_meta = get_media_metadata(result_path)
    final_dur = final_meta.get("duration_seconds", 0)

    print(f"\n==================================================")
    print(f"   BENCHMARK RESULT SUMMARY                       ")
    print(f"   Before FFmpeg Time : {t_before:.2f}")
    print(f"   After FFmpeg Time  : {t_after:.2f}")
    print(f"   FFmpeg Render Time : {ffmpeg_sec} seconds")
    print(f"   Output Duration    : {final_dur:.2f} seconds")
    print(f"   Output File Size   : {os.path.getsize(result_path) / 1024 / 1024:.2f} MB")
    print(f"==================================================\n")

if __name__ == "__main__":
    run_benchmark()
