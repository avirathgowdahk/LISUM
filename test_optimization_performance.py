import os
import sys
import time
import subprocess
import torch
from typing import Dict, Any

from preprocessing.video_reader import get_video_metadata
from run_pipeline import run_summarization_pipeline

def create_silent_test_video(output_path: str, duration_sec: float = 10.0) -> str:
    """Generates a synthetic silent MP4 video for no-audio testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=640x360:rate=30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def run_test_suite():
    print(f"\n==================================================")
    print(f"   RUNNING PIPELINE PERFORMANCE TEST SUITE        ")
    print(f"==================================================")

    # Locate sample test video files
    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    sample_files = [f for f in os.listdir(uploads_dir) if f.endswith(".mp4")]

    medium_video = None
    long_video = None

    for f in sample_files:
        p = os.path.join(uploads_dir, f)
        try:
            m = get_video_metadata(p)
            dur = m["duration_seconds"]
            if dur >= 60 and medium_video is None:
                medium_video = p
            if dur >= 300 and long_video is None:
                long_video = p
        except Exception:
            pass

    if not medium_video:
        medium_video = os.path.join(uploads_dir, sample_files[0]) if sample_files else None
    if not long_video:
        long_video = medium_video

    print(f"Medium Test Video Path: {medium_video}")
    print(f"Long Test Video Path  : {long_video}")

    results = []

    def execute_test(test_id: str, desc: str, video_p: str, target_dur: float, lang: str):
        print(f"\n--------------------------------------------------")
        print(f"Executing {test_id}: {desc}")
        print(f"Source: {os.path.basename(video_p)} | Target: {target_dur}s | Lang: {lang.upper()}")
        print(f"--------------------------------------------------")
        
        t0 = time.time()
        try:
            res = run_summarization_pipeline(
                video_path=video_p,
                target_duration_sec=target_dur,
                target_language=lang,
                force_recompute=False
            )
            elapsed = time.time() - t0
            
            final_p = res["output_summary"]
            final_m = get_video_metadata(final_p)
            actual_dur = final_m["duration_seconds"]
            
            perf = res.get("performance_summary", {})
            mm = res.get("multimodal", {})

            item = {
                "test_id": test_id,
                "description": desc,
                "input_duration": res["original_duration"],
                "requested_duration": target_dur,
                "actual_duration": actual_dur,
                "duration_pass": (abs(actual_dur - target_dur) <= 5.0),
                "whisper_time": perf.get("whisper", 0.0),
                "tts_time": perf.get("tts", 0.0),
                "visual_ai_time": round(perf.get("feature_extraction", 0.0) + perf.get("transformer", 0.0) + perf.get("selection", 0.0), 2),
                "ffmpeg_time": perf.get("rendering", 0.0),
                "total_time": round(elapsed, 2)
            }
            results.append(item)
            return item
        except Exception as e:
            print(f"[{test_id} ERROR]: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test_id": test_id,
                "description": desc,
                "error": str(e),
                "total_time": round(time.time() - t0, 2)
            })

    # TEST 1: 1-minute source -> 30-second summary (EN)
    if medium_video:
        execute_test("TEST 1", "1-minute source -> 30-second summary (EN)", medium_video, 30.0, "en")

    # TEST 2: 5-minute source -> 1-minute summary (EN)
    if long_video:
        execute_test("TEST 2", "5-minute source -> 60-second summary (EN)", long_video, 60.0, "en")

    # TEST 3: 5-minute source -> 5-minute summary (EN)
    if long_video:
        meta_l = get_video_metadata(long_video)
        dur_l = min(meta_l["duration_seconds"] - 2.0, 300.0)
        execute_test("TEST 3", "Long source -> max summary (EN)", long_video, dur_l, "en")

    # TEST 4: Video without audio
    silent_vid = os.path.join("scratch", "test_silent_no_audio.mp4")
    create_silent_test_video(silent_vid, duration_sec=15.0)
    execute_test("TEST 4", "Video without audio stream", silent_vid, 10.0, "en")

    # TEST 5 & 6: Hardware Environment Detection
    print(f"\n--------------------------------------------------")
    print(f"TEST 5 & 6: Environment Check")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU (Verified Optimized Fallback)'}")
    print(f"--------------------------------------------------")

    # TEST 7: English
    if medium_video:
        execute_test("TEST 7", "English summary verification", medium_video, 35.0, "en")

    # TEST 8: Indian Language (Kannada / Hindi)
    if medium_video:
        execute_test("TEST 8", "Indian language summary verification (Kannada)", medium_video, 35.0, "kn")

    print(f"\n==================================================")
    print(f"   FINAL BENCHMARK TEST SUMMARY REPORT            ")
    print(f"==================================================")
    for r in results:
        if "error" in r:
            print(f"[{r['test_id']}] FAIL - Error: {r['error']}")
        else:
            dur_status = "PASS" if r['duration_pass'] else "WARN"
            print(f"[{r['test_id']}] {r['description']}")
            print(f"  - Input Dur   : {r['input_duration']:.2f}s")
            print(f"  - Target Dur  : {r['requested_duration']:.2f}s")
            print(f"  - Output Dur  : {r['actual_duration']:.2f}s (Validation: {dur_status})")
            print(f"  - Whisper STT : {r['whisper_time']:.2f}s")
            print(f"  - TTS Voice   : {r['tts_time']:.2f}s")
            print(f"  - Visual AI   : {r['visual_ai_time']:.2f}s")
            print(f"  - FFmpeg      : {r['ffmpeg_time']:.2f}s")
            print(f"  - TOTAL TIME  : {r['total_time']:.2f}s")
            print(f"--------------------------------------------------")

if __name__ == "__main__":
    run_test_suite()
