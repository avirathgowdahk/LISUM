import os
import sys
import time
import json

# Force UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline import run_summarization_pipeline
from multimodal.pipeline import run_multimodal_pipeline

def profile_complete_system(video_path: str, target_ratio: float = 0.20):
    print("==================================================")
    print("   COMPLETE SYSTEM PERFORMANCE & ACCURACY TEST   ")
    print("==================================================")
    print(f"Testing Video: {video_path}")

    start_all = time.time()

    # 1. Run Video-to-Video Summarization Pipeline
    print("\n--- STAGE 1: CORE VIDEO-TO-VIDEO SUMMARIZATION ---")
    sum_res = run_summarization_pipeline(video_path, target_ratio=target_ratio)
    summary_video = sum_res["output_summary"]
    selected_segments = sum_res["selected_segments"]

    # 2. Run Multimodal Enhancement Pipeline
    print("\n--- STAGE 2: MULTIMODAL ENHANCEMENT PIPELINE ---")
    mm_res = run_multimodal_pipeline(
        summary_video,
        selected_segments=selected_segments,
        original_video_path=video_path,
        force_recompute=True
    )

    total_all = time.time() - start_all

    print("\n==================================================")
    print("   END-TO-END TIMING & PROFILING REPORT           ")
    print("==================================================")
    print(f"Input Video Duration : {sum_res['original_duration']}s")
    print(f"Summary Duration     : {sum_res['summary_duration']}s ({sum_res['compression_percent']}% compressed)")
    print(f"\n[TIMING BREAKDOWN]")
    print(f"  1. Video Summarization : {sum_res.get('elapsed_seconds', 0)}s")
    if mm_res.get("success"):
        perf = mm_res.get("performance", {})
        print(f"  2. Whisper STT        : {perf.get('time_stt_sec')}s")
        print(f"  3. Text Summarizer    : {perf.get('time_summary_sec')}s")
        print(f"  4. Google Translations: {perf.get('time_translation_sec')}s")
        print(f"  5. TTS Voiceovers (x3): {perf.get('time_tts_sec')}s")
        print(f"  MULTIMODAL TOTAL      : {perf.get('total_time_sec')}s")
    print(f"--------------------------------------------------")
    print(f"TOTAL SYSTEM EXECUTION TIME: {total_all:.2f}s")
    print("==================================================")

    if mm_res.get("success"):
        stats = mm_res.get("stats", {})
        print("\n[TEXT SUMMARIZATION COMPRESSION STATS]")
        print(f"  - Original Transcript Words: {stats.get('transcript_words')}")
        print(f"  - Extracted Summary Words   : {stats.get('summary_words')}")
        print(f"  - Text Compression Ratio    : {stats.get('compression_ratio_pct')}%")

        print("\n[RECOGNIZED TRANSCRIPT SAMPLE]")
        print(f"  \"{mm_res.get('transcript')[:300]}...\"")

        print("\n[EXTRACTED SUMMARY SAMPLE]")
        print(f"  \"{mm_res.get('text_summary')}\"")

        print("\n[TRANSLATION SAMPLES]")
        for lang, item in mm_res.get("multilingual", {}).items():
            if item.get("success"):
                print(f"  [{lang.upper()}] ({item.get('audio_duration_sec')}s audio): {item.get('translated_text')[:150]}...")

    return {
        "video_duration_sec": sum_res['original_duration'],
        "total_time_sec": round(total_all, 2),
        "processing_ratio": round(total_all / max(1, sum_res['original_duration']), 2)
    }

if __name__ == "__main__":
    test_file = os.path.join("uploads", "7507b4c1_videoplayback.mp4")
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    profile_complete_system(test_file)
