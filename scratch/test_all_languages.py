import os
import sys
import time

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import SUPPORTED_LANGUAGES
from run_pipeline import run_summarization_pipeline

def test_all_languages(video_path: str, target_duration_sec: float = 30.0):
    print("=" * 60)
    print("      COMPREHENSIVE MULTILINGUAL TEST SUITE FOR ALL LANGUAGES")
    print("=" * 60)
    print(f"Sample Video Path: {video_path}")
    print(f"Target Duration  : {target_duration_sec}s")
    print(f"Supported Languages ({len(SUPPORTED_LANGUAGES)}): {list(SUPPORTED_LANGUAGES.keys())}")
    print("=" * 60)

    results = {}

    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"\n" + "-" * 50)
        print(f" TESTING LANGUAGE: {code.upper()} ({name}) ")
        print("-" * 50)

        t0 = time.time()
        out_vid = os.path.join("outputs", f"test_all_lang_{code}_{int(target_duration_sec)}s.mp4")

        try:
            res = run_summarization_pipeline(
                video_path=video_path,
                target_duration_sec=target_duration_sec,
                target_language=code,
                output_path=out_vid,
                force_recompute=True
            )

            mm = res.get("multimodal", {})
            audio_file = mm.get("audio_file")
            audio_dur = mm.get("audio_duration_sec", 0.0)
            srt_file = mm.get("srt_file")
            translated_text = mm.get("translated_text", "")
            final_summary_video = res.get("output_summary")
            summary_dur = res.get("summary_duration", 0.0)

            audio_ok = audio_file is not None and os.path.exists(audio_file) and audio_dur > 0.5
            srt_ok = srt_file is not None and os.path.exists(srt_file) and os.path.getsize(srt_file) > 0
            video_ok = final_summary_video is not None and os.path.exists(final_summary_video) and os.path.getsize(final_summary_video) > 1000

            success = res.get("success", False) or (audio_ok and srt_ok and video_ok)

            results[code] = {
                "name": name,
                "success": success,
                "translated_text_snippet": translated_text[:80] + "..." if len(translated_text) > 80 else translated_text,
                "audio_duration_sec": round(audio_dur, 2),
                "audio_ok": audio_ok,
                "srt_ok": srt_ok,
                "video_ok": video_ok,
                "summary_duration_sec": round(summary_dur, 2),
                "elapsed_sec": round(time.time() - t0, 2)
            }

            print(f"[TEST RESULT {code.upper()}] Success: {success} | Audio: {audio_dur:.2f}s | Summary Vid: {summary_dur:.2f}s | Elapsed: {time.time() - t0:.2f}s")

        except Exception as e:
            import traceback
            traceback.print_exc()
            results[code] = {
                "name": name,
                "success": False,
                "error": str(e),
                "elapsed_sec": round(time.time() - t0, 2)
            }

    print("\n" + "=" * 70)
    print("              FINAL MULTILINGUAL TEST RESULTS REPORT")
    print("=" * 70)
    fmt = "{:<6} | {:<18} | {:<8} | {:<10} | {:<8} | {:<9} | {:<8}"
    print(fmt.format("Code", "Language Name", "Status", "Audio Dur", "SRT OK", "Video OK", "Time (s)"))
    print("-" * 70)

    all_passed = True
    for code, info in results.items():
        status = "PASS" if info.get("success") else "FAIL"
        if not info.get("success"):
            all_passed = False
        a_dur = f"{info.get('audio_duration_sec', 0)}s"
        srt_s = "YES" if info.get("srt_ok") else "NO"
        vid_s = "YES" if info.get("video_ok") else "NO"
        t_s = f"{info.get('elapsed_sec', 0)}s"
        print(fmt.format(code.upper(), info['name'].split()[0], status, a_dur, srt_s, vid_s, t_s))

    print("=" * 70)
    print(f"OVERALL ALL LANGUAGES TEST RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)

if __name__ == "__main__":
    sample_video = "uploads/0f729125_videoplayback.mp4"
    if not os.path.exists(sample_video):
        for f in os.listdir("uploads"):
            if f.endswith(".mp4"):
                sample_video = os.path.join("uploads", f)
                break
    test_all_languages(sample_video, target_duration_sec=30.0)
