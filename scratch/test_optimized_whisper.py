import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import time
from multimodal.speech_to_text import transcribe_audio, get_whisper_model

def test_whisper_speed_and_progress():
    print("--- 1. Testing Whisper Model Singleton Initialization ---")
    model, backend, device, compute_type = get_whisper_model()
    print(f"Loaded: backend={backend}, device={device}, compute_type={compute_type}")
    assert backend == "faster-whisper", f"Expected faster-whisper, got {backend}"
    assert compute_type in ["int8", "float16", "float32"], f"Unexpected compute_type {compute_type}"

    # Find a test audio or video file
    test_files = [
        "cache/audio/164afa80_videoplayback_2_normalized.wav",
        "cache/audio/7507b4c1_videoplayback_normalized.wav",
        "uploads/7507b4c1_videoplayback.mp4"
    ]
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break

    if not test_file:
        print("No test audio/video file found to test.")
        return

    print(f"\n--- 2. Testing transcribe_audio on '{test_file}' ---")
    progress_records = []
    def progress_handler(stage, pct, msg):
        progress_records.append((stage, pct, msg))
        first_line = msg.split('\n')[0]
        second_line = msg.split('\n')[1] if '\n' in msg else ''
        print(f"[Callback] Stage: {stage} | Progress: {pct}% | Msg: {first_line} {second_line}")

    t0 = time.time()
    res = transcribe_audio(
        test_file,
        output_dir="outputs/multimodal",
        force_recompute=True,
        progress_callback=progress_handler
    )
    t_first = time.time() - t0

    assert res.get("success"), f"Transcription failed: {res.get('error')}"
    assert len(res.get("segments", [])) > 0, "No segments generated"
    assert len(res.get("transcript_text", "")) > 0, "No transcript generated"
    
    # Verify word timestamps
    first_seg = res["segments"][0]
    assert "words" in first_seg, "Segment missing words"
    assert "start" in first_seg and "end" in first_seg, "Segment missing start/end timestamps"
    print(f"Verified segment with {len(first_seg['words'])} words with timestamps.")
    print(f"Sample words: {first_seg['words'][:3]}")

    print(f"\nFirst transcription finished in {t_first:.2f}s with {len(progress_records)} progress updates.")
    assert len(progress_records) > 0, "Expected progress updates"

    # Test cache hit
    print("\n--- 3. Testing Transcription Cache Hit ---")
    t0_c = time.time()
    cache_res = transcribe_audio(
        test_file,
        output_dir="outputs/multimodal",
        force_recompute=False
    )
    t_cached = time.time() - t0_c
    print(f"Cached transcription returned in {t_cached:.4f}s (Cache flag: {cache_res.get('cached')})")
    assert cache_res.get("cached") is True or t_cached < 0.5, "Expected instant cache hit"

    print("\n ALL WHISPER OPTIMIZATIONS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_whisper_speed_and_progress()
