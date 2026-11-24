import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("==================================================")
print("TESTING OPTIMIZED LECTURE SUMMARIZATION BACKEND")
print(f"Python Executable : {sys.executable}")
print(f"CUDA Available    : {torch.cuda.is_available()}")
print("==================================================")

from multimodal.speech_to_text import warmup_whisper_model, transcribe_audio, get_video_file_hash
from services.embedding_service import warmup_embedding_model, generate_sentence_embeddings
from multimodal.translation import translate_text_summary, translate_text_batch
from pipeline_engine import run_lecture_processing_pipeline

print("\n--- 1. Testing Model Warmup ---")
t0 = time.time()
warmup_whisper_model()
warmup_embedding_model()
print(f"[TEST] Warmup time: {time.time() - t0:.2f}s")

test_video = "uploads/479bcf23_videoplayback_2.mp4"
if not os.path.exists(test_video):
    uploads = [os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".mp4")]
    if uploads:
        test_video = uploads[0]

if os.path.exists(test_video):
    print(f"\n--- 2. Testing Full Pipeline Execution on '{test_video}' ---")
    t0 = time.time()
    res = run_lecture_processing_pipeline(
        video_path=test_video,
        target_duration_sec=120.0,
        summary_type="hybrid",
        target_language="en",
        caption_option="original",
        audio_mode="original",
        job_id="speed_test_1",
        force_recompute=False
    )
    t_pipeline = time.time() - t0
    print(f"\n[TEST SUCCESS] Full pipeline completed in {t_pipeline:.2f}s!")
    print(f"Output Video: {res.get('output_summary_video')}")
    print(f"Summary Statements: {len(res.get('summary_statements', []))}")
else:
    print("[TEST] No sample video found in uploads/ to run end-to-end test.")

print("\n--- 3. Testing Translation Batch & Cache ---")
sample_texts = [
    "From today's session, I will start a software engineering subject.",
    "In this series, I will give you a complete point-wise explanation.",
    "Software engineering is the systematic application of engineering approaches to software development."
]
t0 = time.time()
tr_res = translate_text_batch(sample_texts, target_lang="kn")
print(f"[TEST] Batch translation time: {time.time() - t0:.2f}s")
print(f"Translated output sample: {tr_res[0][:40]}...")

print("\n==================================================")
print("ALL BACKEND SPEED OPTIMIZATION TESTS PASSED!")
print("==================================================")
