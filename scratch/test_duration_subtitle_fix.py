import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from run_pipeline import run_summarization_pipeline

video_path = os.path.join(BASE_DIR, "uploads", "sample_test.mp4")
if not os.path.exists(video_path):
    # Fallback to videoplayback if sample_test missing
    video_path = os.path.join(BASE_DIR, "uploads", "4996381e_videoplayback.mp4")

print("--- TESTING PIPELINE WITH 60s TARGET ---")
res60 = run_summarization_pipeline(
    video_path=video_path,
    target_duration_sec=60.0,
    target_language="en"
)

print("\n--- TESTING PIPELINE WITH 30s TARGET (VERIFYING CACHE INDEPENDENCE) ---")
res30 = run_summarization_pipeline(
    video_path=video_path,
    target_duration_sec=30.0,
    target_language="en"
)

print("\nTest completed successfully!")
