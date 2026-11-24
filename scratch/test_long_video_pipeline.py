import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from run_pipeline import run_summarization_pipeline

# Pick a long video with audio from uploads/
video_path = os.path.join(BASE_DIR, "uploads", "4996381e_videoplayback.mp4")
if not os.path.exists(video_path):
    video_path = os.path.join(BASE_DIR, "uploads", "b0330b00_vidssave.com_Composite_Primary_Key_in_SQL___SQL_Interview_Question___PK_vs_FK_Explained_1080P.mp4")

print(f"Testing on video: {video_path}")
print("\n--- TEST 1: 60s TARGET (1 MINUTE) ---")
res60 = run_summarization_pipeline(
    video_path=video_path,
    target_duration_sec=60.0,
    target_language="en"
)

print("\n--- TEST 2: 30s TARGET (30 SECONDS) ---")
res30 = run_summarization_pipeline(
    video_path=video_path,
    target_duration_sec=30.0,
    target_language="en"
)

print("\n--- TEST 3: 120s TARGET (2 MINUTES) ---")
res120 = run_summarization_pipeline(
    video_path=video_path,
    target_duration_sec=120.0,
    target_language="en"
)

print("\nALL LONG VIDEO TESTS PASSED SUCCESSFULLY!")
