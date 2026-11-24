import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, JOBS

print("[TEST] Testing result page template rendering...")
with app.test_client() as client:
    # Setup mock job in JOBS
    mock_job_id = "58ee4a46"
    JOBS[mock_job_id] = {
        "stage": "COMPLETED",
        "result": {
            "input_video": "uploads/91a4fcae_videoplayback_2.mp4",
            "output_summary_video": "outputs/91a4fcae_videoplayback_2_summary_58ee4a46.mp4",
            "original_duration": 892.74,
            "target_duration_sec": 300.0,
            "summary_duration": 297.79,
            "target_language": "kn",
            "caption_option": "original",
            "compression_percent": 66.6,
            "summary_text": "Software engineering characteristics and models.",
            "structured_sections": [
                {"title": "### Software Engineering", "bullets": ["Characteristics and models"], "start_time": 14.72}
            ],
            "summary_statements": [
                {"text": "Software engineering is systematic.", "start_time": 14.72, "end_time": 17.84, "similarity_score": 1.0}
            ],
            "timestamped_sentences": [
                {"text": "Software engineering is systematic.", "start_time": 14.72, "end_time": 17.84}
            ],
            "topic_timeline": [
                {"title": "### Introduction", "start_time": 14.72, "key_point": "Course overview"}
            ],
            "selected_clips": []
        }
    }

    res = client.get(f"/result/{mock_job_id}")
    print(f"[TEST] HTTP Response Status: {res.status_code}")
    assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}"
    print("[TEST SUCCESS] Result page rendered cleanly with 200 OK!")
