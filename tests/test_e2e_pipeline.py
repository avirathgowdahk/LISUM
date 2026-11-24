import os
import pytest
from pipeline_engine import run_lecture_processing_pipeline
from evaluation.metrics import evaluate_duration_compliance

class TestEndToEndPipeline:
    def test_pipeline_execution_and_caching(self):
        video_path = os.path.join("uploads", "7507b4c1_videoplayback.mp4")
        if not os.path.exists(video_path):
            pytest.skip("Test video file not found in uploads/")

        target_dur = 7.0
        job_id = "pytest_e2e_job"

        # Execute pipeline (uses precomputed cache for fast CI/CD execution)
        result = run_lecture_processing_pipeline(
            video_path=video_path,
            target_duration_sec=target_dur,
            summary_type="hybrid",
            target_language="en",
            caption_option="original",
            audio_mode="original",
            job_id=job_id,
            force_recompute=False
        )

        assert result is not None
        assert os.path.normpath(result["input_video"]) == os.path.normpath(video_path)
        assert os.path.exists(result["output_summary_video"])
        assert os.path.getsize(result["output_summary_video"]) > 0

        # Validate duration compliance
        compliance = evaluate_duration_compliance(target_dur, result["summary_duration"])
        assert compliance["compliant"] is True

        # Validate transcripts and sentences
        assert len(result["raw_transcript"]) > 0
        assert len(result["timestamped_sentences"]) > 0
        assert len(result["summary_statements"]) > 0
        assert len(result["selected_clips"]) > 0

        # Validate subtitles
        assert os.path.exists(result["srt_subtitle_file"])
        assert len(result["keywords"]) > 0

        # Validate performance times recorded
        perf = result.get("performance_summary", {})
        assert "TOTAL" in perf
