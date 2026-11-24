import os
import io
import json
import pytest
from app import JOBS

class TestWebRoutes:
    @pytest.fixture(autouse=True)
    def setup_mock_job(self):
        """Injects a completed mock job into JOBS store for testing results and downloads."""
        test_job_id = "test_mock_job_123"
        
        # Create a dummy output summary video and dummy files for download tests
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("outputs/subtitles", exist_ok=True)
        dummy_video = os.path.join("outputs", f"mock_summary_{test_job_id}.mp4")
        with open(dummy_video, "w") as f:
            f.write("mock video content")

        dummy_srt = os.path.join("outputs", "subtitles", f"{test_job_id}_en.srt")
        with open(dummy_srt, "w") as f:
            f.write("1\n00:00:00,000 --> 00:00:05,000\nMock Subtitle\n")

        JOBS[test_job_id] = {
            "job_id": test_job_id,
            "stage": "COMPLETED",
            "progress": 100,
            "message": "Completed",
            "elapsed_sec": 12.5,
            "result": {
                "job_id": test_job_id,
                "lecture_id": f"lec_{test_job_id}",
                "input_video": os.path.join("uploads", "sample_test.mp4"),
                "output_summary_video": dummy_video,
                "original_duration": 120.0,
                "target_duration_sec": 30.0,
                "summary_duration": 28.5,
                "compression_percent": 76.2,
                "target_language": "en",
                "caption_option": "original",
                "raw_transcript": "This is a raw transcript of the lecture.",
                "summary_text": "This is the generated summary.",
                "translated_summary_text": "This is the translated summary.",
                "summary_statements": [
                    {"statement_id": 1, "text": "Statement 1", "start_time": 0.0, "end_time": 10.0, "similarity_score": 0.85}
                ],
                "timestamped_sentences": [
                    {"sentence_id": 1, "text": "Sentence 1", "start_time": 0.0, "end_time": 10.0, "importance_score": 0.8, "educational_category": "concept", "topic_id": 1}
                ],
                "topic_timeline": ["Topic 1: Foundations", "Topic 2: Applications"],
                "topic_bullets": ["Topic 1: Foundations", "Topic 2: Applications"],
                "structured_sections": [
                    {"title": "Foundations", "content": "Basic lecture concepts explained."}
                ],
                "selected_clips": [
                    {"clip_id": 1, "start_time": 0.0, "end_time": 10.0, "duration": 10.0, "importance_score": 0.85}
                ],
                "srt_subtitle_file": dummy_srt,
                "metadata": {
                    "duration_seconds": 120.0,
                    "resolution": "1920x1080",
                    "width": 1920,
                    "height": 1080
                }
            }
        }
        yield
        # Cleanup
        if test_job_id in JOBS:
            del JOBS[test_job_id]

    def test_get_index(self, client):
        response = client.get('/')
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Lecture" in html or "Summarization" in html
        assert "target_duration" in html or "duration" in html

    def test_upload_missing_file(self, client):
        response = client.post('/api/upload', data={})
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_upload_invalid_extension(self, client):
        data = {
            'video': (io.BytesIO(b"fake text content"), 'invalid_file.txt')
        }
        response = client.post('/api/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        json_data = response.get_json()
        assert "Unsupported file format" in json_data.get("error", "")

    def test_url_missing_param(self, client):
        response = client.post('/api/url', json={})
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_summarize_missing_path(self, client):
        response = client.post('/api/summarize', json={})
        assert response.status_code == 400
        data = response.get_json()
        assert "Valid video_path is required" in data.get("error", "")

    def test_summarize_nonexistent_file(self, client):
        response = client.post('/api/summarize', json={'video_path': 'does_not_exist_file.mp4'})
        assert response.status_code == 400
        data = response.get_json()
        assert "Valid video_path is required" in data.get("error", "")

    def test_stream_nonexistent_job(self, client):
        response = client.get('/api/stream/nonexistent_job_999')
        assert response.status_code == 200
        data = response.get_data(as_text=True)
        assert "FAILED" in data
        assert "Job not found" in data

    def test_stream_completed_job(self, client):
        response = client.get('/api/stream/test_mock_job_123')
        assert response.status_code == 200
        data = response.get_data(as_text=True)
        assert "COMPLETED" in data
        assert "test_mock_job_123" in data

    def test_get_transcript_endpoint(self, client):
        # 404 for unknown job
        res_404 = client.get('/api/transcript/unknown_job')
        assert res_404.status_code == 404

        # 200 for completed job
        res_200 = client.get('/api/transcript/test_mock_job_123')
        assert res_200.status_code == 200
        data = res_200.get_json()
        assert "segments" in data
        assert len(data["segments"]) == 1

    def test_get_summary_endpoint(self, client):
        res_200 = client.get('/api/summary/test_mock_job_123')
        assert res_200.status_code == 200
        data = res_200.get_json()
        assert "summary" in data
        assert "statements" in data
        assert len(data["statements"]) == 1

    def test_get_clips_endpoint(self, client):
        res_200 = client.get('/api/clips/test_mock_job_123')
        assert res_200.status_code == 200
        data = res_200.get_json()
        assert "selected_clips" in data
        assert len(data["selected_clips"]) == 1

    def test_download_endpoint_types(self, client):
        # Invalid type
        res_bad = client.get('/api/download/test_mock_job_123/invalid_type')
        assert res_bad.status_code == 400

        # Transcript download
        res_txt = client.get('/api/download/test_mock_job_123/transcript')
        assert res_txt.status_code == 200

        # Summary download
        res_sum = client.get('/api/download/test_mock_job_123/summary')
        assert res_sum.status_code == 200

        # JSON download
        res_json = client.get('/api/download/test_mock_job_123/json')
        assert res_json.status_code == 200

        # Video download
        res_vid = client.get('/api/download/test_mock_job_123/video')
        assert res_vid.status_code == 200

        # Subtitles download
        res_srt = client.get('/api/download/test_mock_job_123/subtitles')
        assert res_srt.status_code == 200

    def test_view_result_page(self, client):
        # Missing or incomplete job redirects to '/'
        res_redirect = client.get('/result/unknown_job')
        assert res_redirect.status_code == 302

        # Completed job renders result.html
        res_dashboard = client.get('/result/test_mock_job_123')
        assert res_dashboard.status_code == 200
        html = res_dashboard.get_data(as_text=True)
        assert "test_mock_job_123" in html
        assert "Foundations" in html or "Summary" in html
