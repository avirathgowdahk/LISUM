import os
import pytest
import numpy as np

from models.transcript import TranscriptSegment, Sentence
from services.transcript_processor import process_and_segment_transcript
from services.lecture_preprocessor import classify_lecture_sentences
from services.embedding_service import generate_sentence_embeddings
from services.importance_service import compute_importance_scores
from services.redundancy_service import apply_redundancy_deduplication
from services.topic_service import segment_lecture_topics
from services.summarization_service import generate_lecture_summary
from services.alignment_service import align_summary_to_transcript
from services.segment_service import expand_clip_boundaries, select_optimal_video_segments
from services.subtitle_service import generate_subtitles_and_keywords
from services.translation_service import translate_lecture_summary
from services.video_processor import validate_and_inspect_video

class TestModularServices:
    def test_transcript_processor(self, sample_segments):
        sentences = process_and_segment_transcript(sample_segments)
        assert len(sentences) >= len(sample_segments)
        for s in sentences:
            assert isinstance(s, Sentence)
            assert s.text.strip() != ""
            assert s.start_time >= 0.0
            assert s.end_time >= s.start_time

    def test_lecture_preprocessor(self, sample_sentences):
        categorized = classify_lecture_sentences(sample_sentences)
        categories = [s.educational_category for s in categorized]
        assert "admin_noise" in categories
        assert "definition" in categories
        assert "concept" in categories
        assert "example" in categories
        assert "conclusion" in categories

    def test_embedding_service(self, sample_sentences):
        embedded = generate_sentence_embeddings(sample_sentences)
        for s in embedded:
            assert s.embedding is not None
            assert len(s.embedding) > 0
            assert isinstance(s.embedding, list)

    def test_importance_scoring(self, sample_sentences):
        embedded = generate_sentence_embeddings(sample_sentences)
        scored = compute_importance_scores(embedded)
        for s in scored:
            assert 0.0 <= s.importance_score <= 1.0
        # High-information sentences (definition, concept) should score higher than noise
        def_score = next(s.importance_score for s in scored if s.educational_category == "definition")
        noise_score = next(s.importance_score for s in scored if s.educational_category == "admin_noise")
        assert def_score > noise_score

    def test_redundancy_deduplication(self, sample_sentences):
        embedded = generate_sentence_embeddings(sample_sentences)
        scored = compute_importance_scores(embedded)
        deduped = apply_redundancy_deduplication(scored)
        assert len(deduped) == len(sample_sentences)
        assert all(hasattr(s, 'importance_score') for s in deduped)

    def test_topic_segmentation(self, sample_sentences):
        embedded = generate_sentence_embeddings(sample_sentences)
        segmented, topic_sections = segment_lecture_topics(embedded, target_num_topics=2)
        assert len(topic_sections) >= 1
        assert all(s.topic_id >= 0 for s in segmented)

    def test_summarization_service(self, sample_sentences):
        embedded = generate_sentence_embeddings(sample_sentences)
        scored = compute_importance_scores(embedded)
        segmented, topics = segment_lecture_topics(scored, target_num_topics=2)

        # Extractive Summary
        ext_sum = generate_lecture_summary(segmented, topics, summary_type="extractive", target_duration_sec=30.0)
        assert len(ext_sum.text) > 0
        assert len(ext_sum.statements) > 0

        # Hybrid Summary
        hyb_sum = generate_lecture_summary(segmented, topics, summary_type="hybrid", target_duration_sec=30.0)
        assert len(hyb_sum.text) > 0

    def test_alignment_service(self, sample_sentences):
        embedded = generate_sentence_embeddings(sample_sentences)
        scored = compute_importance_scores(embedded)
        segmented, topics = segment_lecture_topics(scored, target_num_topics=2)
        summary = generate_lecture_summary(segmented, topics, summary_type="hybrid", target_duration_sec=30.0)

        aligned = align_summary_to_transcript(summary, segmented)
        for stmt in aligned.statements:
            assert stmt.start_time >= 0.0
            assert stmt.end_time >= stmt.start_time
            assert 0.0 <= stmt.similarity_score <= 1.0

    def test_segment_selection_knapsack(self, sample_sentences):
        embedded = generate_sentence_embeddings(sample_sentences)
        scored = compute_importance_scores(embedded)
        segmented, topics = segment_lecture_topics(scored, target_num_topics=2)
        summary = generate_lecture_summary(segmented, topics, summary_type="hybrid", target_duration_sec=30.0)
        aligned = align_summary_to_transcript(summary, segmented)

        raw_intervals = [(stmt.start_time, stmt.end_time, stmt.similarity_score) for stmt in aligned.statements]
        clips, video_clips, total_dur = select_optimal_video_segments(raw_intervals, segmented, target_duration_sec=30.0)

        assert total_dur <= 35.0  # Respects budget with tolerance
        assert len(clips) >= 1
        for start_t, end_t in clips:
            assert start_t < end_t

    def test_subtitles_and_keywords(self):
        output_srt = os.path.join("outputs", "subtitles", "test_sample.srt")
        srt_path, keywords = generate_subtitles_and_keywords(
            summary_text="Composite primary keys consist of two or more columns in relational databases.",
            total_duration_sec=15.0,
            output_srt_path=output_srt
        )
        assert os.path.exists(srt_path)
        assert len(keywords) > 0
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "00:00:" in content

    def test_translation_service_english(self):
        text = "This is a sample summary."
        result = translate_lecture_summary(text, target_language="en")
        assert result == text

    def test_video_processor_invalid_file(self):
        with pytest.raises(Exception):
            validate_and_inspect_video("non_existent_file_path.mp4")

    def test_video_processor_valid_file(self):
        valid_path = os.path.join("uploads", "7507b4c1_videoplayback.mp4")
        if os.path.exists(valid_path):
            meta = validate_and_inspect_video(valid_path)
            assert meta["duration_seconds"] > 0
            assert meta["has_audio"] is True
            assert meta["fps"] > 0
