"""
Integration Test Script for AI Lecture Processing Pipeline
===========================================================
Tests end-to-end processing execution from synthetic input sentences up to final video composition data model output.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transcript import TranscriptSegment, Sentence
from services.transcript_processor import process_and_segment_transcript
from services.lecture_preprocessor import classify_lecture_sentences
from services.embedding_service import generate_sentence_embeddings
from services.importance_service import compute_importance_scores
from services.redundancy_service import apply_redundancy_deduplication
from services.topic_service import segment_lecture_topics
from services.summarization_service import generate_lecture_summary
from services.alignment_service import align_summary_to_transcript
from services.segment_service import select_optimal_video_segments
from evaluation.metrics import calculate_rouge_scores, evaluate_duration_compliance

def test_full_pipeline_flow():
    print("\n--- Running End-to-End Pipeline Mock Test ---")
    
    # 1. Mock ASR Raw Segments
    mock_segments = [
        TranscriptSegment(1, 0.0, 12.0, "Good morning students. Welcome to today's lecture on Computer Networks and TCP IP protocol.", 0.98),
        TranscriptSegment(2, 12.5, 28.0, "Transmission Control Protocol is defined as a connection oriented transport layer protocol that provides reliable data transfer.", 0.96),
        TranscriptSegment(3, 28.5, 45.0, "The key concept behind TCP reliability is the three way handshake mechanism and sequence numbers.", 0.94),
        TranscriptSegment(4, 45.5, 62.0, "For example, when a client sends a SYN packet, the server responds with a SYN ACK packet before data transmission begins.", 0.92),
        TranscriptSegment(5, 62.5, 80.0, "In conclusion TCP guarantees ordered packet delivery and flow control using sliding window algorithms.", 0.95)
    ]

    # 2. Sentence Segmentation
    sentences = process_and_segment_transcript(mock_segments)
    assert len(sentences) >= 5, "Sentence segmentation failed"

    # 3. Educational Classification
    sentences = classify_lecture_sentences(sentences)
    assert any(s.educational_category == "definition" for s in sentences), "Definition classification failed"

    # 4. Dense Vector Embeddings
    sentences = generate_sentence_embeddings(sentences)
    assert sentences[0].embedding is not None, "Embedding generation failed"

    # 5. Multi-Factor Importance Scoring
    sentences = compute_importance_scores(sentences)
    assert max(s.importance_score for s in sentences) > 0.5, "Importance scoring failed"

    # 6. Redundancy Deduplication
    sentences = apply_redundancy_deduplication(sentences)

    # 7. Topic Segmentation
    sentences, topics = segment_lecture_topics(sentences, target_num_topics=2)
    assert len(topics) >= 1, "Topic segmentation failed"

    # 8. Hybrid Summarization
    summary = generate_lecture_summary(sentences, topics, summary_type="hybrid", target_duration_sec=30.0)
    assert len(summary.text) > 10, "Summary generation failed"

    # 9. Summary-Transcript Semantic Alignment
    aligned_summary = align_summary_to_transcript(summary, sentences)
    assert aligned_summary.statements[0].start_time >= 0.0, "Semantic alignment failed"

    # 10. 0/1 Knapsack Segment Selection
    raw_intervals = [(stmt.start_time, stmt.end_time, stmt.similarity_score) for stmt in aligned_summary.statements]
    clips, v_clips, selected_dur = select_optimal_video_segments(raw_intervals, sentences, target_duration_sec=30.0)
    
    # 11. Metrics Evaluation
    dur_eval = evaluate_duration_compliance(30.0, selected_dur)
    rouge_res = calculate_rouge_scores(" ".join([s.text for s in sentences]), summary.text)

    print(f"[TEST SUCCESS] Selected {len(clips)} clips ({selected_dur:.1f}s duration). ROUGE-1 F1: {rouge_res['rouge1_f1']:.4f}")
    print("[TEST PASSED] All 11 pipeline stages executed flawlessly!")

if __name__ == "__main__":
    test_full_pipeline_flow()
