"""
===============================================================================
Module: Summarization Service (services/summarization_service.py)
===============================================================================
1. Purpose:
   Generates concise educational lecture summaries using Extractive, Abstractive,
   and Hybrid approaches with long-transcript chunking support.
2. Input:
   List of timestamped Sentence objects, topic sections, summary_type config, target_duration_sec.
3. Processing:
   - Chunking: Handles 1-2 hour lectures by chunking into semantic sections with overlap
   - Extractive: Selects top-scoring sentences respecting topic coverage and target word budget
   - Abstractive: Generates concise readable text via HuggingFace Transformers (sshleifer/distilbart-cnn-12-6)
   - Hybrid: Selects high-value source sentences, generates abstractive text, and maintains links back to source segments
4. Algorithm/Model:
   HuggingFace Transformers (distilbart-cnn-12-6) / Scikit-Learn TF-IDF TextRank.
5. Output:
   LectureSummary object containing full text, summary statements, and topic bullet points.
6. Data Stored:
   LectureSummary, SummaryStatement lists.
7. Failure Cases:
   Abstractive failure gracefully falls back to extractive selection; no text lost.
8. Evaluation:
   ROUGE-1, ROUGE-2, ROUGE-L and compression ratio metrics.
9. Integration:
   Passes summary statements to alignment_service.py for semantic timestamp retrieval.
===============================================================================
"""

import re
from typing import List, Dict, Any, Tuple
from models.transcript import Sentence
from models.summary import LectureSummary, SummaryStatement
from multimodal.text_summary import summarize_transcript_text

def generate_lecture_summary(
    sentences: List[Sentence],
    topic_sections: List[Dict[str, Any]],
    summary_type: str = "hybrid",
    target_duration_sec: float = 120.0,
    lecture_id: str = "lecture_001"
) -> LectureSummary:
    """Generates Extractive, Abstractive, or Hybrid lecture summary."""
    if not sentences:
        return LectureSummary(summary_id="sum_empty", lecture_id=lecture_id, text="No lecture content available.")

    full_transcript = " ".join([s.text for s in sentences])
    total_words = len(re.findall(r'\w+', full_transcript))
    target_words = max(25, int(target_duration_sec * 2.2))

    summary_type = summary_type.lower().strip()

    print(f"[SUMMARIZATION SERVICE] Generating '{summary_type.upper()}' summary for {total_words} transcript words (Target: ~{target_words} words for {target_duration_sec}s duration)...")

    # Group top sentences per topic section to guarantee topic coverage
    topic_top_sentences: Dict[int, List[Sentence]] = {}
    for s in sentences:
        topic_top_sentences.setdefault(s.topic_id, []).append(s)

    selected_sentences: List[Sentence] = []
    accumulated_words = 0

    # Coverage-aware selection: take highest scoring sentence from each topic first
    for tid, t_sents in topic_top_sentences.items():
        best_in_topic = max(t_sents, key=lambda x: x.importance_score)
        if best_in_topic not in selected_sentences:
            selected_sentences.append(best_in_topic)
            accumulated_words += len(re.findall(r'\w+', best_in_topic.text))

    # Fill remaining budget with highest scoring sentences across entire lecture
    remaining_sorted = sorted([s for s in sentences if s not in selected_sentences], key=lambda x: x.importance_score, reverse=True)
    for s in remaining_sorted:
        if accumulated_words >= target_words:
            break
        selected_sentences.append(s)
        accumulated_words += len(re.findall(r'\w+', s.text))

    # Re-sort chronologically
    selected_sentences.sort(key=lambda x: x.sentence_id)

    extractive_text = " ".join([s.text for s in selected_sentences])

    if summary_type == "extractive":
        summary_text = extractive_text
        statements = []
        for idx, s in enumerate(selected_sentences):
            stmt = SummaryStatement(
                statement_id=idx + 1,
                text=s.text,
                source_sentence_ids=[s.sentence_id],
                source_segment_ids=[s.segment_id],
                start_time=s.start_time,
                end_time=s.end_time,
                similarity_score=1.00
            )
            statements.append(stmt)

    elif summary_type == "abstractive":
        # Run abstractive summarizer across full lecture transcript
        res = summarize_transcript_text(full_transcript, target_duration_sec=target_duration_sec)
        summary_text = res.get("text_summary", extractive_text)

        # Split generated abstractive summary into statements
        raw_stmts = [st.strip() for st in re.split(r'(?<=[.!?])\s+', summary_text) if len(st.strip()) > 3]
        statements = []
        for idx, st_text in enumerate(raw_stmts):
            stmt = SummaryStatement(
                statement_id=idx + 1,
                text=st_text,
                source_sentence_ids=[s.sentence_id for s in selected_sentences],
                source_segment_ids=list(set([s.segment_id for s in selected_sentences]))
            )
            statements.append(stmt)

    else: # HYBRID (Default & Recommended)
        # Identify educational anchor sentences (definitions, formulas, concepts)
        anchors = [s for s in selected_sentences if s.educational_category in ("definition", "formula", "concept")]
        narratives = [s for s in selected_sentences if s not in anchors]

        # Summarize narrative explanations abstractively
        narrative_text = " ".join([s.text for s in narratives]) if narratives else extractive_text
        res = summarize_transcript_text(narrative_text, target_duration_sec=max(15.0, target_duration_sec * 0.6))
        condensed_narrative = res.get("text_summary", narrative_text)

        # Merge verbatim educational anchors with condensed explanations
        hybrid_parts = [a.text for a in anchors] + ([condensed_narrative] if condensed_narrative else [])
        summary_text = " ".join(hybrid_parts) if hybrid_parts else extractive_text

        raw_stmts = [st.strip() for st in re.split(r'(?<=[.!?])\s+', summary_text) if len(st.strip()) > 3]
        if not raw_stmts:
            raw_stmts = [summary_text]

        statements = []
        for idx, st_text in enumerate(raw_stmts):
            stmt = SummaryStatement(
                statement_id=idx + 1,
                text=st_text,
                source_sentence_ids=[s.sentence_id for s in selected_sentences],
                source_segment_ids=list(set([s.segment_id for s in selected_sentences]))
            )
            statements.append(stmt)

    # Build topic bullet points, structured sections, and topic timeline
    bullet_points = []
    structured_sections = []
    topic_timeline = []

    for t_info in topic_sections:
        t_id = t_info["topic_id"]
        t_title = t_info.get("title") or f"Topic {t_id}"
        start_ts = t_info.get("start_time", 0.0)
        end_ts = t_info.get("end_time", 0.0)

        t_sents = [s for s in selected_sentences if s.topic_id == t_id]
        if not t_sents:
            t_sents = [s for s in sentences if s.topic_id == t_id][:3]

        if t_sents:
            best = max(t_sents, key=lambda x: x.importance_score)
            bullets = [s.text.strip() for s in t_sents]

            bullet_points.append({
                "topic_id": t_id,
                "title": t_title,
                "bullet": best.text,
                "start_time": best.start_time,
                "end_time": best.end_time
            })

            structured_sections.append({
                "topic_id": t_id,
                "title": t_title,
                "start_time": start_ts,
                "end_time": end_ts,
                "bullets": bullets
            })

            topic_timeline.append({
                "topic_id": t_id,
                "title": t_title,
                "start_time": start_ts,
                "end_time": end_ts,
                "key_point": t_info.get("key_point") or best.text,
                "bullet": best.text
            })

    summary_obj = LectureSummary(
        summary_id=f"sum_{lecture_id}",
        lecture_id=lecture_id,
        text=summary_text,
        summary_type=summary_type,
        language="en",
        statements=statements,
        topic_bullets=bullet_points,
        structured_sections=structured_sections,
        topic_timeline=topic_timeline
    )

    print(f"[SUMMARIZATION SERVICE] Generated {summary_type} summary ({len(summary_text.split())} words, {len(statements)} statements, {len(structured_sections)} structured topic sections).")
    return summary_obj
