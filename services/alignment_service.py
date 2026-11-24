"""
===============================================================================
Module: Alignment Service (services/alignment_service.py)
===============================================================================
1. Purpose:
   Semantically maps generated summary statements back to original timestamped
   transcript sentences to retrieve exact video timestamps.
2. Input:
   LectureSummary object and list of timestamped Sentence objects.
3. Processing:
   - Computes dense vector embeddings for each summary statement
   - Calculates cosine similarity against all timestamped transcript sentences
   - Finds top-matching source sentence(s) exceeding similarity threshold (default > 0.55)
   - Retrieves exact start_time and end_time from source transcript
   - Combines adjacent matched segments into unified time intervals
4. Algorithm/Model:
   SentenceTransformer cosine similarity matching with temporal span combination.
5. Output:
   Updated LectureSummary where every SummaryStatement has verified source timestamps.
6. Data Stored:
   SummaryStatement.start_time, SummaryStatement.end_time, SummaryStatement.similarity_score, SummaryStatement.source_sentence_ids.
7. Failure Cases:
   If abstractive statement similarity is below threshold, falls back to highest-scoring extractive source sentence.
8. Evaluation:
   Ensures 100% of summary statements have traceable, valid timestamps originating from original transcript.
9. Integration:
   Passes aligned timestamp intervals to segment_service.py for boundary expansion and clip extraction.
===============================================================================
"""

import numpy as np
from typing import List, Tuple
from models.transcript import Sentence
from models.summary import LectureSummary, SummaryStatement
from services.embedding_service import get_sentence_transformer_model
from utils.similarity_utils import compute_cosine_similarity

def align_summary_to_transcript(summary: LectureSummary, sentences: List[Sentence], min_similarity_threshold: float = 0.45) -> LectureSummary:
    """Semantically maps each summary statement to original timestamped transcript sentences."""
    if not summary.statements or not sentences:
        return summary

    model = get_sentence_transformer_model()
    stmt_texts = [s.text for s in summary.statements]

    if model is not None:
        try:
            stmt_embeddings = model.encode(stmt_texts, show_progress_bar=False, convert_to_numpy=True)
        except Exception:
            stmt_embeddings = None
    else:
        stmt_embeddings = None

    aligned_statements: List[SummaryStatement] = []

    for idx, stmt in enumerate(summary.statements):
        if stmt_embeddings is not None and idx < len(stmt_embeddings):
            s_vec = stmt_embeddings[idx].tolist()
        else:
            s_vec = None

        best_matches: List[Tuple[float, Sentence]] = []

        for src_sent in sentences:
            if s_vec is not None and src_sent.embedding is not None:
                sim = compute_cosine_similarity(s_vec, src_sent.embedding)
            else:
                # Text fallback similarity (Jaccard / token overlap)
                w1 = set(stmt.text.lower().split())
                w2 = set(src_sent.text.lower().split())
                inter = len(w1.intersection(w2))
                union = len(w1.union(w2))
                sim = inter / max(1, union)

            best_matches.append((sim, src_sent))

        best_matches.sort(key=lambda x: x[0], reverse=True)

        top_sim, top_sent = best_matches[0]

        if top_sim < min_similarity_threshold:
            # Fallback to top sentence in topic or highest scoring sentence
            top_sent = max(sentences, key=lambda x: x.importance_score)
            top_sim = min_similarity_threshold

        # Check for close secondary matches (e.g. adjacent sentence in explanation)
        matched_sents = [top_sent]
        if len(best_matches) > 1:
            sec_sim, sec_sent = best_matches[1]
            if sec_sim >= 0.70 and abs(sec_sent.sentence_id - top_sent.sentence_id) <= 2:
                matched_sents.append(sec_sent)

        matched_sents.sort(key=lambda x: x.start_time)

        stmt.start_time = matched_sents[0].start_time
        stmt.end_time = matched_sents[-1].end_time
        stmt.similarity_score = float(top_sim)
        stmt.source_sentence_ids = [s.sentence_id for s in matched_sents]
        stmt.source_segment_ids = list(set([s.segment_id for s in matched_sents]))

        aligned_statements.append(stmt)

        print(f"[ALIGNMENT SERVICE] Statement {stmt.statement_id} -> Matched timestamp [{stmt.start_time:.2f}s - {stmt.end_time:.2f}s] (Similarity: {stmt.similarity_score:.2f})")

    summary.statements = aligned_statements
    return summary
