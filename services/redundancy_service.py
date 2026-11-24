"""
===============================================================================
Module: Redundancy Service (services/redundancy_service.py)
===============================================================================
1. Purpose:
   Detects semantically duplicate or repetitive lecture sentences and applies
   diversity penalties to select the clearest, highest-information occurrences.
2. Input:
   List of scored Sentence objects with embedding vectors.
3. Processing:
   - Calculates pairwise cosine similarity matrix
   - Groups sentences exceeding similarity threshold (> 0.82)
   - Retains highest-scoring candidate while reducing score of duplicate variants
4. Algorithm/Model:
   Cosine similarity matrix clustering with Maximal Marginal Relevance (MMR) penalty.
5. Output:
   Sentences with adjusted importance scores reflecting redundancy penalties.
6. Data Stored:
   Updated Sentence.importance_score.
7. Failure Cases:
   Preserves progressive educational explanations if context develops over time.
8. Evaluation:
   Reduces inter-sentence redundancy in selected summary content.
9. Integration:
   Feeds clean scored sentences to topic_service.py and segment_service.py.
===============================================================================
"""

import numpy as np
from typing import List
from models.transcript import Sentence
from utils.similarity_utils import compute_cosine_similarity

def apply_redundancy_deduplication(sentences: List[Sentence], similarity_threshold: float = 0.82) -> List[Sentence]:
    """Applies Maximal Marginal Relevance (MMR) style redundancy penalty to sentence scores."""
    if len(sentences) <= 1:
        return sentences

    # Sort indices by initial importance score descending
    sorted_sents = sorted(sentences, key=lambda x: x.importance_score, reverse=True)
    selected: List[Sentence] = []

    for cand in sorted_sents:
        if cand.embedding is None:
            selected.append(cand)
            continue

        is_redundant = False
        for prev in selected:
            if prev.embedding is None:
                continue
            sim = compute_cosine_similarity(cand.embedding, prev.embedding)

            # If highly similar, check temporal distance
            time_gap = abs(cand.start_time - prev.start_time)
            if sim > similarity_threshold:
                if time_gap < 60.0:  # Immediate repetition within 1 minute
                    is_redundant = True
                    # Reduce score of candidate
                    cand.importance_score *= 0.40
                    break
                else:  # Progressive explanation later in lecture
                    cand.importance_score *= 0.80

        selected.append(cand)

    # Re-sort sentences back to chronological order
    sentences.sort(key=lambda x: x.sentence_id)
    print(f"[REDUNDANCY SERVICE] Applied redundancy penalties across {len(sentences)} sentences.")
    return sentences
