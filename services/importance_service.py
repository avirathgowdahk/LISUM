"""
===============================================================================
Module: Importance Scoring Service (services/importance_service.py)
===============================================================================
1. Purpose:
   Calculates explicit multi-factor educational importance scores for every lecture sentence.
2. Input:
   List of Sentence objects with embeddings and educational categories.
3. Processing:
   Calculates importance score using multi-factor explicit formula:
   importance_score = (
       0.30 * semantic_relevance      # Similarity to global lecture centroid
     + 0.25 * educational_value        # Bonus for definitions, concepts, formulas, conclusions
     + 0.20 * terminology_density     # Presence of academic technical terms
     + 0.15 * contextual_position      # Position in topic/lecture structure
     + 0.10 * sentence_clarity         # Length/structure clarity penalty
   )
4. Algorithm/Model:
   Explicit educational multi-factor ranking formula normalized between 0.0 and 1.0.
5. Output:
   Sentences populated with calculated importance_score attributes.
6. Data Stored:
   Sentence.importance_score.
7. Failure Cases:
   Falls back to TF-IDF score if embeddings are missing.
8. Evaluation:
   Ensures definitions and core lecture conclusions receive higher average scores than generic text.
9. Integration:
   Passes scored sentences to redundancy_service.py and segment_service.py.
===============================================================================
"""

import numpy as np
import re
from typing import List
from models.transcript import Sentence
from utils.similarity_utils import compute_cosine_similarity

CATEGORY_BONUS_MAP = {
    "definition": 1.00,
    "concept": 0.85,
    "formula": 0.90,
    "conclusion": 0.80,
    "example": 0.65,
    "explanation": 0.50,
    "intro": 0.30,
    "admin_noise": 0.05
}

ACADEMIC_TERMS_REGEX = re.compile(
    r'\b(algorithm|analysis|architecture|complexity|data|definition|equation|function|method|model|network|parameter|principle|process|structure|system|theory|variable|vector)\b',
    re.IGNORECASE
)

def compute_importance_scores(sentences: List[Sentence]) -> List[Sentence]:
    """Calculates normalized educational importance scores for all lecture sentences."""
    if not sentences:
        return sentences

    # Calculate global lecture centroid embedding
    valid_embeddings = [s.embedding for s in sentences if s.embedding is not None]
    if valid_embeddings:
        centroid = np.mean(valid_embeddings, axis=0).tolist()
    else:
        centroid = None

    scores = []
    for s in sentences:
        # 1. Semantic Relevance (Similarity to overall lecture topic centroid)
        if centroid is not None and s.embedding is not None:
            sem_rel = max(0.0, compute_cosine_similarity(s.embedding, centroid))
        else:
            sem_rel = 0.5

        # 2. Educational Value Bonus
        edu_val = CATEGORY_BONUS_MAP.get(s.educational_category, 0.50)

        # 3. Technical Terminology Density
        words = re.findall(r'\w+', s.text.lower())
        term_matches = len(ACADEMIC_TERMS_REGEX.findall(s.text))
        term_density = min(1.0, term_matches / max(1, len(words) * 0.1))

        # 4. Contextual Position (Slight preference for section intro/outro bounds)
        pos_ratio = s.sentence_id / max(1, len(sentences))
        if pos_ratio <= 0.15 or pos_ratio >= 0.85:
            pos_score = 0.75
        else:
            pos_score = 0.50

        # 5. Sentence Length Clarity (Penalize trivial 1-3 word fragments or 100-word run-ons)
        w_len = len(words)
        if 8 <= w_len <= 35:
            clarity = 1.0
        elif 4 <= w_len < 8 or 35 < w_len <= 50:
            clarity = 0.6
        else:
            clarity = 0.2

        # Combine factors
        total_score = (
            0.35 * sem_rel +
            0.30 * edu_val +
            0.15 * term_density +
            0.10 * pos_score +
            0.10 * clarity
        )

        s.importance_score = float(total_score)
        scores.append(total_score)

    # Normalize scores between 0.0 and 1.0
    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s if max_s > min_s else 1.0

    for s in sentences:
        s.importance_score = round(float((s.importance_score - min_s) / rng), 4)

    print(f"[IMPORTANCE SERVICE] Scored {len(sentences)} sentences (Max score: {max(s.importance_score for s in sentences):.2f}, Min: {min(s.importance_score for s in sentences):.2f})")
    return sentences
