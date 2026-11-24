"""
===============================================================================
Module: Similarity Utilities (utils/similarity_utils.py)
===============================================================================
1. Purpose:
   Provides vector and text semantic similarity calculation tools.
2. Input:
   Numpy embedding vectors, text strings, list of text sentences.
3. Processing:
   - Cosine similarity between 1D/2D vectors
   - TF-IDF matrix generation and text similarity ranking
4. Output:
   Float similarity scores (0.0 to 1.0), similarity matrices.
===============================================================================
"""

import numpy as np
from typing import List, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(vec1: Union[np.ndarray, List[float]], vec2: Union[np.ndarray, List[float]]) -> float:
    """Computes normalized cosine similarity between two 1D vectors."""
    v1 = np.array(vec1, dtype=np.float32).flatten()
    v2 = np.array(vec2, dtype=np.float32).flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    sim = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(sim, -1.0, 1.0))

def compute_tfidf_similarity_matrix(sentences: List[str]) -> np.ndarray:
    """Computes pair-wise TF-IDF cosine similarity matrix for a list of sentences."""
    if not sentences:
        return np.zeros((0, 0), dtype=np.float32)
    if len(sentences) == 1:
        return np.ones((1, 1), dtype=np.float32)

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        return cosine_similarity(tfidf_matrix, tfidf_matrix)
    except Exception:
        # Fallback if vocabulary is empty or only stop words
        return np.eye(len(sentences), dtype=np.float32)
