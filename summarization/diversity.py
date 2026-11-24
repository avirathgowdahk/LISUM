import numpy as np
from typing import List, Tuple

def compute_shot_feature_representations(
    frame_features: np.ndarray,
    timestamps: np.ndarray,
    shots: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Computes mean ResNet-50 feature vector for each shot (M, 2048).
    """
    shot_feats = []
    for (start_t, end_t) in shots:
        mask = (timestamps >= start_t) & (timestamps <= end_t)
        if np.any(mask):
            feat = np.mean(frame_features[mask], axis=0)
        else:
            feat = np.zeros(frame_features.shape[1], dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        shot_feats.append(feat)
    
    return np.array(shot_feats, dtype=np.float32)

def compute_cosine_similarity_matrix(shot_features: np.ndarray) -> np.ndarray:
    """Computes pairwise cosine similarity matrix between shot feature vectors."""
    if len(shot_features) == 0:
        return np.empty((0, 0))
    # Dot product of normalized vectors = cosine similarity
    sim_matrix = np.dot(shot_features, shot_features.T)
    return np.clip(sim_matrix, 0.0, 1.0)

def apply_diversity_penalty(
    shot_scores: List[float],
    shot_features: np.ndarray,
    diversity_lambda: float = 0.3
) -> List[float]:
    """
    Penalizes visually similar high-scoring shots to prevent repetitive summaries.
    Formula: Adjusted_Score(i) = Score(i) - lambda * max_{j != i, selected/higher} CosineSim(i, j)
    """
    scores = np.array(shot_scores, dtype=np.float32)
    n_shots = len(scores)
    if n_shots <= 1:
        return shot_scores

    sim_matrix = compute_cosine_similarity_matrix(shot_features)
    adjusted_scores = scores.copy()

    # Iterate shots in descending order of original importance score
    sorted_indices = np.argsort(-scores)
    selected_indices = []

    for idx in sorted_indices:
        if not selected_indices:
            selected_indices.append(idx)
            continue

        # Find max similarity with already processed higher-scoring shots
        max_sim = np.max(sim_matrix[idx, selected_indices])
        penalty = diversity_lambda * max_sim
        adjusted_scores[idx] = max(0.0, adjusted_scores[idx] - penalty)
        selected_indices.append(idx)

    return [round(float(s), 4) for s in adjusted_scores]
