import numpy as np
from scipy.stats import kendalltau, spearmanr
from typing import Dict, Any, List, Tuple

def compute_overlap_precision_recall_f1(
    pred_summary_mask: np.ndarray,
    user_summary_mask: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes frame-level Precision, Recall, and F1 score between predicted
    summary binary mask and ground-truth binary mask.
    """
    intersection = np.logical_and(pred_summary_mask, user_summary_mask).sum()
    pred_total = pred_summary_mask.sum()
    user_total = user_summary_mask.sum()

    precision = intersection / pred_total if pred_total > 0 else 0.0
    recall = intersection / user_total if user_total > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(float(precision), 4), round(float(recall), 4), round(float(f1), 4)

def evaluate_summarization_metrics(
    pred_scores: np.ndarray,
    gt_scores: np.ndarray,
    pred_summary_mask: np.ndarray,
    gt_summary_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculates evaluation metrics:
    - F1 Score (Primary Metric)
    - Precision
    - Recall
    - Kendall's Tau (Rank Correlation)
    - Spearman's Rho (Rank Correlation)
    """
    precision, recall, f1 = compute_overlap_precision_recall_f1(pred_summary_mask, gt_summary_mask)

    # Calculate Rank Correlation Metrics
    tau, _ = kendalltau(pred_scores, gt_scores)
    rho, _ = spearmanr(pred_scores, gt_scores)

    tau = 0.0 if np.isnan(tau) else float(tau)
    rho = 0.0 if np.isnan(rho) else float(rho)

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "kendall_tau": round(tau, 4),
        "spearman_rho": round(rho, 4)
    }

if __name__ == "__main__":
    # Test sample evaluation calculation
    dummy_pred_scores = np.array([0.9, 0.8, 0.2, 0.1, 0.75])
    dummy_gt_scores = np.array([0.85, 0.7, 0.3, 0.1, 0.8])

    dummy_pred_mask = np.array([1, 1, 0, 0, 1])
    dummy_gt_mask = np.array([1, 1, 0, 0, 0])

    metrics = evaluate_summarization_metrics(
        dummy_pred_scores, dummy_gt_scores, dummy_pred_mask, dummy_gt_mask
    )
    print("Sample Evaluation Metrics Results:")
    for k, v in metrics.items():
        print(f"  - {k:15s}: {v}")
