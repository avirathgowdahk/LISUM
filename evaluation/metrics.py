"""
===============================================================================
Module: Evaluation Metrics (evaluation/metrics.py)
===============================================================================
1. Purpose:
   Provides academic quantitative evaluation metrics for transcription, summarization,
   semantic alignment, and video duration compliance.
2. Evaluated Areas (Section 42 of specifications):
   - Transcription: Word Error Rate (WER), Character Error Rate (CER)
   - Summarization: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, Compression Ratio
   - Temporal Alignment: Timestamp Retrieval Error, Coverage Accuracy
   - Video Generation: Duration Error, Audio-Video Sync Offset
===============================================================================
"""

import re
from typing import Dict, Any, List, Tuple
from rouge_score import rouge_scorer

def calculate_wer_cer(reference_text: str, hypothesis_text: str) -> Dict[str, float]:
    """Calculates Word Error Rate (WER) and Character Error Rate (CER)."""
    ref_words = re.findall(r'\w+', reference_text.lower())
    hyp_words = re.findall(r'\w+', hypothesis_text.lower())

    # Levenshtein distance for words
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return {"wer": 0.0 if m == 0 else 1.0, "cer": 0.0}

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    wer = float(dp[n][m] / max(1, n))

    # Character distance
    ref_chars = list(reference_text.lower())
    hyp_chars = list(hypothesis_text.lower())
    nc, mc = len(ref_chars), len(hyp_chars)
    dpc = [[0] * (mc + 1) for _ in range(nc + 1)]
    for i in range(nc + 1):
        dpc[i][0] = i
    for j in range(mc + 1):
        dpc[0][j] = j
    for i in range(1, nc + 1):
        for j in range(1, mc + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dpc[i][j] = min(dpc[i - 1][j] + 1, dpc[i][j - 1] + 1, dpc[i - 1][j - 1] + cost)

    cer = float(dpc[nc][mc] / max(1, nc))

    return {"wer": round(wer, 4), "cer": round(cer, 4)}

def calculate_rouge_scores(reference_summary: str, generated_summary: str) -> Dict[str, float]:
    """Calculates ROUGE-1, ROUGE-2, ROUGE-L metrics."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)

    return {
        "rouge1_f1": round(scores['rouge1'].fmeasure, 4),
        "rouge2_f1": round(scores['rouge2'].fmeasure, 4),
        "rougeL_f1": round(scores['rougeL'].fmeasure, 4)
    }

def evaluate_duration_compliance(target_duration_sec: float, actual_duration_sec: float) -> Dict[str, Any]:
    """Evaluates how closely the generated video respects target duration."""
    abs_diff = abs(actual_duration_sec - target_duration_sec)
    pct_error = round((abs_diff / max(1.0, target_duration_sec)) * 100, 2)
    passed = abs_diff <= 3.0 or pct_error <= 10.0

    return {
        "target_duration_sec": target_duration_sec,
        "actual_duration_sec": actual_duration_sec,
        "absolute_difference_sec": round(abs_diff, 2),
        "percentage_error": pct_error,
        "compliant": passed
    }
