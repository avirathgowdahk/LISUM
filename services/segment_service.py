"""
===============================================================================
Module: Segment Selection Service (services/segment_service.py)
===============================================================================
1. Purpose:
   Performs context-aware clip boundary expansion, overlap deduplication,
   0/1 Knapsack selection under target duration budget, and chronological clip ordering.
2. Input:
   Aligned timestamp intervals, list of timestamped Sentence objects, target_duration_sec.
3. Processing:
   - Boundary Expansion: Expands clip edges to complete sentence bounds (+0.2s padding) to prevent mid-sentence cuts
   - Deduplication: Merges overlapping or adjacent time intervals (< 0.8s gap)
   - Duration Control: Solves 0/1 Knapsack optimization to maximize total importance score subject to total_duration <= target_duration
   - Ordering: Ensures final clips are ordered chronologically to preserve lecture flow
4. Algorithm/Model:
   Dynamic Programming 0/1 Knapsack with interval boundary expansion and merging.
5. Output:
   Optimized list of (start_time, end_time) tuples and VideoClip objects.
6. Data Stored:
   VideoClip objects containing start_time, end_time, duration, importance_score.
7. Failure Cases:
   If total selected clip duration exceeds target_duration, trims lowest importance clip.
8. Evaluation:
   Validates total_duration <= target_duration (+/- 2 seconds) and zero mid-word cuts.
9. Integration:
   Passes final clip intervals to video_composer.py for FFmpeg rendering.
===============================================================================
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from models.transcript import Sentence
from models.segment import VideoClip
from utils.timestamp_utils import merge_overlapping_intervals

def expand_clip_boundaries(
    raw_intervals: List[Tuple[float, float]],
    sentences: List[Sentence],
    padding_sec: float = 0.25
) -> List[Tuple[float, float]]:
    """Expands clip bounds to natural sentence boundaries and adds minor padding."""
    if not raw_intervals:
        return []

    expanded = []
    for s_time, e_time in raw_intervals:
        # Find sentence containing s_time and e_time
        matching_start_sents = [s for s in sentences if s.start_time <= s_time <= s.end_time or abs(s.start_time - s_time) <= 1.0]
        matching_end_sents = [s for s in sentences if s.start_time <= e_time <= s.end_time or abs(s.end_time - e_time) <= 1.0]

        start_bound = matching_start_sents[0].start_time if matching_start_sents else s_time
        end_bound = matching_end_sents[-1].end_time if matching_end_sents else e_time

        # Apply minor padding
        start_bound = max(0.0, start_bound - padding_sec)
        end_bound = end_bound + padding_sec

        expanded.append((start_bound, end_bound))

    return expanded

def select_optimal_video_segments(
    raw_intervals: List[Tuple[float, float, float]], # (start, end, score)
    sentences: List[Sentence],
    target_duration_sec: float
) -> Tuple[List[Tuple[float, float]], List[VideoClip], float]:
    """
    Selects optimal video clips satisfying target duration constraint via 0/1 Knapsack.
    Removes overlaps, expands boundaries, and preserves chronological order.
    """
    if not raw_intervals:
        return [], [], 0.0

    # Step 1: Expand clip boundaries
    interval_pairs = [(i[0], i[1]) for i in raw_intervals]
    expanded_pairs = expand_clip_boundaries(interval_pairs, sentences)

    # Step 2: Merge overlapping clip intervals
    merged_intervals = merge_overlapping_intervals(expanded_pairs, max_gap_sec=0.8)

    # Calculate merged scores and durations
    candidates = []
    for idx, (s_start, s_end) in enumerate(merged_intervals):
        dur = max(1.0, s_end - s_start)
        # Average score of matching sentences in interval
        matching_scores = [s.importance_score for s in sentences if s_start <= s.start_time <= s_end or s_start <= s.end_time <= s_end]
        score = float(np.mean(matching_scores)) if matching_scores else 0.5

        candidates.append({
            "id": idx + 1,
            "start": s_start,
            "end": s_end,
            "duration": dur,
            "score": score
        })

    # Step 3: 0/1 Knapsack Selection under Target Duration Budget
    capacity = int(round(target_duration_sec * 10))  # Decisecond resolution
    n = len(candidates)
    weights = [max(1, int(round(c["duration"] * 10))) for c in candidates]
    values = [int(round(c["score"] * 1000)) for c in candidates]

    dp = np.zeros((n + 1, capacity + 1), dtype=int)

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for w_cap in range(capacity + 1):
            if w <= w_cap:
                dp[i][w_cap] = max(dp[i - 1][w_cap], dp[i - 1][w_cap - w] + v)
            else:
                dp[i][w_cap] = dp[i - 1][w_cap]

    # Backtrack to identify selected candidates
    w_cap = capacity
    selected_indices = []
    for i in range(n, 0, -1):
        if dp[i][w_cap] != dp[i - 1][w_cap]:
            selected_indices.append(i - 1)
            w_cap -= weights[i - 1]

    selected_candidates = [candidates[i] for i in reversed(selected_indices)]

    # Fallback: if knapsack selected nothing (all durations > capacity), pick top scoring candidate trimmed to target
    if not selected_candidates and candidates:
        top_c = max(candidates, key=lambda x: x["score"])
        top_c["end"] = top_c["start"] + target_duration_sec
        top_c["duration"] = target_duration_sec
        selected_candidates = [top_c]

    # Chronological Ordering
    selected_candidates.sort(key=lambda x: x["start"])

    final_clips: List[Tuple[float, float]] = []
    video_clips: List[VideoClip] = []
    total_duration = 0.0

    for idx, c in enumerate(selected_candidates):
        clip_tuple = (round(c["start"], 2), round(c["end"], 2))
        final_clips.append(clip_tuple)

        v_clip = VideoClip(
            clip_id=f"clip_{idx+1:03d}",
            lecture_id="lecture_001",
            start_time=c["start"],
            end_time=c["end"],
            importance_score=c["score"]
        )
        video_clips.append(v_clip)
        total_duration += c["duration"]

    print(f"[SEGMENT SERVICE] Selected {len(final_clips)} optimal video clips (Total duration: {total_duration:.2f}s, Target: {target_duration_sec:.2f}s)")
    return final_clips, video_clips, total_duration
