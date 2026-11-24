import numpy as np
from typing import List, Tuple, Dict, Any

def select_segments_knapsack(
    shots: List[Tuple[float, float]],
    scores: List[float],
    target_duration_sec: float
) -> Tuple[List[Tuple[float, float]], List[int], float]:
    """
    Selects optimal subset of continuous shots maximizing importance score while
    staying within target_duration_sec budget using dynamic programming (0/1 Knapsack).

    Returns:
    - selected_shots: List of (start_sec, end_sec)
    - selected_indices: List of shot indices
    - total_selected_duration: Total duration of selected summary
    """
    if not shots or target_duration_sec <= 0:
        return [], [], 0.0

    n = len(shots)
    durations = [max(0.1, end - start) for (start, end) in shots]
    
    # Scale durations and budget to discrete integer units (tenths of a second)
    scale = 10
    weights = [int(round(d * scale)) for d in durations]
    capacity = int(round(target_duration_sec * scale))
    
    # Scale scores to integer values for DP table
    int_scores = [int(round(s * 1000)) for s in scores]

    # DP Table: dp[i][w] = max value using subset of first i shots with weight capacity w
    dp = np.zeros((n + 1, capacity + 1), dtype=np.int32)

    for i in range(1, n + 1):
        w = weights[i - 1]
        val = int_scores[i - 1]
        for c in range(capacity + 1):
            if w <= c:
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + val)
            else:
                dp[i][c] = dp[i - 1][c]

    # Backtrack to find selected shot indices
    selected_indices = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected_indices.append(i - 1)
            c -= weights[i - 1]

    selected_indices.sort()
    selected_shots = [shots[idx] for idx in selected_indices]
    total_duration = sum(end - start for (start, end) in selected_shots)

    # Fallback / Budget Top-up: If total duration is below (target_duration_sec - 5.0s), append/trim next best shots
    min_acceptable = max(5.0, target_duration_sec - 5.0)
    max_acceptable = target_duration_sec + 2.0

    if total_duration < min_acceptable and shots:
        unselected = [i for i in range(n) if i not in selected_indices]
        unselected.sort(key=lambda idx: scores[idx], reverse=True)

        for idx in unselected:
            if total_duration >= target_duration_sec - 1.0:
                break
            s, e = shots[idx]
            shot_d = e - s
            remaining_budget = target_duration_sec - total_duration
            if shot_d <= remaining_budget + 2.0:
                selected_shots.append((s, e))
                selected_indices.append(idx)
                total_duration += shot_d
            else:
                # Trim clip to remaining budget
                trimmed_end = round(s + remaining_budget, 2)
                if trimmed_end - s >= 1.0:
                    selected_shots.append((s, trimmed_end))
                    selected_indices.append(idx)
                    total_duration += (trimmed_end - s)
                break

    # Capping check: If total duration exceeds (target_duration_sec + 2.0s), trim the final selected shot
    if total_duration > max_acceptable and selected_shots:
        selected_shots.sort(key=lambda x: x[0])
        accum = 0.0
        adjusted_shots = []
        for s, e in selected_shots:
            d = e - s
            if accum + d <= target_duration_sec:
                adjusted_shots.append((s, e))
                accum += d
            else:
                rem = target_duration_sec - accum
                if rem >= 1.0:
                    adjusted_shots.append((s, round(s + rem, 2)))
                    accum += rem
                break
        selected_shots = adjusted_shots
        total_duration = sum(e - s for (s, e) in selected_shots)

    # Re-sort chronologically
    selected_shots.sort(key=lambda x: x[0])

    # Merge contiguous/overlapping selected shots
    merged_shots = []
    for s, e in selected_shots:
        if merged_shots and s <= merged_shots[-1][1] + 0.15:
            merged_shots[-1] = (merged_shots[-1][0], round(max(merged_shots[-1][1], e), 2))
        else:
            merged_shots.append((round(s, 2), round(e, 2)))
    selected_shots = merged_shots

    # Final Fallback if empty
    if not selected_shots and shots:
        best_idx = int(np.argmax(scores))
        s, e = shots[best_idx]
        trimmed_e = round(s + min(target_duration_sec, e - s), 2)
        selected_shots = [(s, trimmed_e)]
        selected_indices = [best_idx]
        total_duration = trimmed_e - s

    total_duration = round(total_duration, 2)

    print(f"\n[DURATION DEBUG]\nRequested target: {target_duration_sec:.2f} sec\nSelected clip duration: {total_duration:.2f} sec", flush=True)

    return selected_shots, selected_indices, total_duration

if __name__ == "__main__":
    dummy_shots = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0)]
    dummy_scores = [0.8, 0.4, 0.95, 0.3, 0.85]
    budget = 10.0  # target summary duration 10 seconds

    selected_shots, indices, total_dur = select_segments_knapsack(dummy_shots, dummy_scores, budget)
    print(f"Selected shots: {selected_shots}")
    print(f"Indices: {indices}")
    print(f"Total duration: {total_dur}s (Budget: {budget}s)")
