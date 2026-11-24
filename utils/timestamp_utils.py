"""
===============================================================================
Module: Timestamp Utilities (utils/timestamp_utils.py)
===============================================================================
1. Purpose:
   Provides robust timestamp conversion, formatting, interval overlap merging,
   and boundary expansion.
2. Input:
   Floating-point seconds, HH:MM:SS strings, timestamp intervals.
3. Processing:
   - Converts seconds to formatted string (e.g. 01:12:30.40)
   - Merges contiguous or overlapping time intervals
   - Expands interval bounds for sentence/pause completeness
4. Output:
   Formatted time strings, merged list of (start, end) tuples.
===============================================================================
"""

from typing import List, Tuple

def format_timestamp(seconds: float, include_millis: bool = False) -> str:
    """Formats seconds into HH:MM:SS or HH:MM:SS.ms string."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if hours > 0:
        base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        base = f"{minutes:02d}:{secs:02d}"

    if include_millis:
        return f"{base}.{millis:03d}"
    return base

def parse_timestamp(ts_str: str) -> float:
    """Parses HH:MM:SS or MM:SS timestamp string to float seconds."""
    parts = ts_str.strip().split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0

def merge_overlapping_intervals(intervals: List[Tuple[float, float]], max_gap_sec: float = 0.5) -> List[Tuple[float, float]]:
    """
    Merges overlapping or close timestamp intervals.
    Example: [(10.0, 15.0), (14.5, 20.0), (22.0, 30.0)] -> [(10.0, 20.0), (22.0, 30.0)]
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end + max_gap_sec:
            # Overlap or within max_gap: extend previous interval
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append((curr_start, curr_end))

    return merged
