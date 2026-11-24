import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from config import MIN_SHOT_DURATION, MAX_SHOT_DURATION, THRESHOLD_FACTOR, FALLBACK_WINDOW_SEC
from preprocessing.video_reader import get_video_metadata
from preprocessing.frame_sampler import sample_frames

def compute_frame_difference(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """Computes fast normalized 1D histogram difference between two consecutive frames."""
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([gray1], [0], None, [32], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [32], [0, 256])

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return float(score)

def generate_fixed_windows(duration_sec: float, window_sec: float = FALLBACK_WINDOW_SEC) -> List[Tuple[float, float]]:
    """Generates continuous fixed-length time segments (fallback mechanism)."""
    segments = []
    curr = 0.0
    while curr < duration_sec:
        end = min(curr + window_sec, duration_sec)
        if end - curr >= 0.5:  # filter negligible leftover segments
            segments.append((round(curr, 2), round(end, 2)))
        curr = end
    return segments

def detect_shots_and_keyframes(
    video_path: str,
    target_fps: float = 1.0,
    threshold_factor: float = THRESHOLD_FACTOR,
    min_shot_duration: float = MIN_SHOT_DURATION,
    max_shot_duration: float = MAX_SHOT_DURATION,
    fallback_window_sec: float = FALLBACK_WINDOW_SEC
) -> Tuple[List[Tuple[float, float]], List[np.ndarray], List[float], List[int]]:
    """
    Performs shot detection AND keyframe extraction in a SINGLE linear forward video pass.
    Eliminates redundant VideoCapture seeking/decoding.
    """
    metadata = get_video_metadata(video_path)
    duration_sec = metadata["duration_seconds"]

    if duration_sec <= 0:
        return [], [], [], []

    # Adaptive FPS downsampling for long videos (>3 mins) to avoid redundant frame decoding
    if duration_sec >= 180.0 and target_fps > 0.33:
        target_fps = 0.33

    from preprocessing.frame_sampler import sample_frames_fast_ffmpeg
    prev_thumb = None
    timestamps = []
    sampled_records = []  # List of (ts, frame)
    diffs = []

    fast_frames = sample_frames_fast_ffmpeg(video_path, target_fps=target_fps)
    for ts, frame in fast_frames:
        timestamps.append(ts)
        sampled_records.append((ts, frame))
        thumb = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
        if prev_thumb is not None:
            d = compute_frame_difference(prev_thumb, thumb)
            diffs.append(d)
        prev_thumb = thumb

    if len(timestamps) < 2:
        shots = [(0.0, round(duration_sec, 2))]
        kf_frames = [sampled_records[0][1]] if sampled_records else []
        kf_ts = [sampled_records[0][0]] if sampled_records else [0.0]
        return shots, kf_frames, kf_ts, [0] if kf_frames else []

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    threshold = mean_diff + (threshold_factor * std_diff)

    cut_timestamps = [0.0]
    for i, d in enumerate(diffs):
        if d > threshold:
            boundary_time = timestamps[i + 1]
            if boundary_time - cut_timestamps[-1] >= min_shot_duration:
                cut_timestamps.append(round(boundary_time, 2))

    cut_timestamps.append(round(duration_sec, 2))

    segments = []
    for i in range(len(cut_timestamps) - 1):
        seg_start = cut_timestamps[i]
        seg_end = cut_timestamps[i+1]
        seg_dur = seg_end - seg_start

        if seg_dur > max_shot_duration:
            sub_segs = generate_fixed_windows(seg_dur, window_sec=fallback_window_sec)
            for sub_s, sub_e in sub_segs:
                segments.append((round(seg_start + sub_s, 2), round(seg_start + sub_e, 2)))
        elif seg_dur >= min_shot_duration or i == 0:
            segments.append((seg_start, seg_end))
        else:
            if segments:
                prev_s, _ = segments[-1]
                segments[-1] = (prev_s, seg_end)
            else:
                segments.append((seg_start, seg_end))

    if not segments:
        segments = generate_fixed_windows(duration_sec, window_sec=fallback_window_sec)

    # Extract keyframe per shot directly from in-memory sampled_records
    kf_frames = []
    kf_timestamps = []
    kf_shot_indices = []

    ts_arr = np.array([r[0] for r in sampled_records], dtype=np.float32)

    for shot_idx, (s_start, s_end) in enumerate(segments):
        mid_target = (s_start + s_end) / 2.0
        best_idx = int(np.argmin(np.abs(ts_arr - mid_target)))
        kf_ts, kf_img = sampled_records[best_idx]

        kf_frames.append(kf_img)
        kf_timestamps.append(kf_ts)
        kf_shot_indices.append(shot_idx)

    return segments, kf_frames, kf_timestamps, kf_shot_indices

def detect_shots(
    video_path: str,
    target_fps: float = 1.0,
    threshold_factor: float = THRESHOLD_FACTOR,
    min_shot_duration: float = MIN_SHOT_DURATION,
    max_shot_duration: float = MAX_SHOT_DURATION,
    fallback_window_sec: float = FALLBACK_WINDOW_SEC
) -> List[Tuple[float, float]]:
    shots, _, _, _ = detect_shots_and_keyframes(
        video_path, target_fps=target_fps, threshold_factor=threshold_factor,
        min_shot_duration=min_shot_duration, max_shot_duration=max_shot_duration,
        fallback_window_sec=fallback_window_sec
    )
    return shots

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        shots = detect_shots(path)
        print(f"Detected {len(shots)} shots:")
        for idx, (s, e) in enumerate(shots):
            print(f"  Shot {idx+1:02d}: {s:.2f}s -> {e:.2f}s (duration: {e-s:.2f}s)")
    else:
        print("Usage: python preprocessing/shot_detector.py <video_path>")
