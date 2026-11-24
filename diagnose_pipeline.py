import os
import sys
import time
import psutil
import torch
import cv2
import numpy as np
from typing import Dict, Any, List

# Add workspace directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import DIVERSITY_LAMBDA, FEATURE_DIR
from preprocessing.video_reader import get_video_metadata
from preprocessing.frame_sampler import sample_frames_generator, sample_frames, sample_representative_shot_keyframes
from preprocessing.shot_detector import detect_shots
from models.resnet_feature_extractor import ResNet50FeatureExtractor
from models.temporal_transformer import load_trained_model, compute_shot_importance_scores
from summarization.diversity import compute_shot_feature_representations, apply_diversity_penalty
from summarization.segment_selector import select_segments_knapsack
from video_processing.video_merger import generate_summary_video

def check_system_hardware():
    print("==================================================")
    print("           SYSTEM & HARDWARE DIAGNOSTICS          ")
    print("==================================================")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"GPU Model Name   : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU Status       : GPU unavailable — running on CPU.")
    
    cpu_cores_physical = psutil.cpu_count(logical=False)
    cpu_cores_logical = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    print(f"CPU Physical Cores: {cpu_cores_physical}")
    print(f"CPU Logical Cores : {cpu_cores_logical}")
    print(f"Total RAM System  : {ram_gb:.2f} GB (Available: {ram_avail_gb:.2f} GB)")
    print(f"PyTorch CPU Threads: {torch.get_num_threads()}")
    print("==================================================\n", flush=True)

def run_detailed_diagnostic(video_path: str, label: str = "TEST VIDEO"):
    print(f"\n==================================================", flush=True)
    print(f"   RUNNING DIAGNOSTIC BENCHMARK: {label}", flush=True)
    print(f"   Video File: {os.path.basename(video_path)}", flush=True)
    print(f"==================================================", flush=True)
    
    process = psutil.Process(os.getpid())
    ram_start = process.memory_info().rss / (1024 ** 2)

    # A. Video Opening & B. Metadata Extraction
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video {video_path}", flush=True)
        return None
    t_video_open = time.time() - t0
    cap.release()

    t0 = time.time()
    meta = get_video_metadata(video_path)
    t_meta = time.time() - t0

    dur_sec = meta["duration_seconds"]
    fps = meta["fps"]
    total_frames_native = meta["total_frames"]

    print(f"Video Duration : {meta['duration_formatted']} ({dur_sec:.2f}s)", flush=True)
    print(f"Native FPS     : {fps}", flush=True)
    print(f"Total Native Frames: {total_frames_native}", flush=True)

    # C. Frame Sampling Count & Timing
    t0 = time.time()
    sampled_frames, sampled_timestamps, sampled_indices = sample_frames(video_path, target_fps=2.0)
    t_sampling = time.time() - t0
    num_sampled = len(sampled_frames)

    # D. Scene / Shot Detection Timing & Count
    t0 = time.time()
    shots_list = detect_shots(video_path, target_fps=1.0)
    t_scene_detection = time.time() - t0
    num_shots = len(shots_list)

    # E. Representative Frame Extraction Timing & Count
    t0 = time.time()
    rep_frames, rep_timestamps, rep_shot_indices = sample_representative_shot_keyframes(
        video_path, shots_list, keyframes_per_shot=1
    )
    t_rep_extraction = time.time() - t0
    num_rep_frames = len(rep_frames)

    # F. ResNet Feature Extraction Timing & Batched Execution
    t0 = time.time()
    extractor = ResNet50FeatureExtractor()
    features, timestamps, shots = extractor.extract_and_cache(video_path, target_fps=2.0, force_recompute=True)
    t_resnet = time.time() - t0

    # G. Temporal Transformer Inference
    t0 = time.time()
    model, is_trained = load_trained_model(device=extractor.device)
    with torch.no_grad():
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(extractor.device)
        frame_scores_tensor = model(feat_tensor)
        frame_scores = frame_scores_tensor.squeeze(0).cpu().numpy()
    shot_scores = compute_shot_importance_scores(frame_scores, timestamps, shots_list)
    t_transformer = time.time() - t0

    # H. Diversity Calculation
    t0 = time.time()
    shot_features = compute_shot_feature_representations(features, timestamps, shots_list)
    adjusted_scores = apply_diversity_penalty(shot_scores, shot_features, diversity_lambda=DIVERSITY_LAMBDA)
    t_diversity = time.time() - t0

    # I. Knapsack Selection
    t0 = time.time()
    target_dur = round(dur_sec * 0.15, 2)
    selected_shots, selected_indices, summary_dur = select_segments_knapsack(shots_list, adjusted_scores, target_dur)
    t_knapsack = time.time() - t0

    # J. FFmpeg Summary Video Cut
    t0 = time.time()
    scratch_dir = os.path.join(BASE_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    out_summary_path = os.path.join(scratch_dir, f"diag_summary_{os.path.basename(video_path)}")
    selected_shots_sorted = sorted(selected_shots, key=lambda x: x[0])
    generate_summary_video(video_path, selected_shots_sorted, out_summary_path)
    t_ffmpeg = time.time() - t0

    ram_end = process.memory_info().rss / (1024 ** 2)
    total_wall_time = (
        t_video_open + t_meta + t_sampling + t_scene_detection +
        t_rep_extraction + t_resnet + t_transformer + t_diversity +
        t_knapsack + t_ffmpeg
    )

    # Pass 1: sample_frames
    # Pass 2: detect_shots
    # Pass 3: sample_representative_shot_keyframes
    # Pass 4: extract_and_cache
    # Pass 5: generate_summary_video
    total_frames_decoded_approx = (total_frames_native * 2) + num_rep_frames

    print("\n==================================================", flush=True)
    print("PERFORMANCE PROFILE", flush=True)
    print("==================================================", flush=True)
    print(f"Video duration        : {dur_sec:.2f} sec ({meta['duration_formatted']})", flush=True)
    print(f"Sampled frames        : {num_sampled}", flush=True)
    print(f"Detected shots        : {num_shots}", flush=True)
    print(f"Representative frames : {num_rep_frames}", flush=True)
    print(f"Frames sent to ResNet : {num_rep_frames}", flush=True)
    print(f"Approx Frames Decoded : ~{total_frames_decoded_approx}", flush=True)
    print("--------------------------------------------------", flush=True)
    print(f"Metadata: {t_meta:.4f} sec", flush=True)
    print(f"Sampling: {t_sampling:.4f} sec", flush=True)
    print(f"Scene detection: {t_scene_detection:.4f} sec", flush=True)
    print(f"Representative frames: {t_rep_extraction:.4f} sec", flush=True)
    print(f"ResNet: {t_resnet:.4f} sec", flush=True)
    print(f"Transformer: {t_transformer:.4f} sec", flush=True)
    print(f"Diversity: {t_diversity:.4f} sec", flush=True)
    print(f"Knapsack: {t_knapsack:.4f} sec", flush=True)
    print(f"FFmpeg: {t_ffmpeg:.4f} sec", flush=True)
    print("--------------------------------------------------", flush=True)
    print(f"TOTAL: {total_wall_time:.4f} sec", flush=True)
    print(f"PROCESSING RATIO       : {total_wall_time / dur_sec:.4f}", flush=True)
    print(f"RAM Memory Used        : {ram_end:.2f} MB (Start: {ram_start:.2f} MB)", flush=True)
    print("==================================================\n", flush=True)

    return {
        "video": os.path.basename(video_path),
        "duration": dur_sec,
        "total_frames_native": total_frames_native,
        "num_sampled": num_sampled,
        "num_shots": num_shots,
        "num_rep_frames": num_rep_frames,
        "approx_decoded": total_frames_decoded_approx,
        "t_meta": t_meta,
        "t_sampling": t_sampling,
        "t_scene_detection": t_scene_detection,
        "t_rep_extraction": t_rep_extraction,
        "t_resnet": t_resnet,
        "t_transformer": t_transformer,
        "t_diversity": t_diversity,
        "t_knapsack": t_knapsack,
        "t_ffmpeg": t_ffmpeg,
        "total_time": total_wall_time,
        "ram_mb": ram_end
    }

if __name__ == "__main__":
    check_system_hardware()
    
    # Locate test videos for duration scaling test: 30-60 sec, 2 min, 11.25 min
    video_short = os.path.join(BASE_DIR, "uploads", "sample_test.mp4")
    video_mid = os.path.join(BASE_DIR, "uploads", "7507b4c1_videoplayback.mp4")
    video_long = os.path.join(BASE_DIR, "uploads", "00738193_vidssave.com_Composite_Primary_Key_in_SQL___SQL_Interview_Question___PK_vs_FK_Explained_1080P.mp4")

    results = []
    
    if os.path.exists(video_short):
        res1 = run_detailed_diagnostic(video_short, "SHORT VIDEO (~30s)")
        if res1: results.append(res1)
        
    if os.path.exists(video_mid):
        res2 = run_detailed_diagnostic(video_mid, "MEDIUM VIDEO (~2.4 min)")
        if res2: results.append(res2)
        
    if os.path.exists(video_long):
        res3 = run_detailed_diagnostic(video_long, "LONG VIDEO (~11.25 min)")
        if res3: results.append(res3)

    print("\n==================================================", flush=True)
    print("            VIDEO SCALING COMPARISON              ", flush=True)
    print("==================================================", flush=True)
    print(f"{'Video File':<30} | {'Duration':<10} | {'Total Time':<10} | {'Ratio (T/D)':<12}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        print(f"{r['video']:<30} | {r['duration']:8.2f}s | {r['total_time']:8.2f}s | {r['total_time']/r['duration']:10.4f}", flush=True)
    print("==================================================", flush=True)
