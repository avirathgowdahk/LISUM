import os
import sys
import time
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.video_reader import get_video_metadata
from preprocessing.frame_sampler import sample_frames
from preprocessing.shot_detector import detect_shots
from models.resnet_feature_extractor import ResNet50FeatureExtractor
from models.temporal_transformer import TemporalTransformerEncoder, load_trained_model, compute_shot_importance_scores
from summarization.diversity import compute_shot_feature_representations, apply_diversity_penalty
from summarization.segment_selector import select_segments_knapsack
from video_processing.video_merger import generate_summary_video

def profile_video(video_path: str):
    print("==================================================", flush=True)
    print("   PHASE 1 PROFILING — 11.25 MINUTE VIDEO TEST   ", flush=True)
    print("==================================================", flush=True)
    print(f"Target Video: {video_path}", flush=True)
    print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}", flush=True)

    total_start = time.time()

    # Stage 1: Metadata Extraction
    t0 = time.time()
    metadata = get_video_metadata(video_path)
    t_metadata = time.time() - t0
    print(f"\n[1] Metadata Extraction: {t_metadata:.4f}s", flush=True)
    print(f"    Duration: {metadata['duration_formatted']} ({metadata['duration_seconds']}s)", flush=True)
    print(f"    FPS: {metadata['fps']}, Total Frames: {metadata['total_frames']}", flush=True)

    # Stage 2-4: Scalable Shot Detection & Batched Representative ResNet Feature Extraction
    t0 = time.time()
    extractor = ResNet50FeatureExtractor()
    features, timestamps, shots = extractor.extract_and_cache(video_path, target_fps=2.0, force_recompute=True)
    t_resnet_total = time.time() - t0
    shots_list = shots.tolist()
    print(f"\n[2-4] Shot Detection & Scalable ResNet-50 Extraction: {t_resnet_total:.4f}s", flush=True)
    print(f"    Feature Shape: {features.shape}", flush=True)
    print(f"    Shots Detected: {len(shots_list)}", flush=True)

    # Stage 5: Temporal Transformer Inference
    t0 = time.time()
    model, is_trained = load_trained_model(device=extractor.device)
    with torch.no_grad():
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(extractor.device)
        frame_scores_tensor = model(feat_tensor)
        frame_scores = frame_scores_tensor.squeeze(0).cpu().numpy()
    shot_scores = compute_shot_importance_scores(frame_scores, timestamps, shots_list)
    t_transformer = time.time() - t0
    print(f"\n[5] Transformer Model Inference: {t_transformer:.4f}s")

    # Stage 6: Diversity & Redundancy Penalty
    t0 = time.time()
    shot_features = compute_shot_feature_representations(features, timestamps, shots_list)
    adjusted_scores = apply_diversity_penalty(shot_scores, shot_features, diversity_lambda=0.3)
    t_diversity = time.time() - t0
    print(f"\n[6] Diversity Calculation: {t_diversity:.4f}s")

    # Stage 7: Knapsack Segment Selection
    t0 = time.time()
    target_duration_sec = round(metadata["duration_seconds"] * 0.15, 2)
    selected_shots, _, summary_duration = select_segments_knapsack(shots_list, adjusted_scores, target_duration_sec)
    t_knapsack = time.time() - t0
    print(f"\n[7] Knapsack Selection: {t_knapsack:.4f}s")
    print(f"    Selected Shots: {len(selected_shots)}, Summary Duration: {summary_duration}s")

    # Stage 8: FFmpeg Output Generation
    t0 = time.time()
    out_path = os.path.join("outputs", "profile_test_summary_15pct.mp4")
    selected_shots.sort(key=lambda x: x[0])
    generate_summary_video(video_path, selected_shots, out_path)
    t_ffmpeg = time.time() - t0
    print(f"\n[8] FFmpeg Video Rendering: {t_ffmpeg:.4f}s")

    total_time = time.time() - total_start

    print("\n==================================================")
    print("   PROFILING SUMMARY TABLE                        ")
    print("==================================================", flush=True)
    print(f"Stage 1: Metadata Extraction   : {t_metadata:8.4f}s ({t_metadata/total_time*100:5.1f}%)", flush=True)
    print(f"Stage 2-4: Scalable ResNet-50  : {t_resnet_total:8.4f}s ({t_resnet_total/total_time*100:5.1f}%)", flush=True)
    print(f"Stage 5: Transformer Inference : {t_transformer:8.4f}s ({t_transformer/total_time*100:5.1f}%)", flush=True)
    print(f"Stage 6: Diversity Calculation : {t_diversity:8.4f}s ({t_diversity/total_time*100:5.1f}%)", flush=True)
    print(f"Stage 7: Knapsack Selection    : {t_knapsack:8.4f}s ({t_knapsack/total_time*100:5.1f}%)", flush=True)
    print(f"Stage 8: FFmpeg Video Cut      : {t_ffmpeg:8.4f}s ({t_ffmpeg/total_time*100:5.1f}%)", flush=True)
    print("--------------------------------------------------", flush=True)
    print(f"TOTAL EXECUTION TIME           : {total_time:8.4f}s", flush=True)
    print(f"PROCESSING RATIO (Time/Dur)    : {total_time / metadata['duration_seconds']:.4f}", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    vid = os.path.join("uploads", "00738193_vidssave.com_Composite_Primary_Key_in_SQL___SQL_Interview_Question___PK_vs_FK_Explained_1080P.mp4")
    profile_video(vid)
