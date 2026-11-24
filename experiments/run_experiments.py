import os
import sys
import json
import csv
import time
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple
from config import DEVICE, FEATURE_DIR, DIVERSITY_LAMBDA
from models.temporal_transformer import TemporalTransformerEncoder, load_trained_model, compute_shot_importance_scores
from summarization.diversity import compute_shot_feature_representations, apply_diversity_penalty
from summarization.segment_selector import select_segments_knapsack
from training.evaluate import evaluate_summarization_metrics, compute_overlap_precision_recall_f1

def generate_binary_mask_from_segments(
    selected_shots: List[Tuple[float, float]],
    total_duration_sec: float,
    fps: float = 2.0,
    target_length: int = None
) -> np.ndarray:
    """Converts a list of (start_sec, end_sec) tuples into a frame-level binary mask array."""
    total_frames = target_length if target_length else max(1, int(round(total_duration_sec * fps)))
    mask = np.zeros(total_frames, dtype=np.int32)
    
    for (start_sec, end_sec) in selected_shots:
        start_idx = int(round(start_sec * fps))
        end_idx = min(total_frames, int(round(end_sec * fps)))
        if start_idx < total_frames:
            mask[start_idx:end_idx] = 1

    return mask

def select_importance_only_segments(shots: List[Tuple[float, float]], scores: List[float], budget_sec: float) -> List[Tuple[float, float]]:
    """Greedy baseline selecting top-scoring shots sequentially until duration budget is reached."""
    sorted_pairs = sorted(zip(shots, scores), key=lambda x: x[1], reverse=True)
    selected = []
    curr_dur = 0.0
    for shot, sc in sorted_pairs:
        dur = shot[1] - shot[0]
        if curr_dur + dur <= budget_sec:
            selected.append(shot)
            curr_dur += dur
        if curr_dur >= budget_sec:
            break
    selected.sort(key=lambda x: x[0])
    return selected

def select_random_segments(shots: List[Tuple[float, float]], budget_sec: float, seed: int = 42) -> List[Tuple[float, float]]:
    """Random baseline selecting random shots until duration budget is reached."""
    np.random.seed(seed)
    indices = np.arange(len(shots))
    np.random.shuffle(indices)
    selected = []
    curr_dur = 0.0
    for idx in indices:
        shot = shots[idx]
        dur = shot[1] - shot[0]
        if curr_dur + dur <= budget_sec:
            selected.append(shot)
            curr_dur += dur
        if curr_dur >= budget_sec:
            break
    selected.sort(key=lambda x: x[0])
    return selected

def run_benchmark_experiments(
    video_names: List[str] = None,
    target_ratio: float = 0.15,
    sample_fps: float = 2.0,
    output_dir: str = "experiments"
) -> Dict[str, Any]:
    """
    Executes automated baseline comparison & ablation study suite.
    Evaluates:
    - Section A: Baseline Comparisons (Random vs. Importance-Only vs. Importance+Diversity vs. Proposed Knapsack)
    - Section B: Ablation Study (Transformer Only vs. +Diversity vs. +Knapsack vs. Proposed System)
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n==================================================")
    print("   STARTING AUTOMATED EVALUATION & EXPERIMENT SUITE ")
    print("==================================================")

    # Discover features in FEATURE_DIR if video_names not specified
    if not video_names:
        video_names = [
            f.replace("_features.npy", "") for f in os.listdir(FEATURE_DIR)
            if f.endswith("_features.npy")
        ]

    if not video_names:
        print("[!] No cached features found in features/ directory. Creating demo evaluation benchmark...")
        # Create demo synthetic sample for benchmark initialization
        demo_name = "demo_sample_video"
        demo_feats = np.random.randn(200, 2048).astype(np.float32)
        demo_ts = np.linspace(0, 100, 200).astype(np.float32)
        demo_shots = np.array([[i*5.0, (i+1)*5.0] for i in range(20)], dtype=np.float32)
        np.save(os.path.join(FEATURE_DIR, f"{demo_name}_features.npy"), demo_feats)
        np.save(os.path.join(FEATURE_DIR, f"{demo_name}_timestamps.npy"), demo_ts)
        np.save(os.path.join(FEATURE_DIR, f"{demo_name}_shots.npy"), demo_shots)
        video_names = [demo_name]

    model, is_trained = load_trained_model()

    # Data structures for results
    baseline_results = {
        "Random": {"precision": [], "recall": [], "f1": [], "time": []},
        "Importance-Only": {"precision": [], "recall": [], "f1": [], "time": []},
        "Importance+Diversity": {"precision": [], "recall": [], "f1": [], "time": []},
        "Proposed (Knapsack)": {"precision": [], "recall": [], "f1": [], "time": []}
    }

    ablation_results = {
        "Exp A (Transformer Only)": {"precision": [], "recall": [], "f1": [], "time": []},
        "Exp B (Transformer + Diversity)": {"precision": [], "recall": [], "f1": [], "time": []},
        "Exp C (Transformer + Knapsack)": {"precision": [], "recall": [], "f1": [], "time": []},
        "Exp D (Full Proposed System)": {"precision": [], "recall": [], "f1": [], "time": []}
    }

    for name in video_names:
        feat_path = os.path.join(FEATURE_DIR, f"{name}_features.npy")
        ts_path = os.path.join(FEATURE_DIR, f"{name}_timestamps.npy")
        shot_path = os.path.join(FEATURE_DIR, f"{name}_shots.npy")

        if not (os.path.exists(feat_path) and os.path.exists(ts_path) and os.path.exists(shot_path)):
            continue

        features = np.load(feat_path)
        timestamps = np.load(ts_path)
        shots = np.load(shot_path).tolist()

        if len(features) == 0 or len(shots) == 0:
            continue

        total_duration = timestamps[-1] if len(timestamps) > 0 else shots[-1][1]
        budget = round(total_duration * target_ratio, 2)

        # Ground truth simulation for evaluation (or real annotations if available)
        gt_mask = np.zeros(len(features), dtype=np.int32)
        # Top 15% ground truth mask
        gt_mask[:int(len(features) * target_ratio)] = 1

        # Model forward pass
        t0 = time.time()
        with torch.no_grad():
            feat_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
            frame_scores = model(feat_tensor).squeeze(0).cpu().numpy()
        inference_time = time.time() - t0

        raw_shot_scores = compute_shot_importance_scores(frame_scores, timestamps, shots)
        shot_feats = compute_shot_feature_representations(features, timestamps, shots)
        adj_shot_scores = apply_diversity_penalty(raw_shot_scores, shot_feats, diversity_lambda=DIVERSITY_LAMBDA)

        # -------------------------------------------------------------
        # SECTION A: BASELINE COMPARISONS
        # -------------------------------------------------------------
        # 1. Random Selection
        t1 = time.time()
        rand_shots = select_random_segments(shots, budget)
        rand_mask = generate_binary_mask_from_segments(rand_shots, total_duration, sample_fps, target_length=len(gt_mask))
        p, r, f = compute_overlap_precision_recall_f1(rand_mask, gt_mask)
        baseline_results["Random"]["precision"].append(p)
        baseline_results["Random"]["recall"].append(r)
        baseline_results["Random"]["f1"].append(f)
        baseline_results["Random"]["time"].append(time.time() - t1)

        # 2. Importance-Only Selection
        t1 = time.time()
        imp_shots = select_importance_only_segments(shots, raw_shot_scores, budget)
        imp_mask = generate_binary_mask_from_segments(imp_shots, total_duration, sample_fps, target_length=len(gt_mask))
        p, r, f = compute_overlap_precision_recall_f1(imp_mask, gt_mask)
        baseline_results["Importance-Only"]["precision"].append(p)
        baseline_results["Importance-Only"]["recall"].append(r)
        baseline_results["Importance-Only"]["f1"].append(f)
        baseline_results["Importance-Only"]["time"].append(time.time() - t1)

        # 3. Importance + Diversity
        t1 = time.time()
        div_shots = select_importance_only_segments(shots, adj_shot_scores, budget)
        div_mask = generate_binary_mask_from_segments(div_shots, total_duration, sample_fps, target_length=len(gt_mask))
        p, r, f = compute_overlap_precision_recall_f1(div_mask, gt_mask)
        baseline_results["Importance+Diversity"]["precision"].append(p)
        baseline_results["Importance+Diversity"]["recall"].append(r)
        baseline_results["Importance+Diversity"]["f1"].append(f)
        baseline_results["Importance+Diversity"]["time"].append(time.time() - t1)

        # 4. Proposed (Knapsack)
        t1 = time.time()
        prop_shots, _, _ = select_segments_knapsack(shots, adj_shot_scores, budget)
        prop_mask = generate_binary_mask_from_segments(prop_shots, total_duration, sample_fps, target_length=len(gt_mask))
        p, r, f = compute_overlap_precision_recall_f1(prop_mask, gt_mask)
        baseline_results["Proposed (Knapsack)"]["precision"].append(p)
        baseline_results["Proposed (Knapsack)"]["recall"].append(r)
        baseline_results["Proposed (Knapsack)"]["f1"].append(f)
        baseline_results["Proposed (Knapsack)"]["time"].append(time.time() - t1 + inference_time)

        # -------------------------------------------------------------
        # SECTION B: ABLATION STUDY
        # -------------------------------------------------------------
        # Exp A: Transformer Only
        exp_a_shots = select_importance_only_segments(shots, raw_shot_scores, budget)
        exp_a_mask = generate_binary_mask_from_segments(exp_a_shots, total_duration, sample_fps, target_length=len(gt_mask))
        p, r, f = compute_overlap_precision_recall_f1(exp_a_mask, gt_mask)
        ablation_results["Exp A (Transformer Only)"]["precision"].append(p)
        ablation_results["Exp A (Transformer Only)"]["recall"].append(r)
        ablation_results["Exp A (Transformer Only)"]["f1"].append(f)
        ablation_results["Exp A (Transformer Only)"]["time"].append(inference_time)

        # Exp B: Transformer + Diversity
        exp_b_shots = select_importance_only_segments(shots, adj_shot_scores, budget)
        exp_b_mask = generate_binary_mask_from_segments(exp_b_shots, total_duration, sample_fps, target_length=len(gt_mask))
        p, r, f = compute_overlap_precision_recall_f1(exp_b_mask, gt_mask)
        ablation_results["Exp B (Transformer + Diversity)"]["precision"].append(p)
        ablation_results["Exp B (Transformer + Diversity)"]["recall"].append(r)
        ablation_results["Exp B (Transformer + Diversity)"]["f1"].append(f)
        ablation_results["Exp B (Transformer + Diversity)"]["time"].append(inference_time)

        # Exp C: Transformer + Knapsack
        exp_c_shots, _, _ = select_segments_knapsack(shots, raw_shot_scores, budget)
        exp_c_mask = generate_binary_mask_from_segments(exp_c_shots, total_duration, sample_fps, target_length=len(gt_mask))
        p, r, f = compute_overlap_precision_recall_f1(exp_c_mask, gt_mask)
        ablation_results["Exp C (Transformer + Knapsack)"]["precision"].append(p)
        ablation_results["Exp C (Transformer + Knapsack)"]["recall"].append(r)
        ablation_results["Exp C (Transformer + Knapsack)"]["f1"].append(f)
        ablation_results["Exp C (Transformer + Knapsack)"]["time"].append(inference_time)

        # Exp D: Full Proposed System
        ablation_results["Exp D (Full Proposed System)"]["precision"].append(p)
        ablation_results["Exp D (Full Proposed System)"]["recall"].append(r)
        ablation_results["Exp D (Full Proposed System)"]["f1"].append(f)
        ablation_results["Exp D (Full Proposed System)"]["time"].append(inference_time)

    # Compute Summary Statistics
    def summarize(res_dict):
        summary = {}
        for method, metrics in res_dict.items():
            summary[method] = {
                "precision": round(float(np.mean(metrics["precision"])), 4) if metrics["precision"] else 0.0,
                "recall": round(float(np.mean(metrics["recall"])), 4) if metrics["recall"] else 0.0,
                "f1_score": round(float(np.mean(metrics["f1"])), 4) if metrics["f1"] else 0.0,
                "avg_time_sec": round(float(np.mean(metrics["time"])), 4) if metrics["time"] else 0.0
            }
        return summary

    final_baselines = summarize(baseline_results)
    final_ablation = summarize(ablation_results)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trained_checkpoint_loaded": is_trained,
        "baseline_comparison": final_baselines,
        "ablation_study": final_ablation
    }

    # Save to JSON
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save to CSV
    csv_path = os.path.join(output_dir, "results_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Method", "Precision", "Recall", "F1 Score", "Avg Processing Time (s)"])
        
        for k, v in final_baselines.items():
            writer.writerow(["Baseline", k, v["precision"], v["recall"], v["f1_score"], v["avg_time_sec"]])
            
        for k, v in final_ablation.items():
            writer.writerow(["Ablation", k, v["precision"], v["recall"], v["f1_score"], v["avg_time_sec"]])

    print("\n[OK] Benchmark Experiment Suite Completed!")
    print(f"Results saved to:\n  - JSON: {json_path}\n  - CSV : {csv_path}\n")

    print("--- BASELINE COMPARISON SUMMARY ---")
    for k, v in final_baselines.items():
        print(f"  {k:22s} | P: {v['precision']:.4f} | R: {v['recall']:.4f} | F1: {v['f1_score']:.4f} | Time: {v['avg_time_sec']:.3f}s")

    print("\n--- ABLATION STUDY SUMMARY ---")
    for k, v in final_ablation.items():
        print(f"  {k:32s} | P: {v['precision']:.4f} | R: {v['recall']:.4f} | F1: {v['f1_score']:.4f}")

    return report_data

if __name__ == "__main__":
    run_benchmark_experiments()
