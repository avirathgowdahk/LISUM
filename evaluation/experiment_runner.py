"""
===============================================================================
Module: Academic Experiments (evaluation/experiment_runner.py)
===============================================================================
1. Purpose:
   Automates academic benchmark comparisons across Extractive, Abstractive, and Hybrid
   lecture summarization approaches as specified in Section 43 of project requirements.
2. Output:
   Comparative evaluation report JSON and tabular metric breakdown.
===============================================================================
"""

import time
import json
from typing import Dict, Any, List
from pipeline_engine import run_lecture_processing_pipeline
from evaluation.metrics import calculate_rouge_scores, evaluate_duration_compliance

def run_academic_ablation_experiment(video_path: str, target_duration_sec: float = 120.0) -> Dict[str, Any]:
    """Runs comparative evaluation between Extractive, Abstractive, and Hybrid pipeline variants."""
    print(f"\n==================================================")
    print(f"   STARTING ACADEMIC ABLATION EXPERIMENTS         ")
    print(f"==================================================")

    results = {}

    for mode in ["extractive", "abstractive", "hybrid"]:
        print(f"\n--- Running Experiment: Mode = '{mode.upper()}' ---")
        t0 = time.time()

        res = run_lecture_processing_pipeline(
            video_path=video_path,
            target_duration_sec=target_duration_sec,
            summary_type=mode,
            job_id=f"exp_{mode}",
            force_recompute=True
        )

        t_elapsed = round(time.time() - t0, 2)
        dur_eval = evaluate_duration_compliance(target_duration_sec, res["summary_duration"])

        # Compute self-ROUGE against full transcript as coverage reference
        rouge_metrics = calculate_rouge_scores(res["raw_transcript"][:2000], res["summary_text"])

        results[mode] = {
            "summary_type": mode,
            "processing_time_sec": t_elapsed,
            "summary_word_count": len(res["summary_text"].split()),
            "output_duration_sec": res["summary_duration"],
            "duration_error_sec": dur_eval["absolute_difference_sec"],
            "duration_compliant": dur_eval["compliant"],
            "rouge_metrics": rouge_metrics,
            "topic_coverage_count": len(res["topic_bullets"]),
            "selected_clips_count": len(res["selected_clips"])
        }

    report_path = "outputs/academic_experiment_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================================================")
    print(f"   ACADEMIC EXPERIMENT COMPLETE! Saved to '{report_path}'")
    print(f"==================================================")
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_academic_ablation_experiment(sys.argv[1])
    else:
        print("Usage: python evaluation/experiment_runner.py <video_path>")
