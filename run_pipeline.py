import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Any

from config import DIVERSITY_LAMBDA, FRAME_SAMPLE_FPS, PROCESSING_MODE
from preprocessing.video_reader import get_video_metadata
from preprocessing.shot_detector import detect_shots
from models.resnet_feature_extractor import get_resnet_feature_extractor
from models.temporal_transformer import load_trained_model, compute_shot_importance_scores
from summarization.diversity import compute_shot_feature_representations, apply_diversity_penalty
from summarization.segment_selector import select_segments_knapsack
from video_processing.video_merger import generate_summary_video
from multimodal.pipeline import run_multimodal_pipeline

def run_summarization_pipeline(
    video_path: str,
    target_duration_sec: float = None,
    target_language: str = "en",
    target_ratio: float = None,
    sample_fps: float = None,
    output_path: str = None,
    progress_callback = None,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Executes the complete AI-Powered Multimodal Video-to-Video Summarization Pipeline:
    1. Reads Video Metadata
    2. Samples Frames & Detects Continuous Shots
    3. Extracts ResNet Features (CUDA/CPU, Batched, Cached)
    4. Predicts Temporal Importance Scores via Transformer
    5. Applies Visual Diversity Penalty
    6. Selects Optimal Segments via 0/1 Knapsack under Target Duration Constraint
    7. Cuts & Merges Visual Clips
    8. Runs Multimodal Pipeline: STT -> Abstractive Summary -> Translation -> AI Narration MP3 -> SRT Subtitles
    9. FFmpeg Rendering: Mutes original audio, muxes AI narration, burns SRT subtitles
    10. Performance Profiling & ffprobe Target Duration Validation
    """
    pipeline_start_time = time.time()
    sample_fps = sample_fps or FRAME_SAMPLE_FPS or 2.0

    print(f"\n==================================================")
    print(f"   STARTING VIDEO SUMMARIZATION PIPELINE         ")
    print(f"   PROCESSING MODE: {PROCESSING_MODE}             ")
    print(f"==================================================")

    # Stage 1: Video Preprocessing (Metadata Extraction)
    t_stage_start = time.time()
    print("\n[Step 1/10] Extracting Video Metadata...")
    if progress_callback: progress_callback(10, 1, "Extracting video metadata...")
    metadata = get_video_metadata(video_path)
    time_preprocessing = time.time() - t_stage_start
    print(f"[PERFORMANCE] Video preprocessing: {time_preprocessing:.2f} seconds")

    original_duration = metadata["duration_seconds"]

    if target_duration_sec is None:
        if target_ratio is not None:
            target_duration_sec = round(original_duration * target_ratio, 2)
        else:
            target_duration_sec = round(original_duration * 0.15, 2)

    # Enforce upper bound constraint (target < original)
    target_duration_sec = min(original_duration - 1.0, max(5.0, target_duration_sec))

    print(f"[DURATION DEBUG]")
    print(f"Frontend selected duration: {target_duration_sec:.2f} seconds")
    print(f"Pipeline target duration  : {target_duration_sec:.2f} seconds")
    print(f"Input Video               : {video_path}")
    print(f"Original Length           : {original_duration:.2f}s ({metadata['duration_formatted']})")
    print(f"Target Language           : {target_language.upper()}")

    # Stage 2: ResNet Visual Feature Extraction
    t_stage_start = time.time()
    print("\n[Step 2-4/10] Extracting Visual Features & Detecting Scenes...")
    if progress_callback: progress_callback(30, 4, "Extracting visual features & detecting scenes...")
    extractor = get_resnet_feature_extractor()
    features, timestamps, shots = extractor.extract_and_cache(video_path, target_fps=sample_fps, force_recompute=force_recompute)
    time_feat_extraction = time.time() - t_stage_start
    print(f"[PERFORMANCE] Feature extraction: {time_feat_extraction:.2f} seconds")

    # Stage 3: Temporal Transformer Importance Prediction
    t_stage_start = time.time()
    print("\n[Step 5/10] Computing Temporal Importance Scores via Transformer...")
    if progress_callback: progress_callback(45, 5, "Predicting temporal scene importance...")
    model, is_trained = load_trained_model(device=extractor.device)

    with torch.no_grad():
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(extractor.device)
        frame_scores_tensor = model(feat_tensor)
        frame_scores = frame_scores_tensor.squeeze(0).cpu().numpy()

    shot_scores = compute_shot_importance_scores(frame_scores, timestamps, shots.tolist())
    time_transformer = time.time() - t_stage_start
    print(f"[PERFORMANCE] Transformer: {time_transformer:.2f} seconds")

    # Stage 4: Diversity & Redundancy Penalty + Segment Selection (0/1 Knapsack)
    t_stage_start = time.time()
    print("\n[Step 6-7/10] Applying Diversity Penalty & Solving 0/1 Knapsack Selection...")
    if progress_callback: progress_callback(60, 6, "Selecting optimal video segments...")
    shot_features = compute_shot_feature_representations(features, timestamps, shots.tolist())
    adjusted_scores = apply_diversity_penalty(shot_scores, shot_features, diversity_lambda=DIVERSITY_LAMBDA)
    
    selected_shots, selected_indices, summary_duration = select_segments_knapsack(
        shots.tolist(), adjusted_scores, target_duration_sec
    )
    selected_shots.sort(key=lambda x: x[0])
    time_selection = time.time() - t_stage_start
    print(f"[PERFORMANCE] Selection: {time_selection:.2f} seconds")

    # Stage 5: Clip Generation & Extraction
    t_stage_start = time.time()
    print(f"\n[Step 8/10] Selected {len(selected_shots)} key visual segments (Duration: {summary_duration:.2f}s)...")
    if progress_callback: progress_callback(70, 7, "Preparing visual clips...")
    time_clip_gen = time.time() - t_stage_start
    print(f"[PERFORMANCE] Clip generation: {time_clip_gen:.2f} seconds")

    # Stage 6: Multimodal Processing (STT, Text Summary, Translation, TTS, SRT)
    t_stage_start = time.time()
    print("\n[Step 9/10] Executing Multimodal Pipeline (Whisper STT, Summary, Translation, TTS & Subtitles)...")
    if progress_callback: progress_callback(80, 8, "Generating AI narration & subtitles...")

    mm_dir = os.path.join("outputs", "multimodal")
    mm_res = run_multimodal_pipeline(
        summary_video_path=video_path,
        output_dir=mm_dir,
        target_language=target_language,
        target_duration_sec=target_duration_sec,
        selected_segments=selected_shots,
        original_video_path=video_path,
        force_recompute=force_recompute
    )
    time_multimodal = time.time() - t_stage_start

    mm_perf = mm_res.get("performance", {})
    time_whisper = mm_perf.get("time_stt_sec", 0.0)
    time_text_summary = mm_perf.get("time_summary_sec", 0.0)
    time_translation = mm_perf.get("time_translation_sec", 0.0)
    time_tts = mm_perf.get("time_tts_sec", 0.0)
    time_srt = mm_perf.get("time_srt_sec", 0.0)

    narration_audio = mm_res.get("audio_file") if mm_res.get("success") else None
    srt_subtitles = mm_res.get("srt_file") if mm_res.get("success") else None

    # Stage 7: Final FFmpeg Rendering (Mute original, Mux narration, Burn subtitles)
    t_stage_start = time.time()
    print("\n[Step 10/10] Rendering Final Video via FFmpeg...")
    if progress_callback: progress_callback(92, 9, "Rendering final video summary via FFmpeg...")

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join("outputs", f"{base_name}_summary_{target_language}_{int(target_duration_sec)}s.mp4")

    final_video = generate_summary_video(
        input_video=video_path,
        selected_segments=selected_shots,
        output_summary_path=output_path,
        narration_audio_path=narration_audio,
        srt_subtitle_path=srt_subtitles
    )
    time_rendering = time.time() - t_stage_start
    print(f"[PERFORMANCE] Rendering: {time_rendering:.2f} seconds")

    total_pipeline_time = time.time() - pipeline_start_time

    # Validate output duration via ffprobe
    final_meta = get_video_metadata(final_video)
    final_duration = final_meta["duration_seconds"]

    min_dur = max(5.0, target_duration_sec - 5.0)
    max_dur = target_duration_sec + 2.0
    passed = (min_dur <= final_duration <= max_dur)
    status_str = "PASS" if passed else "WARNING: duration mismatch"

    print(f"\n[TARGET] Requested: {target_duration_sec:.2f}s")
    print(f"[OUTPUT] Actual: {final_duration:.2f}s")
    print(f"[VALIDATION] {status_str}")

    print(f"\n==================================================")
    print(f"PERFORMANCE SUMMARY")
    print(f"==================================================")
    print(f"Video preprocessing : {time_preprocessing:.2f}s")
    print(f"Feature extraction  : {time_feat_extraction:.2f}s")
    print(f"Transformer         : {time_transformer:.2f}s")
    print(f"Selection           : {time_selection:.2f}s")
    print(f"Clip generation     : {time_clip_gen:.2f}s")
    print(f"Whisper             : {time_whisper:.2f}s")
    print(f"Text summary        : {time_text_summary:.2f}s")
    print(f"Translation         : {time_translation:.2f}s")
    print(f"TTS                 : {time_tts:.2f}s")
    print(f"SRT                 : {time_srt:.2f}s")
    print(f"Rendering           : {time_rendering:.2f}s")
    print(f"TOTAL               : {total_pipeline_time:.2f}s")
    print(f"==================================================")

    if progress_callback: progress_callback(100, 10, "Summarization completed successfully!")

    compression_pct = round((1 - (summary_duration / original_duration)) * 100, 1)

    return {
        "input_video": video_path,
        "output_summary": final_video,
        "original_duration": original_duration,
        "target_duration_sec": target_duration_sec,
        "summary_duration": summary_duration,
        "target_language": target_language,
        "compression_percent": compression_pct,
        "selected_segments": selected_shots,
        "elapsed_seconds": round(total_pipeline_time, 2),
        "performance_summary": {
            "preprocessing": round(time_preprocessing, 2),
            "feature_extraction": round(time_feat_extraction, 2),
            "transformer": round(time_transformer, 2),
            "selection": round(time_selection, 2),
            "clip_generation": round(time_clip_gen, 2),
            "whisper": round(time_whisper, 2),
            "text_summary": round(time_text_summary, 2),
            "translation": round(time_translation, 2),
            "tts": round(time_tts, 2),
            "srt": round(time_srt, 2),
            "rendering": round(time_rendering, 2),
            "total": round(total_pipeline_time, 2)
        },
        "multimodal": mm_res
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vpath = sys.argv[1]
        tdur = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
        tlang = sys.argv[3] if len(sys.argv) > 3 else "en"
        run_summarization_pipeline(vpath, target_duration_sec=tdur, target_language=tlang)
    else:
        print("Usage: python run_pipeline.py <video_path> [target_duration_sec e.g. 120] [target_language e.g. hi]")
