import os
import sys
import time
import torch
import numpy as np

from preprocessing.video_reader import get_video_metadata
from preprocessing.shot_detector import detect_shots
from preprocessing.frame_sampler import sample_representative_shot_keyframes
from models.resnet_feature_extractor import get_resnet_feature_extractor
from models.temporal_transformer import load_trained_model, compute_shot_importance_scores
from summarization.diversity import compute_shot_feature_representations, apply_diversity_penalty
from summarization.segment_selector import select_segments_knapsack
from video_processing.clip_extractor import extract_multiple_clips
from multimodal.speech_to_text import transcribe_audio, extract_segments_audio_from_video
from multimodal.text_summary import summarize_transcript_text
from multimodal.translation import translate_text_summary
from multimodal.tts import text_to_speech
from multimodal.subtitle_generator import generate_srt_subtitles
from video_processing.video_merger import generate_summary_video

def profile_pipeline(video_path: str, target_duration_sec: float = 60.0, target_language: str = "en"):
    print(f"\n==================================================")
    print(f"   PROFILING 15 PIPELINE STAGES (UNCACHED RUN)   ")
    print(f"   Target Video: {os.path.basename(video_path)}")
    print(f"   Target Duration: {target_duration_sec}s | Lang: {target_language.upper()}")
    print(f"==================================================")

    # 1. Video loading
    t0 = time.time()
    meta = get_video_metadata(video_path)
    t_video_loading = time.time() - t0
    original_duration = meta["duration_seconds"]

    from preprocessing.shot_detector import detect_shots_and_keyframes
    # 2. Shot detection & 3. Frame sampling (Single linear forward pass)
    t0 = time.time()
    shots_list, kf_frames, kf_timestamps, kf_shot_indices = detect_shots_and_keyframes(video_path, target_fps=1.0)
    t_shot_detection = time.time() - t0
    t_frame_sampling = 0.00  # Combined in single pass

    # 4. ResNet feature extraction
    t0 = time.time()
    import models.resnet_feature_extractor
    models.resnet_feature_extractor._GLOBAL_RESNET_EXTRACTOR = None
    extractor = get_resnet_feature_extractor()
    kf_feature_batches = []
    batch_size = 32
    for i in range(0, len(kf_frames), batch_size):
        batch = kf_frames[i : i + batch_size]
        feats = extractor.extract_batch(batch)
        kf_feature_batches.append(feats)
    kf_features = np.vstack(kf_feature_batches) if kf_feature_batches else np.empty((0, 2048), dtype=np.float32)
    
    num_shots = len(shots_list)
    shot_vectors = np.zeros((num_shots, 2048), dtype=np.float32)
    shot_counts = np.zeros(num_shots, dtype=np.int32)
    for feat, s_idx in zip(kf_features, kf_shot_indices):
        if 0 <= s_idx < num_shots:
            shot_vectors[s_idx] += feat
            shot_counts[s_idx] += 1
    for s_idx in range(num_shots):
        if shot_counts[s_idx] > 0:
            shot_vectors[s_idx] /= shot_counts[s_idx]
            norm = np.linalg.norm(shot_vectors[s_idx])
            if norm > 1e-6:
                shot_vectors[s_idx] /= norm

    num_timestamps = max(1, int(round(original_duration * 2.0)))
    timestamps_list = [round(i / 2.0, 3) for i in range(num_timestamps)]
    timestamps = np.array(timestamps_list, dtype=np.float32)
    features = np.zeros((num_timestamps, 2048), dtype=np.float32)
    for t_idx, ts_sec in enumerate(timestamps_list):
        matched_shot = 0
        for s_idx, (s_start, s_end) in enumerate(shots_list):
            if s_start <= ts_sec <= s_end:
                matched_shot = s_idx
                break
        features[t_idx] = shot_vectors[matched_shot]
    shots = np.array(shots_list, dtype=np.float32)
    t_resnet = time.time() - t0

    # 5. Temporal Transformer
    t0 = time.time()
    model, is_trained = load_trained_model(device=extractor.device)
    with torch.no_grad():
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(extractor.device)
        frame_scores_tensor = model(feat_tensor)
        frame_scores = frame_scores_tensor.squeeze(0).cpu().numpy()
    shot_scores = compute_shot_importance_scores(frame_scores, timestamps, shots.tolist())
    t_transformer = time.time() - t0

    # 6. Diversity calculation
    t0 = time.time()
    shot_features = compute_shot_feature_representations(features, timestamps, shots.tolist())
    adjusted_scores = apply_diversity_penalty(shot_scores, shot_features, diversity_lambda=0.3)
    t_diversity = time.time() - t0

    # 7. Knapsack selection
    t0 = time.time()
    selected_shots, selected_indices, summary_duration = select_segments_knapsack(
        shots.tolist(), adjusted_scores, target_duration_sec
    )
    selected_shots.sort(key=lambda x: x[0])
    t_knapsack = time.time() - t0

    # 8. Clip extraction (Handled in single pass during final FFmpeg rendering)
    t_clip_extraction = 0.00

    # 9. Audio extraction
    t0 = time.time()
    output_dir = os.path.join("outputs", "multimodal_profile")
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "summary_segment_audio.wav")
    extract_segments_audio_from_video(video_path, selected_shots, wav_path)
    t_audio_extraction = time.time() - t0

    # 10. Whisper STT
    t0 = time.time()
    stt_res = transcribe_audio(
        video_path,
        output_dir=output_dir,
        selected_segments=selected_shots,
        force_recompute=True
    )
    t_whisper = time.time() - t0
    transcript_text = stt_res.get("transcript_text", "")

    # 11. Text summary
    t0 = time.time()
    summary_res = summarize_transcript_text(
        transcript_text,
        output_dir=output_dir,
        target_duration_sec=target_duration_sec
    )
    t_text_summary = time.time() - t0
    summary_text = summary_res.get("text_summary", "")

    # 12. Translation
    t0 = time.time()
    trans_res = translate_text_summary(summary_text, target_lang=target_language, output_dir=output_dir)
    t_translation = time.time() - t0
    translated_text = trans_res.get("translated_text", "")

    # 13. TTS
    t0 = time.time()
    tts_res = text_to_speech(translated_text, language=target_language, output_dir=output_dir, force_recompute=True)
    t_tts = time.time() - t0
    audio_file = tts_res.get("audio_file")
    audio_dur = tts_res.get("audio_duration_sec", 30.0)

    # 14. SRT generation
    t0 = time.time()
    srt_path = os.path.join(output_dir, "summary_captions.srt")
    generate_srt_subtitles(
        summary_text=translated_text,
        total_audio_duration_sec=audio_dur,
        output_srt_path=srt_path,
        stt_segments=stt_res.get("segments"),
        selected_segments=selected_shots
    )
    t_srt = time.time() - t0

    # 15. Final FFmpeg rendering
    t0 = time.time()
    final_output = os.path.join(output_dir, "profile_final_summary.mp4")
    generate_summary_video(
        input_video=video_path,
        selected_segments=selected_shots,
        output_summary_path=final_output,
        narration_audio_path=audio_file,
        srt_subtitle_path=srt_path
    )
    t_ffmpeg = time.time() - t0

    total_time = (
        t_video_loading + t_shot_detection + t_frame_sampling + t_resnet +
        t_transformer + t_diversity + t_knapsack + t_clip_extraction +
        t_audio_extraction + t_whisper + t_text_summary + t_translation +
        t_tts + t_srt + t_ffmpeg
    )

    print("\n========== PERFORMANCE ==========")
    print(f"Video loading       : {t_video_loading:.2f} sec")
    print(f"Shot detection      : {t_shot_detection:.2f} sec")
    print(f"Frame sampling      : {t_frame_sampling:.2f} sec")
    print(f"ResNet              : {t_resnet:.2f} sec")
    print(f"Transformer         : {t_transformer:.2f} sec")
    print(f"Diversity           : {t_diversity:.2f} sec")
    print(f"Knapsack            : {t_knapsack:.2f} sec")
    print(f"Clip extraction     : {t_clip_extraction:.2f} sec")
    print(f"Audio extraction    : {t_audio_extraction:.2f} sec")
    print(f"Whisper             : {t_whisper:.2f} sec")
    print(f"Text summary        : {t_text_summary:.2f} sec")
    print(f"Translation         : {t_translation:.2f} sec")
    print(f"TTS                 : {t_tts:.2f} sec")
    print(f"SRT                 : {t_srt:.2f} sec")
    print(f"FFmpeg              : {t_ffmpeg:.2f} sec")
    print("-----------------------------------")
    print(f"TOTAL               : {total_time:.2f} sec")
    print("===================================\n")

if __name__ == "__main__":
    uploads = [f for f in os.listdir("uploads") if f.endswith(".mp4")]
    if uploads:
        vid_p = os.path.join("uploads", uploads[0])
        profile_pipeline(vid_p, target_duration_sec=60.0, target_language="en")
