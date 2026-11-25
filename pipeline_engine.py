"""
===============================================================================
Module: Master Pipeline Engine (pipeline_engine.py) - ORIGINAL AUDIO PRESERVED
===============================================================================
1. Purpose:
   Executes the speed-optimized end-to-end 25+ stage AI-Powered Lecture Video Processing,
   Intelligent Summarization, Timestamp Alignment, and Short Video Generation pipeline.
2. MANDATED AUDIO REQUIREMENT:
   - audio_mode = "original" (Preserves ORIGINAL LECTURER VOICE from source video)
   - Disables TTS / AI narration generation for the summarized video audio
   - Remaps subtitle timestamps to ORIGINAL_AUDIO summary timeline
===============================================================================
"""

import os
import time
import json
import hashlib
from typing import Dict, Any, Callable, Optional

from models.job import ProcessingJob, ProcessingStage
from services.video_processor import validate_and_inspect_video
from services.audio_processor import process_lecture_audio
from services.transcription_service import run_transcription
from services.transcript_processor import process_and_segment_transcript
from services.lecture_preprocessor import classify_lecture_sentences
from services.embedding_service import generate_sentence_embeddings
from services.importance_service import compute_importance_scores
from services.redundancy_service import apply_redundancy_deduplication
from services.topic_service import segment_lecture_topics
from services.summarization_service import generate_lecture_summary
from services.alignment_service import align_summary_to_transcript
from services.segment_service import select_optimal_video_segments
from services.video_composer import compose_final_lecture_video
from services.subtitle_service import generate_subtitles_and_keywords
from services.translation_service import translate_lecture_summary, translate_structured_outputs

CACHE_DIR = os.path.join("cache", "pipeline")

def compute_video_hash(file_path: str) -> str:
    """Calculates fast MD5 hash of video file metadata (size + header)."""
    if not os.path.exists(file_path):
        return "unknown"
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    hash_str = f"{file_name}_{file_size}"
    return hashlib.md5(hash_str.encode('utf-8')).hexdigest()[:10]

def run_lecture_processing_pipeline(
    video_path: str,
    target_duration_sec: float = 120.0,
    summary_type: str = "hybrid",
    target_language: str = "en",
    caption_option: str = "original",
    audio_mode: str = "original",
    job_id: str = "job_default",
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """Executes full 25+ stage lecture summarization pipeline using ORIGINAL LECTURER AUDIO."""
    pipeline_start = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)

    job = ProcessingJob(job_id=job_id, lecture_id=f"lec_{job_id}")

    v_hash = compute_video_hash(video_path)
    cache_key = f"{v_hash}_{int(target_duration_sec)}_{summary_type}_{target_language}_{caption_option}_{audio_mode}"
    cache_json_path = os.path.join(CACHE_DIR, f"{cache_key}_pipeline_cache.json")

    def update_status(stage: str, progress_pct: int, msg: str):
        job.update_stage(stage, msg, progress_pct)
        print(f"[{progress_pct:3d}%] [{stage}] {msg}")
        if progress_callback:
            progress_callback(stage, progress_pct, msg)

    # -------------------------------------------------------------------------
    # GLOBAL CACHE CHECK
    # -------------------------------------------------------------------------
    if not force_recompute and os.path.exists(cache_json_path):
        try:
            with open(cache_json_path, "r", encoding="utf-8") as f:
                cached_res = json.load(f)
            if cached_res.get("output_summary_video") and os.path.exists(cached_res["output_summary_video"]):
                print(f"\n[PIPELINE CACHE HIT] Returning precomputed result for '{os.path.basename(video_path)}' in < 0.1s!")
                update_status("COMPLETED", 100, "Loaded from cache instantly!")
                cached_res["cached"] = True
                cached_res["job_id"] = job_id
                job.result = cached_res
                return cached_res
        except Exception as ce:
            print(f"[PIPELINE Warning] Cache load failed ({ce}). Running pipeline...")

    perf_times = {}

    try:
        print(f"\n==================================================")
        print(f"   STARTING LECTURE PROCESSING PIPELINE           ")
        print(f"   Video: {video_path}")
        print(f"   Audio Mode: {audio_mode.upper()} (ORIGINAL SPEAKER VOICE)")
        print(f"   Target Duration: {target_duration_sec}s | Summary: {summary_type.upper()} | Language: {target_language.upper()}")
        print(f"==================================================")

        # -------------------------------------------------------------------------
        # STAGE 1: Video Validation & Metadata Extraction (0–10%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        update_status("VALIDATING", 5, "Validating video format & audio stream integrity...")
        video_meta = validate_and_inspect_video(video_path)

        if not video_meta.get("has_audio"):
            print("[ERROR] Source video contains no audio stream.")
            raise ValueError("Source video contains no audio stream.")

        original_duration = video_meta["duration_seconds"]
        target_duration_sec = min(original_duration - 1.0, max(5.0, target_duration_sec))
        perf_times["metadata_extraction"] = round(time.time() - t0, 3)
        print(f"[PERF] Metadata extraction: {perf_times['metadata_extraction']} sec")
        update_status("VALIDATING", 10, f"Video validated ({video_meta['duration_seconds']}s, {video_meta['resolution']})")

        # -------------------------------------------------------------------------
        # STAGE 2: Audio Extraction & Preprocessing (10–20%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        print(f"[20%] Audio extraction started")
        update_status("EXTRACTING_AUDIO", 15, "Extracting speech-normalized 16kHz mono WAV audio...")
        audio_path = process_lecture_audio(video_path)
        perf_times["audio_extraction"] = round(time.time() - t0, 3)
        print(f"[22%] Audio extraction completed ({perf_times['audio_extraction']}s)")
        update_status("PREPROCESSING_AUDIO", 20, "✓ Audio extraction complete & volume normalized.")

        # -------------------------------------------------------------------------
        # STAGE 3: Automatic Speech Recognition (Whisper ASR) (20–50%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        update_status("TRANSCRIBING", 25, "Starting Whisper ASR speech recognition...")
        raw_segments, transcript_text, detected_lang = run_transcription(
            audio_path,
            progress_callback=update_status,
            force_recompute=force_recompute
        )
        perf_times["whisper_transcription"] = round(time.time() - t0, 3)
        print(f"[PERF] Whisper transcription: {perf_times['whisper_transcription']} sec")
        update_status("TRANSCRIBING", 50, f"Whisper ASR completed ({len(raw_segments)} speech segments).")

        # -------------------------------------------------------------------------
        # STAGE 4: Transcript Post-Processing & Sentence Mapping (50–60%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        print(f"[50%] Transcript cleaning started")
        update_status("POST_PROCESSING_TRANSCRIPT", 52, "Cleaning transcript & mapping sentence timestamps...")
        sentences = process_and_segment_transcript(raw_segments)
        perf_times["transcript_processing"] = round(time.time() - t0, 3)
        print(f"[PERF] Transcript processing: {perf_times['transcript_processing']} sec")
        update_status("POST_PROCESSING_TRANSCRIPT", 60, f"Segmented {len(sentences)} timestamped sentences.")

        # -------------------------------------------------------------------------
        # STAGE 5: Educational Classification & Dense Embeddings (60–75%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        update_status("SCORING_IMPORTANCE", 62, "Classifying educational structure (definitions, formulas, concepts)...")
        sentences = classify_lecture_sentences(sentences)

        t_emb = time.time()
        update_status("SCORING_IMPORTANCE", 65, "Generating batched sentence vector embeddings...")
        sentences = generate_sentence_embeddings(sentences)
        perf_times["embedding_generation"] = round(time.time() - t_emb, 3)
        print(f"[PERF] Embedding generation: {perf_times['embedding_generation']} sec")

        t_imp = time.time()
        update_status("SCORING_IMPORTANCE", 70, "Computing multi-factor educational importance scores...")
        sentences = compute_importance_scores(sentences)
        sentences = apply_redundancy_deduplication(sentences)
        sentences, topic_sections = segment_lecture_topics(sentences)
        perf_times["importance_scoring"] = round(time.time() - t_imp, 3)
        print(f"[PERF] Importance scoring: {perf_times['importance_scoring']} sec")
        update_status("SCORING_IMPORTANCE", 75, "Importance scoring & topic segmentation complete.")

        # -------------------------------------------------------------------------
        # STAGE 6: Summarization (75–85%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        update_status("GENERATING_SUMMARY", 78, f"Generating concise {summary_type.upper()} lecture summary...")
        lecture_summary = generate_lecture_summary(
            sentences=sentences,
            topic_sections=topic_sections,
            summary_type=summary_type,
            target_duration_sec=target_duration_sec,
            lecture_id=job.lecture_id
        )
        perf_times["summarization"] = round(time.time() - t0, 3)
        print(f"[PERF] Summarization: {perf_times['summarization']} sec")
        update_status("GENERATING_SUMMARY", 85, "Summary generated successfully.")

        # -------------------------------------------------------------------------
        # STAGE 7: Summary-to-Transcript Semantic Alignment (85–92%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        update_status("ALIGNING_SUMMARY", 88, "Semantically mapping summary back to original video timestamps...")
        aligned_summary = align_summary_to_transcript(lecture_summary, sentences)
        perf_times["timestamp_mapping"] = round(time.time() - t0, 3)
        print(f"[PERF] Timestamp mapping: {perf_times['timestamp_mapping']} sec")
        update_status("ALIGNING_SUMMARY", 92, "Summary timestamp alignment complete.")

        # -------------------------------------------------------------------------
        # STAGE 8: 0/1 Knapsack Segment Selection (90–92%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        update_status("SELECTING_SEGMENTS", 90, "Solving 0/1 Knapsack for target duration clip selection...")
        raw_intervals = [
            (stmt.start_time, stmt.end_time, stmt.similarity_score)
            for stmt in aligned_summary.statements if stmt.end_time > stmt.start_time
        ]

        final_clip_tuples, video_clips, selected_clip_duration = select_optimal_video_segments(
            raw_intervals=raw_intervals,
            sentences=sentences,
            target_duration_sec=target_duration_sec
        )
        perf_times["video_extraction"] = round(time.time() - t0, 3)
        print(f"[PERF] Segment selection: {perf_times['video_extraction']} sec")

        # -------------------------------------------------------------------------
        # STAGE 9: Subtitles, Translation & Final Video Composition with ORIGINAL AUDIO (92–100%)
        # -------------------------------------------------------------------------
        t0 = time.time()
        update_status("TRANSLATING", 92, f"Translating summary, transcript & generating SRT subtitles ({target_language.upper()})...")
        translated_summary_text = translate_lecture_summary(aligned_summary.text, target_language=target_language)

        raw_structured_sections = getattr(aligned_summary, "structured_sections", [])
        raw_summary_statements = [st.to_dict() for st in aligned_summary.statements]
        raw_topic_timeline = getattr(aligned_summary, "topic_timeline", [])
        raw_timestamped_sentences = [s.to_dict() for s in sentences]

        (
            out_structured_sections,
            out_summary_statements,
            out_topic_timeline,
            out_timestamped_sentences,
            out_raw_transcript,
        ) = translate_structured_outputs(
            target_language=target_language,
            structured_sections=raw_structured_sections,
            summary_statements=raw_summary_statements,
            topic_timeline=raw_topic_timeline,
            timestamped_sentences=raw_timestamped_sentences,
            raw_transcript=transcript_text,
        )

        subtitle_lang = target_language if caption_option == "translated" else "en"
        srt_path = os.path.join("outputs", "subtitles", f"{job_id}_{subtitle_lang}.srt")
        srt_file, keywords = generate_subtitles_and_keywords(
            summary_text=translated_summary_text if caption_option == "translated" else aligned_summary.text,
            total_duration_sec=selected_clip_duration,
            output_srt_path=srt_path,
            stt_segments=[s.to_dict() for s in raw_segments],
            selected_segments=final_clip_tuples,
            audio_mode=audio_mode,
            target_language=subtitle_lang
        )

        output_video_path = os.path.join("outputs", f"{os.path.splitext(video_meta['filename'])[0]}_summary_{job_id}.mp4")
        burn_subs = caption_option in ["burn", "burn_subtitles"]

        final_video = compose_final_lecture_video(
            input_video=video_path,
            selected_segments=final_clip_tuples,
            output_path=output_video_path,
            narration_audio_path=None,  # NO TTS narration!
            srt_subtitle_path=srt_file if caption_option != "none" else None,
            audio_mode=audio_mode,
            burn_subtitles=burn_subs,
            progress_callback=update_status
        )
        perf_times["final_rendering"] = round(time.time() - t0, 3)
        print(f"[PERF] Final video composition: {perf_times['final_rendering']} sec")

        # Total Pipeline Time
        total_time = round(time.time() - pipeline_start, 3)
        perf_times["TOTAL"] = total_time

        compression_pct = round((1.0 - (selected_clip_duration / max(1.0, original_duration))) * 100, 1)

        result = {
            "job_id": job_id,
            "lecture_id": job.lecture_id,
            "input_video": video_path,
            "output_summary_video": final_video,
            "original_duration": original_duration,
            "target_duration_sec": target_duration_sec,
            "summary_duration": selected_clip_duration,
            "compression_percent": compression_pct,
            "summary_type": summary_type,
            "target_language": target_language,
            "caption_option": caption_option,
            "audio_mode": audio_mode,
            "detected_language": detected_lang,
            "raw_transcript": out_raw_transcript,
            "timestamped_sentences": out_timestamped_sentences,
            "topic_sections": topic_sections,
            "structured_sections": out_structured_sections,
            "topic_timeline": out_topic_timeline,
            "summary_text": aligned_summary.text,
            "translated_summary_text": translated_summary_text,
            "summary_statements": out_summary_statements,
            "topic_bullets": aligned_summary.topic_bullets,
            "selected_clips": [c.to_dict() for c in video_clips],
            "selected_clip_tuples": final_clip_tuples,
            "srt_subtitle_file": srt_file,
            "keywords": keywords,
            "elapsed_seconds": total_time,
            "performance_summary": perf_times,
            "metadata": video_meta
        }

        # Save to cache
        try:
            with open(cache_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as ce:
            print(f"[PIPELINE Warning] Could not write cache ({ce})")

        print(f"\n==================================================")
        print(f"PERFORMANCE BREAKDOWN SUMMARY")
        print(f"==================================================")
        for stage_name, sec_val in perf_times.items():
            if stage_name != "TOTAL":
                print(f"[PERF] {stage_name:<24}: {sec_val:.2f} sec")
        print(f"[PERF] ----------------------------------------")
        print(f"[PERF] TOTAL                   : {total_time:.2f} sec")
        print(f"==================================================")

        update_status("COMPLETED", 100, f"Lecture summarization completed in {total_time:.1f}s!")
        job.result = result
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_msg = str(e)
        update_status("FAILED", 0, f"Pipeline execution failed: {err_msg}")
        job.error = err_msg
        raise e
