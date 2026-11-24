import os
import sys
import json
import time
import hashlib
from typing import Dict, Any, List, Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal.speech_to_text import transcribe_audio
from multimodal.text_summary import summarize_transcript_text
from multimodal.translation import translate_text_summary
from multimodal.tts import text_to_speech
from multimodal.subtitle_generator import generate_srt_subtitles
from config import WHISPER_MODEL

def run_multimodal_pipeline(
    summary_video_path: str,
    output_dir: str = "outputs/multimodal",
    target_language: str = "en",
    target_duration_sec: float = None,
    force_recompute: bool = False,
    selected_segments: List[Tuple[float, float]] = None,
    original_video_path: str = None
) -> Dict[str, Any]:
    """
    Executes single-language multimodal processing pipeline with timing profiling & caching:
    1. Speech-to-Text Transcription via singleton Whisper
    2. Abstractive/Narration-ready Text Summarization
    3. Translation into target language
    4. Text-To-Speech (gTTS / pyttsx3) Audio Narration
    5. Synchronized SRT Subtitle Generation
    """
    start_time = time.time()
    lang = target_language.lower().strip()

    abs_path = os.path.abspath(summary_video_path)
    file_size = os.path.getsize(summary_video_path) if os.path.exists(summary_video_path) else 0
    file_base = os.path.splitext(os.path.basename(summary_video_path))[0]
    target_dur_str = f"{float(target_duration_sec):.1f}" if target_duration_sec else "full"
    path_hash = hashlib.md5(f"{abs_path}_{file_size}_{target_dur_str}_{lang}".encode('utf-8')).hexdigest()[:8]
    base_name = f"{file_base}_{path_hash}"

    result_json_path = os.path.join(output_dir, f"{base_name}_{lang}_multimodal_result.json")

    # Pipeline Level Cache Check
    if not force_recompute and os.path.exists(result_json_path):
        try:
            with open(result_json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            tr_txt = cached_data.get("translated_text", "")
            if cached_data.get("success") and tr_txt and not ("Error 500" in tr_txt or "Server Error" in tr_txt or "That’s an error" in tr_txt):
                print(f"[MULTIMODAL] Returning cached multimodal pipeline results for '{base_name}' ({lang})...")
                srt_path = cached_data.get("srt_file") or os.path.join(output_dir, f"{base_name}_{lang}_captions.srt")
                t_srt_cache_start = time.time()
                generate_srt_subtitles(
                    summary_text=cached_data.get("translated_text", ""),
                    total_audio_duration_sec=cached_data.get("audio_duration_sec", 30.0),
                    output_srt_path=srt_path,
                    stt_segments=cached_data.get("stt_segments"),
                    selected_segments=selected_segments
                )
                time_srt_cache = time.time() - t_srt_cache_start
                cached_data["srt_file"] = srt_path
                cached_data["cached"] = True
                print(f"[PERFORMANCE] SRT Generation: {time_srt_cache:.2f} seconds (Cached Pipeline)")
                return cached_data
        except Exception as ce:
            print(f"[MULTIMODAL Warning] Cache load failed: {ce}")

    print(f"\n==================================================")
    print(f"   STARTING MULTIMODAL ENHANCEMENT PIPELINE       ")
    print(f"==================================================")
    print(f"Target Video   : {summary_video_path}")
    print(f"Target Language: {lang.upper()}")
    print(f"Target Duration: {target_duration_sec}s" if target_duration_sec else "Target Duration: Full")

    # Stage 1: Speech-To-Text Transcription
    t0 = time.time()
    stt_res = transcribe_audio(
        summary_video_path,
        output_dir=output_dir,
        model_size=WHISPER_MODEL,
        selected_segments=selected_segments,
        original_video_path=original_video_path,
        force_recompute=force_recompute
    )
    time_stt = time.time() - t0
    print(f"[PERFORMANCE] Whisper: {time_stt:.2f} seconds")

    if not stt_res.get("success"):
        return {
            "success": False,
            "stage": stt_res.get("stage", "speech_to_text"),
            "error": stt_res.get("error", "Speech recognition failed."),
            "summary_video": summary_video_path
        }

    transcript_text = stt_res["transcript_text"]
    audio_dur = stt_res["audio_duration_sec"]
    detected_lang = stt_res["language"]

    # Stage 2: Narration-Ready Text Summarization
    t0 = time.time()
    summary_res = summarize_transcript_text(
        transcript_text,
        output_dir=output_dir,
        base_name=base_name,
        target_duration_sec=target_duration_sec
    )
    time_summary = time.time() - t0
    print(f"[PERFORMANCE] Text summary: {time_summary:.2f} seconds")

    if not summary_res.get("success"):
        return {
            "success": False,
            "stage": "text_summary",
            "error": summary_res.get("error", "Text summarization failed."),
            "transcript": transcript_text
        }

    real_summary_text = summary_res["text_summary"]
    bullet_points = summary_res["bullet_points"]

    # Stage 3: Translation into Selected Language
    t0 = time.time()
    trans_res = translate_text_summary(real_summary_text, target_lang=lang, output_dir=output_dir)
    time_trans = time.time() - t0
    print(f"[PERFORMANCE] Translation: {time_trans:.2f} seconds")

    if not trans_res.get("success"):
        return {
            "success": False,
            "stage": "translation",
            "language": lang,
            "error": trans_res.get("error", f"Translation failed for language '{lang}'")
        }

    translated_text = trans_res["translated_text"]

    # Stage 4: TTS AI Narration Synthesis
    t0 = time.time()
    tts_res = text_to_speech(translated_text, language=lang, output_dir=output_dir, base_name=f"{base_name}_{lang}")
    time_tts = time.time() - t0
    print(f"[PERFORMANCE] TTS: {time_tts:.2f} seconds")

    if not tts_res.get("success"):
        return {
            "success": False,
            "stage": "tts",
            "language": lang,
            "error": tts_res.get("error", f"TTS synthesis failed for language '{lang}'")
        }

    audio_file = tts_res["audio_file"]
    tts_duration = tts_res["audio_duration_sec"]

    # Stage 5: SRT Subtitle Generation
    t0 = time.time()
    srt_duration = max(tts_duration, float(target_duration_sec)) if target_duration_sec else tts_duration
    srt_path = os.path.join(output_dir, f"{base_name}_{lang}_captions.srt")
    generate_srt_subtitles(
        summary_text=translated_text,
        total_audio_duration_sec=srt_duration,
        output_srt_path=srt_path,
        stt_segments=stt_res.get("segments"),
        selected_segments=selected_segments
    )
    time_srt = time.time() - t0
    print(f"[PERFORMANCE] SRT: {time_srt:.2f} seconds")

    total_time = time.time() - start_time

    transcript_words = len(transcript_text.split())
    summary_words = len(real_summary_text.split())
    comp_ratio = round((summary_words / max(1, transcript_words)) * 100, 1)

    result = {
        "success": True,
        "summary_video": summary_video_path,
        "selected_language": lang,
        "detected_language": detected_lang,
        "transcript": transcript_text,
        "text_summary": real_summary_text,
        "translated_text": translated_text,
        "bullet_points": bullet_points,
        "audio_file": audio_file,
        "audio_url": f"/outputs/multimodal/{os.path.basename(audio_file)}",
        "audio_duration_sec": tts_duration,
        "srt_file": srt_path,
        "srt_url": f"/outputs/multimodal/{os.path.basename(srt_path)}",
        "stt_segments": stt_res.get("segments"),
        "stats": {
            "transcript_words": transcript_words,
            "summary_words": summary_words,
            "compression_ratio_pct": comp_ratio
        },
        "performance": {
            "time_stt_sec": round(time_stt, 4),
            "time_summary_sec": round(time_summary, 4),
            "time_translation_sec": round(time_trans, 4),
            "time_tts_sec": round(time_tts, 4),
            "time_srt_sec": round(time_srt, 4),
            "total_time_sec": round(total_time, 4)
        }
    }

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n==================================================")
    print(f"   MULTIMODAL PIPELINE COMPLETED IN {total_time:.2f}s ({lang.upper()})")
    print(f"==================================================")
    print(f"[MULTIMODAL PERFORMANCE BREAKDOWN]")
    print(f"  - Whisper STT          : {time_stt:.4f}s")
    print(f"  - Text Summarizer      : {time_summary:.4f}s")
    print(f"  - Translation ({lang})   : {time_trans:.4f}s")
    print(f"  - TTS Narration ({lang}) : {time_tts:.4f}s ({tts_duration:.2f}s audio)")
    print(f"  - Subtitles SRT        : {time_srt:.4f}s")
    print(f"  - TOTAL MULTIMODAL     : {total_time:.4f}s")
    print(f"==================================================")

    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = run_multimodal_pipeline(sys.argv[1])
        print("\nPipeline Result:", json.dumps(res, indent=2))
