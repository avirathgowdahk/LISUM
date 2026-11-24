import os
import sys
import subprocess
import json
import time
import hashlib
import torch
from typing import List, Tuple, Dict, Any, Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import (
    WHISPER_MODEL, WHISPER_MODEL_SIZE, WHISPER_BACKEND, WHISPER_COMPUTE_TYPE, DEVICE
)

# Global Singleton Model Storage
_GLOBAL_WHISPER_MODEL = None
_GLOBAL_WHISPER_KEY = None
_GLOBAL_WHISPER_BACKEND = None

def get_whisper_model(
    model_name: str = None,
    backend: str = None,
    device: str = None,
    compute_type: str = None
) -> Tuple[Any, str, str, str]:
    """
    Lazy-loads and returns a singleton Whisper model instance.
    Model is loaded ONCE globally and reused across all jobs.
    Auto-detects CUDA vs CPU and prefers faster-whisper (CTranslate2).
    """
    global _GLOBAL_WHISPER_MODEL, _GLOBAL_WHISPER_KEY, _GLOBAL_WHISPER_BACKEND

    model_name = model_name or WHISPER_MODEL or WHISPER_MODEL_SIZE or "base"
    backend = (backend or WHISPER_BACKEND or "faster-whisper").lower()
    
    # Auto-detect CUDA
    is_cuda = torch.cuda.is_available()
    target_device = "cuda" if is_cuda else "cpu"
    
    if compute_type is None or compute_type == "auto":
        compute_type = "float16" if is_cuda else "int8"

    key = f"{backend}_{model_name}_{target_device}_{compute_type}"

    if _GLOBAL_WHISPER_MODEL is not None and (_GLOBAL_WHISPER_KEY == key or _GLOBAL_WHISPER_KEY.startswith(f"{backend}_{model_name}_{target_device}")):
        return _GLOBAL_WHISPER_MODEL, _GLOBAL_WHISPER_BACKEND, target_device, compute_type

    print(f"\n==================================================")
    print(f"[Whisper] Device: {'GPU' if is_cuda else 'CPU'}")
    print(f"[Whisper] Compute type: {compute_type}")
    print(f"[Whisper] Model: {model_name} (Backend: {backend})")
    print(f"==================================================")

    model_obj = None
    active_backend = None

    # Try faster-whisper first if backend is auto or faster-whisper
    if backend in ["auto", "faster-whisper", "faster_whisper", "ctranslate2"]:
        try:
            from faster_whisper import WhisperModel
            try:
                model_obj = WhisperModel(
                    model_name,
                    device=target_device,
                    compute_type=compute_type,
                    cpu_threads=min(8, os.cpu_count() or 4),
                    num_workers=1,
                    download_root=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                )
            except Exception as e_ct:
                if not is_cuda and compute_type == "int8":
                    print(f"[Whisper Warning] int8 compute_type failed ({e_ct}). Trying float32...")
                    compute_type = "float32"
                    model_obj = WhisperModel(
                        model_name,
                        device=target_device,
                        compute_type="float32",
                        cpu_threads=min(8, os.cpu_count() or 4),
                        num_workers=1,
                        download_root=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                    )
                else:
                    raise e_ct

            active_backend = "faster-whisper"
            print(f"[Whisper] Model loaded successfully: faster-whisper ({model_name})")
        except Exception as e:
            print(f"[Whisper Warning] faster-whisper initialization failed ({e}). Falling back to OpenAI whisper...")

    # Fallback to OpenAI Whisper
    if model_obj is None:
        import whisper
        print(f"[Whisper] Loading OpenAI whisper model '{model_name}' on {target_device}...")
        model_obj = whisper.load_model(model_name, device=target_device)
        active_backend = "openai-whisper"
        print(f"[Whisper] Model loaded successfully: OpenAI whisper ({model_name})")

    _GLOBAL_WHISPER_MODEL = model_obj
    _GLOBAL_WHISPER_KEY = key
    _GLOBAL_WHISPER_BACKEND = active_backend

    return _GLOBAL_WHISPER_MODEL, _GLOBAL_WHISPER_BACKEND, target_device, compute_type


def warmup_whisper_model():
    """Pre-loads singleton Whisper model at startup to eliminate first-request latency."""
    try:
        model, backend, dev, ctype = get_whisper_model()
        print(f"[Whisper] Device: {'GPU' if dev == 'cuda' else 'CPU'}")
        print(f"[Whisper] Compute type: {ctype}")
        print(f"[Whisper] Model loaded successfully: {backend} ({WHISPER_MODEL})")
    except Exception as e:
        print(f"[Whisper Warning] Pre-load failed: {e}")


def get_video_file_hash(file_path: str) -> str:
    """Calculates deterministic SHA256 content hash based on file size & key byte chunks."""
    if not os.path.exists(file_path):
        return "unknown_hash"
    file_size = os.path.getsize(file_path)
    hasher = hashlib.sha256()
    hasher.update(f"size_{file_size}".encode('utf-8'))
    try:
        with open(file_path, 'rb') as f:
            hasher.update(f.read(512 * 1024))
            if file_size > 1024 * 1024:
                f.seek(-512 * 1024, os.SEEK_END)
                hasher.update(f.read(512 * 1024))
    except Exception:
        hasher.update(os.path.basename(file_path).encode('utf-8'))
    return hasher.hexdigest()[:12]


def check_video_has_audio(video_path: str) -> bool:
    """Uses ffprobe to detect if the input video file contains an audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return bool(res.stdout and res.stdout.strip())


def get_audio_duration(audio_path: str) -> float:
    """Uses ffprobe to get exact duration in seconds of an audio file."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio_from_video(video_path: str, output_wav_path: str) -> str:
    """Extracts mono 16kHz WAV audio from video file using FFmpeg (skips if cached)."""
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 1000:
        return output_wav_path

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")
    return output_wav_path


def extract_segments_audio_from_video(
    video_path: str,
    segments: List[Tuple[float, float]],
    output_wav_path: str
) -> str:
    """
    Extracts audio clips for selected timestamps directly from source video
    and concatenates them into a single 16kHz mono WAV file in a SINGLE fast FFmpeg pass.
    """
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    if not segments:
        return extract_audio_from_video(video_path, output_wav_path)

    # Merge contiguous/overlapping segments to minimize FFmpeg filter chains
    merged_segs = []
    for s, e in sorted(segments, key=lambda x: x[0]):
        if merged_segs and s <= merged_segs[-1][1] + 0.15:
            merged_segs[-1] = (merged_segs[-1][0], max(merged_segs[-1][1], e))
        else:
            merged_segs.append((s, e))

    if len(merged_segs) == 1:
        s_start, s_end = merged_segs[0]
        if s_start <= 0.5:
            # Single continuous clip starting near 0: fast trim using -ss / -to
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{s_start:.3f}",
                "-to", f"{s_end:.3f}",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_wav_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 1000:
                return output_wav_path

    # Build single-pass FFmpeg filter_complex command for merged segments
    filter_chains = []
    concat_inputs = []
    for idx, (start_sec, end_sec) in enumerate(merged_segs):
        dur = max(0.2, end_sec - start_sec)
        filter_chains.append(f"[0:a]atrim=start={start_sec:.3f}:end={end_sec:.3f},asetpts=PTS-STARTPTS[a{idx}]")
        concat_inputs.append(f"[a{idx}]")

    concat_filter = f"{''.join(concat_inputs)}concat=n={len(merged_segs)}:v=0:a=1[outa]"
    full_filter = f"{';'.join(filter_chains)};{concat_filter}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter_complex", full_filter,
        "-map", "[outa]",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
        print(f"[STT Warning] Single-pass filter_complex audio extraction failed. Fallback to full audio extraction...")
        return extract_audio_from_video(video_path, output_wav_path)

    return output_wav_path


def transcribe_audio(
    video_path: str,
    output_dir: str = "outputs/multimodal",
    model_size: str = None,
    selected_segments: List[Tuple[float, float]] = None,
    original_video_path: str = None,
    force_recompute: bool = False,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Transcribes audio track using lazy-loaded singleton Whisper model.
    Utilizes caching, faster-whisper CTranslate2, VAD filtering, direct audio reuse,
    and real segment-by-segment progress reporting with speed and ETA.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("cache/transcripts", exist_ok=True)
    os.makedirs("cache/audio", exist_ok=True)

    model_size = model_size or WHISPER_MODEL or WHISPER_MODEL_SIZE or "base"
    source_for_audio = original_video_path if (original_video_path and os.path.exists(original_video_path)) else video_path

    # Build content-based cache key using file SHA256 hash & model size
    content_hash = get_video_file_hash(source_for_audio)
    file_base = os.path.splitext(os.path.basename(source_for_audio))[0]
    seg_str = "_".join([f"{s:.1f}-{e:.1f}" for s, e in selected_segments]) if selected_segments else "full"

    json_path = os.path.join(output_dir, f"transcript_{content_hash}_{model_size}_{seg_str}.json")
    txt_path = os.path.join(output_dir, f"transcript_{content_hash}_{model_size}_{seg_str}.txt")
    cache_json_path = os.path.join("cache", "transcripts", f"transcript_{content_hash}_{model_size}_{seg_str}.json")
    cache_txt_path = os.path.join("cache", "transcripts", f"transcript_{content_hash}_{model_size}_{seg_str}.txt")
    wav_path = os.path.join("cache", "audio", f"audio_{content_hash}_{seg_str}.wav")

    # Step 1: Transcription Cache Check (Requirement 5)
    for c_json, c_txt in [(json_path, txt_path), (cache_json_path, cache_txt_path)]:
        if not force_recompute and os.path.exists(c_json) and os.path.exists(c_txt):
            try:
                with open(c_json, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data.get("transcript_text"):
                    print(f"[WHISPER CACHE HIT] Using precomputed transcript for '{file_base}' (Hash: {content_hash})")
                    if progress_callback:
                        progress_callback("TRANSCRIBING", 50, "Loaded transcript from cache instantly!")
                    return {
                        "success": True,
                        "stage": "speech_to_text",
                        "audio_duration_sec": cached_data.get("audio_duration_sec", 0.0),
                        "language": cached_data.get("language", "en"),
                        "transcript_text": cached_data.get("transcript_text"),
                        "segments": cached_data.get("segments", []),
                        "txt_file": c_txt,
                        "json_file": c_json,
                        "cached": True
                    }
            except Exception as ce:
                print(f"[Whisper Warning] Failed loading transcript cache: {ce}")

    # Step 2: Validate Audio Stream / File
    is_audio_file = source_for_audio.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"))
    if not is_audio_file:
        print(f"[STT] Checking audio stream availability for '{source_for_audio}'...")
        if not check_video_has_audio(source_for_audio):
            print(f"[STT Error] No audio track detected in '{source_for_audio}'.")
            return {
                "success": False,
                "stage": "audio_extraction",
                "error": "No audio track detected in the video file.",
                "transcript_text": None
            }

    # Step 3: Extract Audio Track (TRANSCRIBE ONLY THE EXTRACTED AUDIO - Requirement 2 & 7)
    # If source is already a 16kHz WAV file and no specific clip segments are requested, use it directly!
    if is_audio_file and source_for_audio.lower().endswith(".wav") and not (selected_segments and len(selected_segments) > 0):
        wav_path = source_for_audio
        print(f"[STT] Transcribing extracted audio directly without re-encoding: '{wav_path}'")
    else:
        t_extract_start = time.time()
        print(f"[STT] Audio extraction path: '{wav_path}'...")
        try:
            if selected_segments and len(selected_segments) > 0:
                print(f"[STT] Extracting summary segment audio ({len(selected_segments)} clips)...")
                extract_segments_audio_from_video(source_for_audio, selected_segments, wav_path)
            else:
                extract_audio_from_video(source_for_audio, wav_path)
        except Exception as e:
            print(f"[STT Error] Audio extraction failed: {e}")
            return {
                "success": False,
                "stage": "audio_extraction",
                "error": f"Audio extraction failed: {str(e)}",
                "transcript_text": None
            }
        extract_time = time.time() - t_extract_start
        print(f"[STT] Audio extracted successfully in {extract_time:.2f}s.")

    audio_dur = get_audio_duration(wav_path)

    # Step 4: Lazy Load / Fetch Singleton Model (Requirement 1 & 6)
    model, backend_used, device_used, compute_type_used = get_whisper_model(model_name=model_size)

    # Step 5: Run Inference with Real-Time Progress Updates (Requirements 1, 3, 4, 8, 9, 11)
    t_stt_start = time.time()
    formatted_segments = []
    transcript_text = ""
    detected_lang = "en"

    print(f"[25%] Whisper transcription started")
    print(f"[Whisper] Audio duration: {audio_dur/60:.1f} minutes ({audio_dur:.1f} seconds)")
    print(f"[Whisper] Device: {'GPU' if device_used == 'cuda' else 'CPU'}")
    print(f"[Whisper] Compute type: {compute_type_used}")
    print(f"[Whisper] Model: {model_size}")
    print(f"[Whisper] VAD enabled: True")
    print(f"[Whisper] Word timestamps enabled: True")

    if progress_callback:
        progress_callback("TRANSCRIBING", 25, "Whisper transcription started...")

    try:
        if backend_used == "faster-whisper":
            beam_sz = 1  # Greedy decoding: 4-5x faster on CPU, robust against hallucinations
            segments_gen, info = model.transcribe(
                wav_path,
                word_timestamps=True,
                beam_size=beam_sz,
                best_of=1,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300)
            )
            detected_lang = info.language or "en"

            raw_segs = []
            last_report_time = time.time()
            last_reported_pct = 25

            # Iterate over generator segment by segment for live progress reporting
            for seg in segments_gen:
                raw_segs.append(seg)
                now = time.time()

                processed_sec = min(audio_dur, max(0.0, float(seg.end)))
                fraction = processed_sec / max(0.1, audio_dur)
                audio_pct = min(100, int(fraction * 100))

                # Pipeline scale: maps 25% -> 49%
                whisper_ui_pct = min(49, 25 + int(fraction * 24))

                elapsed = max(0.1, now - t_stt_start)
                speed = processed_sec / elapsed
                if speed > 0.05:
                    rem_sec = max(0, (audio_dur - processed_sec) / speed)
                    if rem_sec >= 60:
                        rem_str = f"{int(rem_sec // 60)}m {int(rem_sec % 60):02d}s"
                    else:
                        rem_str = f"{int(rem_sec)}s"
                else:
                    rem_str = "calculating..."

                speed_str = f"{speed:.1f}× realtime" if speed >= 0.1 else f"{speed:.2f}× realtime"
                status_msg = (
                    f"Transcribing lecture with Whisper...\n"
                    f"Processed: {audio_pct}% | Speed: {speed_str} | Estimated remaining: {rem_str}"
                )

                if (now - last_report_time >= 0.7) or (whisper_ui_pct > last_reported_pct):
                    last_report_time = now
                    last_reported_pct = whisper_ui_pct
                    print(f"[Whisper] Transcription progress: {audio_pct}% (speed: {speed_str}, est. remaining: {rem_str})")
                    if progress_callback:
                        progress_callback("TRANSCRIBING", whisper_ui_pct, status_msg)

            if progress_callback:
                progress_callback("TRANSCRIBING", 50, "Finalizing transcript timestamps & speech segments...")

            seg_texts = []
            for seg in raw_segs:
                s_text = seg.text.strip()
                if s_text:
                    seg_texts.append(s_text)

                words_list = []
                if seg.words:
                    for w in seg.words:
                        w_t = w.word.strip()
                        if w_t:
                            words_list.append({
                                "word": w_t,
                                "start": round(float(w.start), 3),
                                "end": round(float(w.end), 3)
                            })

                formatted_segments.append({
                    "id": seg.id,
                    "start": round(float(seg.start), 2),
                    "end": round(float(seg.end), 2),
                    "text": s_text,
                    "words": words_list
                })

            transcript_text = " ".join(seg_texts).strip()

        else:
            # OpenAI Whisper Fallback
            print(f"[STT] Transcribing via OpenAI whisper ({model_size}, {device_used})...")
            fp16_flag = (device_used == "cuda")
            try:
                res = model.transcribe(wav_path, fp16=fp16_flag, word_timestamps=True)
            except Exception:
                res = model.transcribe(wav_path, fp16=fp16_flag)

            transcript_text = res.get("text", "").strip()
            detected_lang = res.get("language", "en")
            raw_segments = res.get("segments", [])

            for seg in raw_segments:
                words_list = []
                for w in seg.get("words", []):
                    w_text = w.get("word", "").strip()
                    if w_text:
                        words_list.append({
                            "word": w_text,
                            "start": round(float(w.get("start", 0)), 3),
                            "end": round(float(w.get("end", 0)), 3)
                        })

                formatted_segments.append({
                    "id": seg.get("id"),
                    "start": round(float(seg.get("start", 0)), 2),
                    "end": round(float(seg.get("end", 0)), 2),
                    "text": seg.get("text", "").strip(),
                    "words": words_list
                })

            if progress_callback:
                progress_callback("TRANSCRIBING", 50, "Finalizing transcript timestamps & speech segments...")

    except Exception as e:
        print(f"[STT Error] Whisper transcription failed: {e}")
        return {
            "success": False,
            "stage": "speech_to_text",
            "error": f"Whisper transcription failed: {str(e)}",
            "transcript_text": None
        }

    processing_time = time.time() - t_stt_start
    rtf = processing_time / max(0.01, audio_dur)

    print(f"\n==================================================")
    print(f"[WHISPER PERFORMANCE METRICS]")
    print(f"Backend       : {backend_used}")
    print(f"Model         : {model_size}")
    print(f"Device        : {device_used.upper()}")
    print(f"Compute Type  : {compute_type_used}")
    print(f"Audio Duration: {audio_dur:.2f}s ({audio_dur/60:.2f}m)")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"RTF           : {rtf:.4f} (Lower is better)")
    print(f"Language      : {detected_lang}")
    print(f"Transcript    : {len(transcript_text)} characters")
    print(f"==================================================")

    if not transcript_text:
        print(f"[STT Warning] No spoken words could be recognized in audio.")
        return {
            "success": False,
            "stage": "speech_to_text",
            "error": "No speech could be recognized in the video audio track.",
            "transcript_text": None
        }

    # Save transcript files to both output_dir and cache/transcripts
    transcript_json_data = {
        "video_source": os.path.basename(video_path),
        "audio_duration_sec": audio_dur,
        "language": detected_lang,
        "transcript_text": transcript_text,
        "segments": formatted_segments
    }

    for t_txt, t_json in [(txt_path, json_path), (cache_txt_path, cache_json_path)]:
        try:
            with open(t_txt, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            with open(t_json, "w", encoding="utf-8") as f:
                json.dump(transcript_json_data, f, indent=2, ensure_ascii=False)
        except Exception as e_s:
            print(f"[Whisper Warning] Failed writing transcript cache to {t_json}: {e_s}")

    return {
        "success": True,
        "stage": "speech_to_text",
        "audio_duration_sec": audio_dur,
        "language": detected_lang,
        "transcript_text": transcript_text,
        "segments": formatted_segments,
        "txt_file": txt_path,
        "json_file": json_path,
        "rtf": round(rtf, 4),
        "processing_time_sec": round(processing_time, 2)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = transcribe_audio(sys.argv[1])
        print("\nSTT Output:", res)
    else:
        print("Usage: python multimodal/speech_to_text.py <path_to_video>")
