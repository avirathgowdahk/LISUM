"""
===============================================================================
Module: Main Flask Web Application (app.py)
===============================================================================
1. Purpose:
   Serves as the central Web Server & API Gateway for the AI Lecture Video Processing System.
2. Responsibilities:
   - Registers route blueprints (upload, processing, results, streaming)
   - Serves frontend academic UI templates and static assets
   - Provides global job state tracking and Server-Sent Events (SSE) progress streaming
===============================================================================
"""

import os
import sys
import torch
from flask import Flask, render_template, redirect, url_for, send_from_directory

from config import (
    UPLOAD_DIR, OUTPUT_DIR, SUPPORTED_LANGUAGES, WHISPER_MODEL, PROCESSING_MODE
)
from routes.upload import upload_bp
from routes.processing import processing_bp, init_jobs_store as init_proc_jobs
from routes.results import results_bp, init_jobs_store as init_res_jobs
from routes.streaming import streaming_bp, init_jobs_store as init_stream_jobs

from multimodal.speech_to_text import warmup_whisper_model
from services.embedding_service import warmup_embedding_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['OUTPUT_FOLDER'] = OUTPUT_DIR
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # Max 2GB upload limit
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global job dictionary
JOBS = {}

init_proc_jobs(JOBS)
init_res_jobs(JOBS)
init_stream_jobs(JOBS)

app.register_blueprint(upload_bp)
app.register_blueprint(processing_bp)
app.register_blueprint(results_bp)
app.register_blueprint(streaming_bp)

# Startup Device Logging & Model Warmup
is_cuda = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU Execution Engine"
print(f"\n==================================================")
print(f"[SYSTEM INITIALIZATION]")
print(f"CUDA Available: {is_cuda}")
print(f"Engine Device : {device_name}")
print(f"Processing    : {PROCESSING_MODE}")
print(f"Whisper Model : {WHISPER_MODEL}")
print(f"==================================================")

try:
    print("[WARMUP] Initializing model engines in background...")
    warmup_whisper_model()
    warmup_embedding_model()
except Exception as e_w:
    print(f"[WARMUP Warning] Model warmup failed: {e_w}")
print(f"==================================================\n")

@app.template_filter('timefmt')
def timefmt(seconds):
    """Formats floating point seconds into clean MM:SS format."""
    try:
        if seconds is None:
            return "00:00"
        if hasattr(seconds, '_fail_with_undefined_error'):
            return "00:00"
        sec = float(seconds)
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"
    except Exception:
        return "00:00"


@app.template_filter('strip')
def strip_filter(val):
    """Trims whitespace from strings safely."""
    if val is None:
        return ""
    return str(val).strip()

@app.route('/')
def index():
    """Renders main academic upload and configuration page."""
    return render_template('index.html', languages=SUPPORTED_LANGUAGES)

@app.route('/result/<job_id>')
def view_result(job_id: str):
    """Renders full 7-section academic result dashboard in exact logical section order."""
    job = JOBS.get(job_id)
    if not job or job.get("stage") != "COMPLETED":
        return redirect(url_for('index'))

    res = job["result"]
    original_filename = os.path.basename(res["input_video"])
    summary_filename = os.path.basename(res["output_summary_video"])
    rel_input_path = os.path.relpath(res["input_video"], UPLOAD_DIR).replace("\\", "/")

    lang_code = res.get("target_language", "en")
    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())
    caption_opt = res.get("caption_option", "original")
    caption_display = "Original Captions" if caption_opt == "original" else ("Translated Captions" if caption_opt == "translated" else "None")

    video_meta = res.get("metadata", {})
    res_str = video_meta.get("resolution", "640 × 358")
    if not res_str or res_str == "unknown":
        w = video_meta.get("width", 640)
        h = video_meta.get("height", 358)
        res_str = f"{w} × {h}"

    return render_template(
        'result.html',
        job_id=job_id,
        original_filename=original_filename,
        original_duration=res["original_duration"],
        target_duration_sec=res["target_duration_sec"],
        summary_duration=res["summary_duration"],
        target_language=lang_name,
        target_language_code=lang_code,
        compression_percent=res["compression_percent"],
        caption_mode_display=caption_display,
        video_resolution=res_str,
        original_video_url=f"/uploads/{rel_input_path}",
        summary_video_url=url_for('serve_output', filename=summary_filename),
        text_summary=res.get("translated_summary_text") or res.get("summary_text", ""),
        structured_sections=res.get("structured_sections", []),
        summary_statements=res.get("summary_statements", []),
        timestamped_sentences=res.get("timestamped_sentences", []),
        topic_timeline=res.get("topic_timeline", []) or res.get("topic_bullets", []),
        topic_bullets=res.get("topic_bullets", []),
        selected_clips=res.get("selected_clips", [])
    )

@app.route('/uploads/<path:filename>')
def serve_upload(filename: str):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<path:filename>')
def serve_output(filename: str):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("cache/audio", exist_ok=True)
    os.makedirs("cache/transcripts", exist_ok=True)
    os.makedirs("outputs/subtitles", exist_ok=True)

    print("Starting Flask Web Application on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
