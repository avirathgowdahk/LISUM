"""
===============================================================================
Module: Processing Routes (routes/processing.py)
===============================================================================
1. Purpose:
   Handles request triggers for video summarization tasks.
2. Endpoints:
   POST /api/summarize
   GET /api/status/<job_id>
===============================================================================
"""

import os
import uuid
import threading
import time
from flask import Blueprint, request, jsonify, current_app
from pipeline_engine import run_lecture_processing_pipeline

processing_bp = Blueprint('processing', __name__)

JOBS_STORE = {}

def init_jobs_store(jobs_dict):
    global JOBS_STORE
    JOBS_STORE = jobs_dict

def process_background_job(
    job_id: str,
    video_path: str,
    target_duration_sec: float,
    summary_type: str,
    target_language: str,
    caption_option: str
):
    """Worker thread running master pipeline engine."""
    job_start = time.time()
    try:
        def progress_cb(stage, progress_pct, msg):
            if job_id in JOBS_STORE:
                JOBS_STORE[job_id].update({
                    "stage": stage,
                    "progress": progress_pct,
                    "message": msg,
                    "elapsed_sec": round(time.time() - job_start, 1)
                })

        res = run_lecture_processing_pipeline(
            video_path=video_path,
            target_duration_sec=target_duration_sec,
            summary_type=summary_type,
            target_language=target_language,
            caption_option=caption_option,
            job_id=job_id,
            progress_callback=progress_cb
        )

        total_elapsed = round(time.time() - job_start, 1)
        JOBS_STORE[job_id].update({
            "stage": "COMPLETED",
            "progress": 100,
            "message": f"Summarization completed in {total_elapsed}s!",
            "elapsed_sec": total_elapsed,
            "result": res
        })
    except Exception as e:
        JOBS_STORE[job_id].update({
            "stage": "FAILED",
            "progress": 0,
            "message": f"Processing failed: {str(e)}",
            "error": str(e)
        })

@processing_bp.route('/api/summarize', methods=['POST'])
def handle_summarize():
    """Starts background video summarization task."""
    data = request.json or {}
    video_path = data.get('video_path')

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Valid video_path is required."}), 400

    target_duration_sec = float(data.get('target_duration_sec', 120.0))
    summary_type = data.get('summary_type', 'hybrid').lower()
    target_language = data.get('language', 'en').lower()
    caption_option = data.get('caption_option', 'original').lower()

    job_id = str(uuid.uuid4())[:8]

    JOBS_STORE[job_id] = {
        "job_id": job_id,
        "stage": "VALIDATING",
        "progress": 5,
        "message": "Starting lecture video processing...",
        "video_path": video_path,
        "target_duration_sec": target_duration_sec,
        "summary_type": summary_type,
        "language": target_language,
        "caption_option": caption_option,
        "result": None
    }

    thread = threading.Thread(
        target=process_background_job,
        args=(job_id, video_path, target_duration_sec, summary_type, target_language, caption_option)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "started",
        "message": "Lecture summarization task started"
    })

@processing_bp.route('/api/status/<job_id>')
def get_job_status(job_id: str):
    """Returns status of background job."""
    job = JOBS_STORE.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)
