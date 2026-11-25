"""
===============================================================================
Module: Results Routes (routes/results.py)
===============================================================================
1. Purpose:
   Provides JSON endpoints for structured transcripts, summaries, timestamp clips,
   and file download endpoints (MP4 video, TXT transcript, JSON summary, SRT subtitles).
2. Endpoints:
   GET /api/transcript/<job_id>
   GET /api/summary/<job_id>
   GET /api/clips/<job_id>
   GET /api/download/<job_id>/<file_type>
===============================================================================
"""

import os
from flask import Blueprint, jsonify, send_from_directory, current_app

results_bp = Blueprint('results', __name__)

JOBS_STORE = {}

def init_jobs_store(jobs_dict):
    global JOBS_STORE
    JOBS_STORE = jobs_dict

@results_bp.route('/api/transcript/<job_id>')
def get_transcript_json(job_id: str):
    """Returns machine-readable transcript JSON."""
    job = JOBS_STORE.get(job_id)
    if not job or job.get("stage") != "COMPLETED":
        return jsonify({"error": "Job not found or incomplete"}), 404

    res = job["result"]
    return jsonify({
        "lecture_id": res["lecture_id"],
        "language": res["target_language"],
        "duration": res["original_duration"],
        "segments": res["timestamped_sentences"]
    })

@results_bp.route('/api/summary/<job_id>')
def get_summary_json(job_id: str):
    """Returns machine-readable summary JSON with timestamp alignment mapping."""
    job = JOBS_STORE.get(job_id)
    if not job or job.get("stage") != "COMPLETED":
        return jsonify({"error": "Job not found or incomplete"}), 404

    res = job["result"]
    return jsonify({
        "lecture_id": res["lecture_id"],
        "summary": res.get("translated_summary_text") or res["summary_text"],
        "statements": res["summary_statements"],
        "topic_bullets": res["topic_bullets"]
    })

@results_bp.route('/api/clips/<job_id>')
def get_clips_json(job_id: str):
    """Returns selected video clips metadata."""
    job = JOBS_STORE.get(job_id)
    if not job or job.get("stage") != "COMPLETED":
        return jsonify({"error": "Job not found or incomplete"}), 404

    res = job["result"]
    return jsonify({
        "lecture_id": res["lecture_id"],
        "selected_clips": res["selected_clips"]
    })

@results_bp.route('/api/download/<job_id>/<file_type>')
def download_output_file(job_id: str, file_type: str):
    """Serves file downloads (video, transcript, summary, subtitles)."""
    job = JOBS_STORE.get(job_id)
    if not job or job.get("stage") != "COMPLETED":
        return jsonify({"error": "Job not found or incomplete"}), 404

    res = job["result"]
    out_dir = current_app.config['OUTPUT_FOLDER']
    os.makedirs(out_dir, exist_ok=True)

    if file_type == 'video':
        path = res["output_summary_video"]
    elif file_type == 'transcript':
        path = os.path.join(out_dir, f"{job_id}_transcript.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(res["raw_transcript"])
    elif file_type == 'summary':
        path = os.path.join(out_dir, f"{job_id}_summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(res["translated_summary_text"])
    elif file_type == 'subtitles':
        path = res["srt_subtitle_file"]
    elif file_type == 'json':
        path = os.path.join(out_dir, f"{job_id}_full_result.json")
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    else:
        return jsonify({"error": f"Invalid download file_type '{file_type}'"}), 400

    if not path or not os.path.exists(path):
        return jsonify({"error": f"File '{file_type}' not found on server."}), 404

    directory = os.path.dirname(os.path.abspath(path))
    filename = os.path.basename(path)
    return send_from_directory(directory, filename, as_attachment=True)
