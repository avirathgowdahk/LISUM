"""
===============================================================================
Module: Real-time Progress Streaming Route (routes/streaming.py)
===============================================================================
1. Purpose:
   Provides Server-Sent Events (SSE) HTTP endpoint for streaming real-time pipeline
   progress events to the user's web browser.
2. Endpoint:
   GET /api/stream/<job_id>
3. Payload:
   data: {"stage": "transcription", "progress": 45, "message": "Transcribing lecture audio..."}
===============================================================================
"""

import time
import json
from flask import Blueprint, Response, jsonify

streaming_bp = Blueprint('streaming', __name__)

# Global job dictionary reference set by app.py
JOBS_STORE = {}

def init_jobs_store(jobs_dict):
    global JOBS_STORE
    JOBS_STORE = jobs_dict

@streaming_bp.route('/api/stream/<job_id>')
def stream_job_progress(job_id: str):
    """Server-Sent Events (SSE) streaming endpoint emitting live processing status."""
    def generate_events():
        last_progress = -1
        last_stage = None
        last_message = None

        while True:
            job = JOBS_STORE.get(job_id)
            if not job:
                err_data = json.dumps({"stage": "FAILED", "progress": 0, "message": "Job not found"})
                yield f"data: {err_data}\n\n"
                break

            stage = job.get("stage", "PROCESSING")
            progress = job.get("progress", 0)
            message = job.get("message", "Processing...")
            error = job.get("error")

            # Emit event if stage, progress, or message changed
            if progress != last_progress or stage != last_stage or message != last_message or stage in ["COMPLETED", "FAILED"]:
                last_progress = progress
                last_stage = stage
                last_message = message

                event_payload = json.dumps({
                    "job_id": job_id,
                    "stage": stage,
                    "progress": progress,
                    "message": message,
                    "error": error,
                    "elapsed_sec": job.get("elapsed_sec", 0.0)
                })

                yield f"data: {event_payload}\n\n"

            if stage in ["COMPLETED", "FAILED"]:
                break

            time.sleep(0.25)

    return Response(generate_events(), mimetype='text/event-stream')
