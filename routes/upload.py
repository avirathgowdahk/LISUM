"""
===============================================================================
Module: Upload Routes (routes/upload.py)
===============================================================================
1. Purpose:
   Handles video file upload and public video URL ingestion via yt-dlp.
2. Endpoints:
   POST /api/upload
   POST /api/url
===============================================================================
"""

import os
import uuid
from flask import Blueprint, request, jsonify, url_for, current_app
from werkzeug.utils import secure_filename
from services.video_processor import validate_and_inspect_video
from preprocessing.url_downloader import fetch_video_from_url
from config import ALLOWED_EXTENSIONS

upload_bp = Blueprint('upload', __name__)

def is_allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@upload_bp.route('/api/upload', methods=['POST'])
def handle_upload():
    """Validates and uploads lecture video file."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not is_allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file format. Allowed: {list(ALLOWED_EXTENSIONS)}"}), 400

    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{secure_filename(file.filename)}"
    upload_folder = current_app.config['UPLOAD_FOLDER']
    video_path = os.path.join(upload_folder, filename)
    file.save(video_path)

    try:
        meta = validate_and_inspect_video(video_path)
        rel_path = os.path.relpath(video_path, upload_folder).replace("\\", "/")

        return jsonify({
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "video_path": video_path,
            "video_url": f"/uploads/{rel_path}",
            "metadata": meta
        })
    except Exception as e:
        return jsonify({"error": f"Video validation failed: {str(e)}"}), 400

@upload_bp.route('/api/url', methods=['POST'])
def handle_url():
    """Ingests public video URL via yt-dlp."""
    data = request.json or {}
    url = data.get('url')

    if not url:
        return jsonify({"error": "Valid video URL is required"}), 400

    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        res = fetch_video_from_url(url, output_base_dir=upload_folder)
        rel_path = os.path.relpath(res["local_path"], upload_folder).replace("\\", "/")
        res["video_url"] = f"/uploads/{rel_path}"
        res["success"] = True
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": f"URL download failed: {str(e)}"}), 400
