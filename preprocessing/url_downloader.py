import os
import sys
import re
import hashlib
import urllib.parse
import ipaddress
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import yt_dlp
from preprocessing.video_reader import get_video_metadata

try:
    from config import MAX_VIDEO_SIZE_MB, MAX_LECTURE_DURATION_SECONDS
except ImportError:
    MAX_VIDEO_SIZE_MB = 2000.0
    MAX_LECTURE_DURATION_SECONDS = 3 * 3600

# Tracking parameters to strip from URLs
TRACKING_PARAMS = {'si', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'feature', 'fbclid', 'gclid'}
MEDIA_EXTENSIONS = {'.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.m4v'}
TEMP_EXTENSIONS = {'.part', '.ytdl', '.json', '.info', '.tmp', '.temp'}

def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable hours/minutes/seconds string."""
    total_sec = int(round(seconds))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0:
        parts.append(f"{m}m")
    if s > 0 or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)

def cleanup_temp_files(directory: str):
    """Safely removes temporary and partial download files to prevent disk waste."""
    if not os.path.exists(directory):
        return
    for item in os.listdir(directory):
        if any(item.endswith(te) for te in TEMP_EXTENSIONS):
            try:
                os.remove(os.path.join(directory, item))
            except Exception:
                pass

def normalize_url(url: str) -> str:
    """Strips tracking query parameters and canonicalizes YouTube URLs to standard watch format."""
    if not url:
        return ""
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    netloc = parsed.netloc.lower()
    path = parsed.path

    # Canonicalize YouTube video IDs across different domain/path variants
    video_id = None
    if netloc in ('youtu.be', 'www.youtu.be'):
        video_id = path.lstrip('/').split('/')[0]
    elif netloc in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if path.startswith('/shorts/'):
            video_id = path.split('/shorts/')[1].split('/')[0]
        elif path == '/watch':
            query_dict = urllib.parse.parse_qs(parsed.query)
            if 'v' in query_dict and query_dict['v']:
                video_id = query_dict['v'][0]

    if video_id and re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return f"https://www.youtube.com/watch?v={video_id}"

    # Parse query string and remove tracking keys
    query_dict = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    filtered_query = {k: v for k, v in query_dict.items() if k.lower() not in TRACKING_PARAMS}
    new_query = urllib.parse.urlencode(filtered_query, doseq=True)

    normalized = urllib.parse.urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        new_query,
        ""  # strip fragment
    ))
    return normalized

def validate_public_url(url: str) -> str:
    """
    Validates URL syntax and enforces strict SSRF / private IP blocking rules.
    Returns normalized URL if valid, or raises ValueError with a user-friendly error message.
    """
    if not url or not isinstance(url, str):
        raise ValueError("Please enter a valid video URL.")

    url = url.strip()
    if url.startswith("file://") or url.startswith("/") or re.match(r'^[a-zA-Z]:\\', url):
        raise ValueError("Local filesystem paths are not allowed in the URL field.")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("Only public HTTP and HTTPS video URLs are supported.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Please enter a valid public video URL.")

    hostname_lower = hostname.lower()
    if hostname_lower in ('localhost', '127.0.0.1', '0.0.0.0', '::1') or hostname_lower.endswith('.local'):
        raise ValueError("Localhost and internal network URLs are not allowed.")

    # Check IP address targets
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified:
            raise ValueError("Access to private IP addresses is blocked.")
    except ValueError:
        # Standard domain name
        pass

    return normalize_url(url)

def validate_media_file(filepath: str) -> Dict[str, Any]:
    """
    Verifies downloaded file exists, is a non-empty regular file, and can be parsed for video stream & duration.
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise RuntimeError(f"Media file does not exist: {abs_path}")
    if not os.path.isfile(abs_path):
        raise RuntimeError(f"Media path is not a regular file: {abs_path}")
    
    file_size = os.path.getsize(abs_path)
    if file_size <= 0:
        raise RuntimeError(f"Downloaded media file is empty (0 bytes): {abs_path}")

    try:
        meta = get_video_metadata(abs_path)
        if meta.get("duration_seconds", 0) <= 0:
            raise RuntimeError(f"Invalid video file with zero duration: {abs_path}")
        return meta
    except Exception as e:
        raise RuntimeError(f"FFprobe/OpenCV validation failed for '{abs_path}': {str(e)}")

def ensure_mp4_format(input_path: str, job_dir: str) -> str:
    """
    Ensures video file is MP4 format. Remuxes non-MP4 files using FFmpeg stream copy if needed.
    """
    abs_input = os.path.abspath(input_path)
    ext = os.path.splitext(abs_input)[1].lower()
    if ext == '.mp4':
        return abs_input

    target_mp4 = os.path.join(job_dir, "normalized_video.mp4")
    print(f"[URL Downloader] Remuxing '{ext}' video to MP4 format via FFmpeg: '{abs_input}' -> '{target_mp4}'")
    
    # Try fast stream copy first
    cmd = ["ffmpeg", "-y", "-i", abs_input, "-c", "copy", target_mp4]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if res.returncode == 0 and os.path.exists(target_mp4) and os.path.getsize(target_mp4) > 1024:
            print(f"[URL Downloader] Remux successful via stream copy.")
            return os.path.abspath(target_mp4)
    except Exception as e:
        print(f"[URL Downloader WARNING] Stream copy remux failed ({e}), falling back to video transcode...")

    # Fallback to fast transcode if stream copy fails
    cmd_transcode = ["ffmpeg", "-y", "-i", abs_input, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", target_mp4]
    try:
        res = subprocess.run(cmd_transcode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        if res.returncode == 0 and os.path.exists(target_mp4) and os.path.getsize(target_mp4) > 1024:
            print(f"[URL Downloader] Transcode to MP4 successful.")
            return os.path.abspath(target_mp4)
        else:
            raise RuntimeError(f"FFmpeg transcode failed: {res.stderr[:200]}")
    except Exception as e:
        raise RuntimeError(f"Failed to normalize video to MP4: {str(e)}")

def fetch_video_from_url(url: str, output_base_dir: str = "uploads") -> Dict[str, Any]:
    """
    Safely ingests, downloads, validates, and normalizes public video URLs via yt-dlp & FFmpeg.
    Uses unique job-specific directory for every URL to prevent conflicts.
    """
    print("=" * 60)
    print(f"[URL Downloader] Received URL: {url}")
    normalized_url = validate_public_url(url)
    print(f"[URL Downloader] Normalized URL: {normalized_url}")

    url_hash = hashlib.md5(normalized_url.encode('utf-8')).hexdigest()[:12]
    job_dir = os.path.abspath(os.path.join(output_base_dir, "url_cache", url_hash))
    os.makedirs(job_dir, exist_ok=True)
    print(f"[URL Downloader] Job ID / Hash: {url_hash}")
    print(f"[URL Downloader] Job directory: {job_dir}")

    # Check for existing valid cached media file in job_dir ONLY
    for fname in os.listdir(job_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in MEDIA_EXTENSIONS and not any(fname.endswith(te) for te in TEMP_EXTENSIONS):
            cached_candidate = os.path.join(job_dir, fname)
            try:
                meta = validate_media_file(cached_candidate)
                normalized_cached = ensure_mp4_format(cached_candidate, job_dir)
                size_mb = round(os.path.getsize(normalized_cached) / (1024 * 1024), 2)
                meta["file_size_mb"] = size_mb
                meta["filename"] = f"URL Video ({meta['width']}x{meta['height']})"
                print(f"[URL Downloader] Reusing valid cached video for '{normalized_url}' -> '{normalized_cached}' ({size_mb} MB)")
                return {
                    "success": True,
                    "file_id": url_hash,
                    "filename": meta["filename"],
                    "local_path": normalized_cached,
                    "metadata": meta,
                    "source_type": "url",
                    "source_url": normalized_url,
                    "cached": True
                }
            except Exception as ce:
                print(f"[URL Downloader WARNING] Stale/corrupt cache candidate '{cached_candidate}': {ce}")

    out_template = os.path.join(job_dir, f"video_{url_hash}.%(ext)s")
    max_bytes = int(MAX_VIDEO_SIZE_MB * 1024 * 1024)

    # Clean any leftover partial downloads from prior attempts
    cleanup_temp_files(job_dir)

    # Step 1: Pre-inspect video metadata (download=False) to validate duration, accessibility, and size early
    pre_opts = {
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'nocheckcertificate': True,
    }

    print(f"[URL Downloader] Inspecting video metadata prior to download...")
    video_title = "Public Video"
    video_duration = 0.0

    try:
        with yt_dlp.YoutubeDL(pre_opts) as ydl:
            pre_info = ydl.extract_info(normalized_url, download=False)
            if pre_info:
                video_title = pre_info.get('title', 'Public Video')
                video_duration = float(pre_info.get('duration') or 0.0)
                is_live = pre_info.get('is_live', False)

                if is_live:
                    raise ValueError("Live streams cannot be summarized. Please provide a recorded lecture video URL.")

                if video_duration > MAX_LECTURE_DURATION_SECONDS:
                    dur_str = format_duration(video_duration)
                    max_dur_str = format_duration(MAX_LECTURE_DURATION_SECONDS)
                    raise ValueError(
                        f"The video '{video_title}' is {dur_str} long. "
                        f"Maximum supported lecture duration is {max_dur_str} (3 hours). "
                        f"Please provide an individual lecture video or shorter segment."
                    )

                # Check approximate file size if known from metadata
                est_bytes = pre_info.get('filesize') or pre_info.get('filesize_approx')
                if est_bytes and est_bytes > (max_bytes * 1.5):
                    est_mb = round(est_bytes / (1024 * 1024), 1)
                    raise ValueError(
                        f"The video '{video_title}' exceeds the maximum allowed download limit "
                        f"(estimated {est_mb} MB > {MAX_VIDEO_SIZE_MB:.0f} MB). "
                        f"Please select a shorter lecture or lower resolution video."
                    )
    except yt_dlp.utils.DownloadError as de:
        err_msg = str(de)
        print(f"[URL Downloader ERROR] Stage: Metadata Pre-check | Exception: {err_msg} | URL: {normalized_url}")
        if "Private video" in err_msg or "Sign in" in err_msg:
            raise ValueError("This video is private, age-restricted, or unavailable.")
        elif "Unsupported URL" in err_msg:
            raise ValueError("This video platform or URL is not currently supported.")
        elif "HTTP Error 404" in err_msg:
            raise ValueError("The requested video was not found (404).")
        else:
            raise ValueError(f"Unable to access video URL: {err_msg.split(';')[-1].strip()}")

    # Step 2: Download stream
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720][ext=mp4]/best[height<=720]/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'merge_output_format': 'mp4',
        'max_filesize': max_bytes,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'nocheckcertificate': True,
    }

    print(f"[URL Downloader] Starting yt-dlp download...")
    returned_info_path: Optional[str] = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=True)
            if info:
                if not video_title or video_title == "Public Video":
                    video_title = info.get('title', 'Public Video')
                try:
                    returned_info_path = ydl.prepare_filename(info)
                except Exception:
                    pass
    except yt_dlp.utils.DownloadError as de:
        cleanup_temp_files(job_dir)
        err_msg = str(de)
        print(f"[URL Downloader ERROR] Stage: yt-dlp Download | Exception: {err_msg} | URL: {normalized_url} | Job ID: {url_hash}")
        if "Private video" in err_msg or "Sign in" in err_msg:
            raise ValueError("This video is private, age-restricted, or unavailable.")
        elif "Unsupported URL" in err_msg:
            raise ValueError("This video platform or URL is not currently supported.")
        elif "HTTP Error 404" in err_msg:
            raise ValueError("The requested video was not found (404).")
        else:
            raise ValueError(f"Unable to download video from URL: {err_msg.split(';')[-1].strip()}")
    except Exception as e:
        cleanup_temp_files(job_dir)
        print(f"[URL Downloader ERROR] Stage: yt-dlp Execution | Exception: {str(e)} | URL: {normalized_url} | Job ID: {url_hash}")
        raise RuntimeError(f"Failed to download video: {str(e)}")

    print(f"[URL Downloader] yt-dlp completed. Returned info path: {returned_info_path}")

    # Discover actual output file in job_dir ONLY
    selected_file: Optional[str] = None

    # Step 1: Check returned info path if valid
    if returned_info_path and os.path.exists(returned_info_path) and os.path.isfile(returned_info_path) and os.path.getsize(returned_info_path) > 0:
        selected_file = os.path.abspath(returned_info_path)

    # Step 2: Scan candidate files in job_dir ONLY if returned info path is not directly available
    if not selected_file:
        candidate_files = []
        for fname in os.listdir(job_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in MEDIA_EXTENSIONS and not any(fname.endswith(te) for te in TEMP_EXTENSIONS):
                candidate_files.append(os.path.join(job_dir, fname))
        
        print(f"[URL Downloader] Candidate files in job directory: {candidate_files}")
        if candidate_files:
            # Pick latest or largest media file in job_dir
            candidate_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
            selected_file = os.path.abspath(candidate_files[0])

    if not selected_file or not os.path.exists(selected_file):
        # Check if download was aborted due to max_filesize or interrupted
        part_files = [f for f in os.listdir(job_dir) if any(f.endswith(te) for te in TEMP_EXTENSIONS)]
        cleanup_temp_files(job_dir)
        if part_files:
            print(f"[URL Downloader ERROR] Stage: File Discovery | Found partial/aborted download files: {part_files}")
            raise ValueError(
                f"Video download was stopped because the file size exceeded the maximum limit of {MAX_VIDEO_SIZE_MB:.0f} MB. "
                f"Please choose a shorter lecture video or lower resolution."
            )
        print(f"[URL Downloader ERROR] Stage: File Discovery | Exception: No output media file located | Job ID: {url_hash}")
        raise RuntimeError("Video download completed but output media file could not be located in job directory.")

    print(f"[URL Downloader] Selected file: {selected_file}")
    file_size_bytes = os.path.getsize(selected_file)
    print(f"[URL Downloader] File size: {file_size_bytes} bytes ({round(file_size_bytes / (1024*1024), 2)} MB)")

    # Validate media file using FFprobe / OpenCV
    try:
        meta = validate_media_file(selected_file)
    except Exception as ve:
        cleanup_temp_files(job_dir)
        raise ve
    print(f"[URL Downloader] FFprobe validation: Success | Duration: {meta['duration_formatted']} ({meta['duration_seconds']}s) | Resolution: {meta['width']}x{meta['height']}")

    # Ensure format is MP4 for downstream HTML5 player & pipeline compatibility
    try:
        final_mp4 = ensure_mp4_format(selected_file, job_dir)
    except Exception as fe:
        cleanup_temp_files(job_dir)
        raise fe
    final_size_mb = round(os.path.getsize(final_mp4) / (1024 * 1024), 2)
    meta["file_size_mb"] = final_size_mb
    meta["filename"] = f"{video_title[:40]}.mp4"
    cleanup_temp_files(job_dir)

    print(f"[URL Downloader] Final normalized path: {final_mp4}")
    print(f"[URL Downloader] Download successful for Job ID '{url_hash}'")
    print("=" * 60)

    return {
        "success": True,
        "file_id": url_hash,
        "filename": meta["filename"],
        "local_path": final_mp4,
        "metadata": meta,
        "source_type": "url",
        "source_url": normalized_url,
        "cached": False
    }

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=BaW_jenozKc"
    try:
        res = fetch_video_from_url(test_url)
        print("Fetch Result:", res)
    except Exception as e:
        print("Fetch Error:", e)
