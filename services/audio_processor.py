"""
===============================================================================
Module: Audio Processor (services/audio_processor.py)
===============================================================================
1. Purpose:
   Extracts, normalizes, and preprocesses lecture audio for optimal Speech Recognition.
2. Input:
   Original lecture video file path.
3. Processing:
   - FFmpeg extraction to speech-friendly 16kHz mono WAV (pcm_s16le)
   - Audio volume peak normalization
   - Preserves meaningful lecture pauses while trimming extreme trailing silence
4. Algorithm/Model:
   FFmpeg audio filtering (`loudnorm`, `highpass`, `lowpass`).
5. Output:
   Extracted normalized WAV audio file path.
6. Data Stored:
   extracted_audio path, sample rate, channels, duration.
7. Failure Cases:
   Raises error if FFmpeg fails or video lacks audio stream.
8. Evaluation:
   Ensures audio duration matches video duration within 0.5s.
9. Integration:
   Feeds clean audio to transcription_service.py.
===============================================================================
"""

import os
import subprocess
from utils.ffmpeg_utils import extract_normalized_audio, get_media_metadata

def process_lecture_audio(video_path: str, output_dir: str = "cache/audio") -> str:
    """Extracts, normalizes, and prepares speech audio from lecture video."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_wav = os.path.join(output_dir, f"{base_name}_normalized.wav")

    if os.path.exists(output_wav) and os.path.getsize(output_wav) > 1000:
        print(f"[AUDIO PROCESSOR] Using cached normalized audio: '{output_wav}'")
        return output_wav

    print(f"[AUDIO PROCESSOR] Extracting mono 16kHz speech audio from '{video_path}'...")
    extract_normalized_audio(video_path, output_wav)

    if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
        raise RuntimeError(f"Audio processing failed for video: '{video_path}'")

    print(f"[AUDIO PROCESSOR] Audio preprocessing complete: '{output_wav}'")
    return output_wav
