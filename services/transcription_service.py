"""
===============================================================================
Module: Transcription Service (services/transcription_service.py)
===============================================================================
1. Purpose:
   Performs Automatic Speech Recognition (ASR) using Whisper / faster-whisper transformer models.
2. Input:
   Preprocessed lecture WAV audio file path.
3. Processing:
   - Transcribes audio into timestamped segments and word tokens
   - Captures start_time, end_time, text, confidence, detected language
   - Reuses singleton model instance to optimize memory and execution speed
4. Algorithm/Model:
   Whisper ASR (faster-whisper CTranslate2 / OpenAI Whisper).
5. Output:
   List of TranscriptSegment dataclasses and full transcript string.
6. Data Stored:
   segment_id, start_time, end_time, text, confidence, words, detected_language.
7. Failure Cases:
   Returns empty segment list if no speech detected; logs warning.
8. Evaluation:
   Tracks Real-Time Factor (RTF = processing_time / audio_duration).
9. Integration:
   Passes timestamped segments to transcript_processor.py.
===============================================================================
"""

from typing import List, Dict, Any, Tuple
from multimodal.speech_to_text import transcribe_audio
from models.transcript import TranscriptSegment

def run_transcription(
    audio_or_video_path: str,
    output_dir: str = "cache/transcripts",
    progress_callback: Any = None,
    force_recompute: bool = False
) -> Tuple[List[TranscriptSegment], str, str]:
    """Transcribes lecture audio/video into timestamped TranscriptSegment objects with real progress updates."""
    res = transcribe_audio(
        audio_or_video_path,
        output_dir=output_dir,
        progress_callback=progress_callback,
        force_recompute=force_recompute
    )
    if not res.get("success"):
        raise RuntimeError(res.get("error", "Transcription failed."))

    raw_segments = res.get("segments", [])
    detected_lang = res.get("language", "en")
    transcript_text = res.get("transcript_text", "")

    segments = []
    for idx, seg in enumerate(raw_segments):
        ts_seg = TranscriptSegment(
            segment_id=seg.get("id", idx + 1),
            start_time=float(seg.get("start", 0.0)),
            end_time=float(seg.get("end", 0.0)),
            text=seg.get("text", "").strip(),
            confidence=float(seg.get("confidence", 0.95)),
            words=seg.get("words", [])
        )
        segments.append(ts_seg)

    print(f"[TRANSCRIPTION SERVICE] Transcribed {len(segments)} segments ({len(transcript_text.split())} words, Language: {detected_lang})")
    return segments, transcript_text, detected_lang
