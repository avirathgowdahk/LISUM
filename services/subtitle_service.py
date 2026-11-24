"""
===============================================================================
Module: Subtitle Service (services/subtitle_service.py)
===============================================================================
1. Purpose:
   Generates synchronized SRT subtitles aligned to ORIGINAL_AUDIO timeline and extracts keywords.
2. Directives:
   - Synchronizes subtitle timestamps with original lecturer speech timeline
   - Remaps transcript segment timing for selected clips
===============================================================================
"""

import os
import re
from typing import List, Dict, Any, Tuple
from multimodal.subtitle_generator import generate_srt_subtitles

def generate_subtitles_and_keywords(
    summary_text: str,
    total_duration_sec: float,
    output_srt_path: str,
    stt_segments: List[Dict[str, Any]] = None,
    selected_segments: List[Tuple[float, float]] = None,
    audio_mode: str = "original",
    target_language: str = "en"
) -> Tuple[str, List[str]]:
    """Generates synchronized SRT subtitle file remapped to ORIGINAL_AUDIO timeline."""
    os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)

    generate_srt_subtitles(
        summary_text=summary_text,
        total_audio_duration_sec=total_duration_sec,
        output_srt_path=output_srt_path,
        stt_segments=stt_segments,
        selected_segments=selected_segments,
        audio_mode=audio_mode,
        target_language=target_language
    )

    # Extract key educational keywords
    words = re.findall(r'\b[A-Za-z]{4,}\b', summary_text)
    freq = {}
    stop = {"this", "that", "with", "from", "have", "which", "there", "their", "they", "will", "would", "about", "could"}
    for w in words:
        w_lower = w.lower()
        if w_lower not in stop:
            freq[w_lower] = freq.get(w_lower, 0) + 1

    sorted_keywords = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:6]
    keywords = [kw.title() for kw in sorted_keywords]

    print(f"[SUBTITLE SERVICE] Generated subtitles '{output_srt_path}' (Timeline: ORIGINAL_AUDIO, Keywords: {keywords})")
    return output_srt_path, keywords
