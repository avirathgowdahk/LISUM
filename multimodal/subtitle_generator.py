"""
===============================================================================
Module: Subtitle Generator (multimodal/subtitle_generator.py)
===============================================================================
1. Purpose:
   Generates synchronized SRT subtitles remapped to the ORIGINAL_AUDIO timeline
   of the final summarized lecture video.
2. Directives:
   - Timeline Source: ORIGINAL_AUDIO (or SOURCE_VIDEO_AUDIO)
   - Remaps subtitle start/end timestamps from original video coordinates to final summary timeline
   - Strict cue packaging: max 12 words per cue, max 2 lines per cue
===============================================================================
"""

import sys
import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from multimodal.translation import translate_text_summary, translate_text_batch

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

MAX_WORDS_PER_CUE = 10
MAX_LINES_PER_CUE = 2
MAX_CHARS_PER_LINE = 42
MIN_CUE_DURATION = 1.0
MAX_CUE_DURATION = 4.0

def format_timestamp_srt(seconds: float) -> str:
    """Formats float seconds into SRT timestamp string format: HH:MM:SS,mmm"""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if millis >= 1000:
        secs += 1
        millis -= 1000
    if secs >= 60:
        minutes += 1
        secs -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def wrap_text_into_lines(text: str, max_chars_per_line: int = MAX_CHARS_PER_LINE) -> str:
    """Wraps text into at most 2 balanced lines without breaking words."""
    text = text.strip()
    if not text or len(text) <= max_chars_per_line:
        return text

    words = text.split()
    if len(words) <= 1:
        return text[:max_chars_per_line]

    mid_char = len(text) // 2
    best_split_idx = 1
    min_diff = len(text)

    curr_len = 0
    for i in range(len(words) - 1):
        curr_len += len(words[i]) + (1 if i > 0 else 0)
        diff = abs(curr_len - mid_char)
        if diff < min_diff:
            min_diff = diff
            best_split_idx = i + 1

    line1 = " ".join(words[:best_split_idx])
    line2 = " ".join(words[best_split_idx:])

    if len(line1) > max_chars_per_line:
        line1 = line1[:max_chars_per_line - 3] + "..."
    if len(line2) > max_chars_per_line:
        line2 = line2[:max_chars_per_line - 3] + "..."

    return f"{line1}\n{line2}"

def split_text_into_short_chunks(text: str, max_words: int = MAX_WORDS_PER_CUE) -> List[str]:
    """Splits long text into short natural phrase chunks (5-10 words)."""
    if not text or not text.strip():
        return []

    raw_clauses = re.split(r'(?<=[.!?,;:।])\s+', text.strip())
    chunks = []

    for clause in raw_clauses:
        clause = clause.strip()
        if not clause:
            continue

        words = clause.split()
        total_w = len(words)

        if total_w <= max_words:
            chunks.append(" ".join(words))
        else:
            target_size = 8
            if total_w <= 16:
                target_size = math.ceil(total_w / 2)
            elif total_w <= 24:
                target_size = math.ceil(total_w / 3)

            for i in range(0, total_w, target_size):
                sub_group = words[i:i + target_size]
                if sub_group:
                    chunks.append(" ".join(sub_group))

    return chunks

def remap_segments_to_summary(
    segments: List[Dict[str, Any]],
    selected_intervals: List[Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """Remaps transcript segment timestamps from source video coordinates to final summary video timeline."""
    if not selected_intervals or not segments:
        return segments

    remapped = []
    cum_summary_time = 0.0

    for orig_start_clip, orig_end_clip in selected_intervals:
        clip_duration = max(0.1, orig_end_clip - orig_start_clip)
        clip_summary_start = cum_summary_time
        clip_summary_end = cum_summary_time + clip_duration

        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            text = seg.get("text", "").strip()

            overlap_start = max(seg_start, orig_start_clip)
            overlap_end = min(seg_end, orig_end_clip)

            if overlap_end > overlap_start and (overlap_end - overlap_start) >= 0.2:
                new_start = clip_summary_start + (overlap_start - orig_start_clip)
                new_end = clip_summary_start + (overlap_end - orig_start_clip)

                new_start = max(clip_summary_start, min(clip_summary_end, new_start))
                new_end = max(new_start + 0.3, min(clip_summary_end, new_end))

                remapped.append({
                    "start": round(new_start, 3),
                    "end": round(new_end, 3),
                    "text": text
                })

        cum_summary_time += clip_duration

    return remapped

def generate_srt_subtitles(
    summary_text: str,
    total_audio_duration_sec: float,
    output_srt_path: str,
    stt_segments: Optional[List[Dict[str, Any]]] = None,
    selected_segments: Optional[List[Tuple[float, float]]] = None,
    audio_mode: str = "original",
    target_language: str = "en"
) -> str:
    """Generates synchronized SRT subtitles aligned to ORIGINAL_AUDIO timeline."""
    os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
    duration = max(1.0, float(total_audio_duration_sec))
    raw_cues = []
    timeline_source = "ORIGINAL_AUDIO"

    print(f"\n[SUBTITLE TIMELINE]\nSource: ORIGINAL_AUDIO\nTimeline duration: {duration:.2f} sec")

    # Strategy: Remap timestamped ASR transcript segments onto final summary timeline
    if stt_segments and len(stt_segments) > 0:
        if selected_segments:
            mapped_segs = remap_segments_to_summary(stt_segments, selected_segments)
        else:
            mapped_segs = stt_segments

        seg_texts = [seg.get("text", "") for seg in mapped_segs]
        if target_language != "en":
            seg_texts = translate_text_batch(seg_texts, target_lang=target_language)

        for idx_s, seg in enumerate(mapped_segs):
            s_start = float(seg["start"])
            s_end = float(seg["end"])
            text = seg_texts[idx_s]

            chunks = split_text_into_short_chunks(text, max_words=MAX_WORDS_PER_CUE)
            if not chunks:
                continue

            chunk_dur = (s_end - s_start) / len(chunks)
            for idx, ch in enumerate(chunks):
                c_start = s_start + idx * chunk_dur
                cue_d = max(MIN_CUE_DURATION, min(MAX_CUE_DURATION, chunk_dur))
                c_end = min(duration, c_start + cue_d)
                raw_cues.append({"start": c_start, "end": c_end, "text": ch})

    # Fallback if stt_segments not provided
    if not raw_cues:
        clean_text = summary_text.strip() if summary_text else "Lecture Summary"
        chunks = split_text_into_short_chunks(clean_text, max_words=MAX_WORDS_PER_CUE)
        if not chunks:
            chunks = ["Lecture Summary"]

        chunk_lens = [max(1, len(c)) for c in chunks]
        total_chars = sum(chunk_lens)

        curr_time = 0.0
        for idx, (ch, c_len) in enumerate(zip(chunks, chunk_lens)):
            allocated_dur = (c_len / total_chars) * duration
            cue_dur = max(MIN_CUE_DURATION, min(MAX_CUE_DURATION, allocated_dur))

            start_t = curr_time
            end_t = min(duration, start_t + cue_dur)
            if idx == len(chunks) - 1:
                end_t = duration

            if end_t > start_t:
                raw_cues.append({"start": start_t, "end": end_t, "text": ch})
                curr_time = end_t

    # Build SRT Entries
    srt_entries = []
    valid_cues = []
    max_words_found = 0
    max_lines_found = 0

    for cue in raw_cues:
        start_t = float(cue["start"])
        end_t = float(cue["end"])
        text = str(cue["text"]).strip()

        if not text or start_t < 0 or end_t <= start_t:
            continue

        formatted_text = wrap_text_into_lines(text, max_chars_per_line=MAX_CHARS_PER_LINE)
        lines = formatted_text.split("\n")
        words = text.split()

        word_count = len(words)
        line_count = len(lines)

        max_words_found = max(max_words_found, word_count)
        max_lines_found = max(max_lines_found, line_count)

        valid_cues.append({
            "id": len(valid_cues) + 1,
            "start": start_t,
            "end": end_t,
            "text": formatted_text,
            "words": word_count
        })

        start_str = format_timestamp_srt(start_t)
        end_str = format_timestamp_srt(end_t)
        srt_entries.append(f"{len(valid_cues)}\n{start_str} --> {end_str}\n{formatted_text}\n")

    if not srt_entries:
        start_str = format_timestamp_srt(0.0)
        end_str = format_timestamp_srt(duration)
        srt_entries.append(f"1\n{start_str} --> {end_str}\nLecture Summary\n")
        valid_cues.append({"id": 1, "start": 0.0, "end": duration, "text": "Lecture Summary", "words": 2})

    srt_content = "\n".join(srt_entries)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"[SUBTITLE] SRT generated successfully ({len(valid_cues)} cues, Timeline: {timeline_source}) -> '{output_srt_path}'")
    return output_srt_path
