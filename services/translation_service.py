"""
===============================================================================
Module: Translation Service (services/translation_service.py)
===============================================================================
1. Purpose:
   Translates summaries and subtitles into target languages (Kannada, Hindi, Tamil,
   Telugu, Malayalam, Marathi, Bengali, Spanish, French, German, etc.).
2. Input:
   Original summary text and target language code.
3. Processing:
   Translates text line-by-line while maintaining statement structures and timestamps.
4. Output:
   Translated text summary string.
===============================================================================
"""

from typing import Any, Dict, List, Tuple
from multimodal.translation import translate_text_summary, translate_text_batch

def translate_lecture_summary(summary_text: str, target_language: str = "en") -> str:
    """Translates summary text into requested output language."""
    if not summary_text or target_language.lower() == "en":
        return summary_text

    res = translate_text_summary(summary_text, target_lang=target_language)
    if res.get("success"):
        translated = res.get("translated_text", summary_text)
        print(f"[TRANSLATION SERVICE] Translated summary to '{target_language.upper()}' ({len(translated)} chars).")
        return translated
    else:
        print(f"[TRANSLATION SERVICE Warning] Translation to '{target_language}' failed: {res.get('error')}. Returning original text.")
        return summary_text


def translate_structured_outputs(
    target_language: str,
    structured_sections: List[Any],
    summary_statements: List[Dict[str, Any]],
    topic_timeline: List[Any],
    timestamped_sentences: List[Dict[str, Any]],
    raw_transcript: str,
) -> Tuple[List[Any], List[Dict[str, Any]], List[Any], List[Dict[str, Any]], str]:
    """
    Batch-translates structured summary sections, alignment statements,
    topic timeline entries, and timestamped transcript sentences in one pass.
    """
    lang = (target_language or "en").lower().strip()
    if lang == "en":
        return (
            structured_sections,
            summary_statements,
            topic_timeline,
            timestamped_sentences,
            raw_transcript,
        )

    unique_texts: List[str] = []
    seen = set()

    def _collect(val: Any):
        if isinstance(val, str):
            s = val.strip()
            if s and s not in seen:
                seen.add(s)
                unique_texts.append(s)

    for sec in structured_sections or []:
        if isinstance(sec, dict):
            _collect(sec.get("title"))
            for b in sec.get("bullets") or []:
                _collect(b)
            _collect(sec.get("content"))
        else:
            _collect(sec)

    for stmt in summary_statements or []:
        if isinstance(stmt, dict):
            _collect(stmt.get("text"))

    for topic in topic_timeline or []:
        if isinstance(topic, dict):
            _collect(topic.get("title"))
            _collect(topic.get("key_point"))
            _collect(topic.get("bullet"))
        else:
            _collect(topic)

    for sent in timestamped_sentences or []:
        if isinstance(sent, dict):
            _collect(sent.get("text"))

    if not unique_texts:
        return (
            structured_sections,
            summary_statements,
            topic_timeline,
            timestamped_sentences,
            raw_transcript,
        )

    translated_list = translate_text_batch(unique_texts, target_lang=lang)
    tr_map = dict(zip(unique_texts, translated_list))

    def _tr(val: Any) -> Any:
        if isinstance(val, str):
            s = val.strip()
            return tr_map.get(s, val) if s else val
        return val

    tr_sections: List[Any] = []
    for sec in structured_sections or []:
        if isinstance(sec, dict):
            new_sec = dict(sec)
            if "title" in new_sec:
                new_sec["title"] = _tr(new_sec["title"])
            if "bullets" in new_sec and isinstance(new_sec["bullets"], list):
                new_sec["bullets"] = [_tr(b) for b in new_sec["bullets"]]
            if "content" in new_sec:
                new_sec["content"] = _tr(new_sec["content"])
            tr_sections.append(new_sec)
        else:
            tr_sections.append(_tr(sec))

    tr_statements: List[Dict[str, Any]] = []
    for stmt in summary_statements or []:
        new_stmt = dict(stmt)
        if "text" in new_stmt:
            new_stmt["text"] = _tr(new_stmt["text"])
        tr_statements.append(new_stmt)

    tr_timeline: List[Any] = []
    for topic in topic_timeline or []:
        if isinstance(topic, dict):
            new_topic = dict(topic)
            for k in ("title", "key_point", "bullet"):
                if k in new_topic:
                    new_topic[k] = _tr(new_topic[k])
            tr_timeline.append(new_topic)
        else:
            tr_timeline.append(_tr(topic))

    tr_sentences: List[Dict[str, Any]] = []
    for sent in timestamped_sentences or []:
        new_sent = dict(sent)
        if "text" in new_sent:
            new_sent["text"] = _tr(new_sent["text"])
        tr_sentences.append(new_sent)

    tr_raw_transcript = " ".join(
        s.get("text", "").strip() for s in tr_sentences if s.get("text")
    ) or translate_lecture_summary(raw_transcript, target_language=lang)

    return tr_sections, tr_statements, tr_timeline, tr_sentences, tr_raw_transcript
