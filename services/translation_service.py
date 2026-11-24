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

from multimodal.translation import translate_text_summary

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
