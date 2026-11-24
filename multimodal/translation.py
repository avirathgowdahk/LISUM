import os
import sys
import json
import hashlib
import time
from typing import Dict, Any, List
from deep_translator import GoogleTranslator

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CACHE_DIR = os.path.join("cache", "translations")
CACHE_FILE = os.path.join(CACHE_DIR, "translation_cache.json")
_TRANSLATION_CACHE = {}

def _load_cache():
    global _TRANSLATION_CACHE
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _TRANSLATION_CACHE = json.load(f)
        except Exception:
            _TRANSLATION_CACHE = {}

def _save_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_TRANSLATION_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_load_cache()

def get_text_key(text: str, target_lang: str) -> str:
    h = hashlib.md5(text.strip().encode('utf-8')).hexdigest()[:12]
    return f"{target_lang}_{h}"

def translate_text_summary(summary_text: str, target_lang: str = "hi", output_dir: str = "outputs/multimodal") -> Dict[str, Any]:
    """
    Translates text summary into specified target language using GoogleTranslator.
    Caches translations persistent in cache/translations/ to prevent redundant API calls & rate limits.
    """
    os.makedirs(output_dir, exist_ok=True)
    lang = target_lang.lower().strip()

    if not summary_text or not summary_text.strip():
        return {
            "success": False,
            "stage": "translation",
            "error": "Cannot translate empty text.",
            "translated_text": None
        }

    if lang == "en":
        return {
            "success": True,
            "stage": "translation",
            "language": "en",
            "original_text": summary_text,
            "translated_text": summary_text
        }

    cache_key = get_text_key(summary_text, lang)
    if cache_key in _TRANSLATION_CACHE:
        cached_text = _TRANSLATION_CACHE[cache_key]
        return {
            "success": True,
            "stage": "translation",
            "language": lang,
            "original_text": summary_text,
            "translated_text": cached_text,
            "cached": True
        }

    print(f"[Translation] Translating text summary into '{lang}' using GoogleTranslator...")
    max_retries = 2
    translated_text = None
    last_error = None

    for attempt in range(max_retries):
        try:
            res = GoogleTranslator(source='auto', target=lang).translate(summary_text)
            if res and res.strip() and not ("Error 500" in res or "Server Error" in res or "That’s an error" in res):
                translated_text = res.strip()
                break
            else:
                last_error = f"Received invalid response: '{res[:60]}...'"
                time.sleep(0.5)
        except Exception as e:
            last_error = e
            time.sleep(0.5)

    if not translated_text or not translated_text.strip():
        print(f"[Translation Warning] Translation failed for '{lang}': {last_error}. Returning original text.")
        return {
            "success": False,
            "stage": "translation",
            "language": lang,
            "error": str(last_error or "Empty output"),
            "translated_text": summary_text
        }

    _TRANSLATION_CACHE[cache_key] = translated_text
    _save_cache()

    out_file = os.path.join(output_dir, f"summary_translation_{lang}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(translated_text)

    print(f"[Translation Success] Successfully translated to '{lang}' ({len(translated_text)} chars).")

    return {
        "success": True,
        "stage": "translation",
        "language": lang,
        "original_text": summary_text,
        "translated_text": translated_text,
        "file_path": out_file
    }


def translate_text_batch(texts: List[str], target_lang: str = "hi") -> List[str]:
    """
    Batch translates a list of text strings (e.g., subtitle cues) efficiently with persistent caching.
    """
    lang = target_lang.lower().strip()
    if not texts or lang == "en":
        return texts

    results = []
    uncached_indices = []
    uncached_texts = []

    for idx, txt in enumerate(texts):
        if not txt or not txt.strip():
            results.append(txt)
            continue
        key = get_text_key(txt, lang)
        if key in _TRANSLATION_CACHE:
            results.append(_TRANSLATION_CACHE[key])
        else:
            results.append(None)
            uncached_indices.append(idx)
            uncached_texts.append(txt)

    if uncached_texts:
        print(f"[Translation Batch] Translating {len(uncached_texts)} uncached items to '{lang}'...")
        try:
            # Batch translate in chunks of 25
            chunk_size = 25
            translator = GoogleTranslator(source='auto', target=lang)
            for i in range(0, len(uncached_texts), chunk_size):
                sub_texts = uncached_texts[i:i + chunk_size]
                try:
                    translated_sub = translator.translate_batch(sub_texts)
                    for j, tr in enumerate(translated_sub):
                        orig_idx = uncached_indices[i + j]
                        orig_txt = sub_texts[j]
                        clean_tr = tr.strip() if tr and tr.strip() else orig_txt
                        results[orig_idx] = clean_tr
                        _TRANSLATION_CACHE[get_text_key(orig_txt, lang)] = clean_tr
                except Exception as e_batch:
                    print(f"[Translation Batch Warning] Chunk batch translation failed ({e_batch}). Keeping originals.")
                    for j, orig_txt in enumerate(sub_texts):
                        orig_idx = uncached_indices[i + j]
                        results[orig_idx] = orig_txt
            _save_cache()
        except Exception as e:
            print(f"[Translation Batch Warning] Batch translation failed ({e}). Returning original texts.")
            for idx in uncached_indices:
                if results[idx] is None:
                    results[idx] = texts[idx]

    # Fill any remaining Nones with original text
    return [res if res is not None else texts[i] for i, res in enumerate(results)]

if __name__ == "__main__":
    test_summary = "Machine learning is a subset of artificial intelligence enabling automated learning from data."
    print("Hindi Translation:", translate_text_summary(test_summary, "hi"))
    print("Kannada Translation:", translate_text_summary(test_summary, "kn"))
