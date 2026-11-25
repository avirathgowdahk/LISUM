import os
import sys
import json
import hashlib
import time
import urllib.request
import urllib.parse
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


def _gtx_translate(text: str, target_lang: str) -> str:
    """
    Translates text using Google Translate's gtx endpoint (fast, supports POST for longer text),
    falling back to deep_translator.GoogleTranslator if needed.
    """
    if not text or not text.strip():
        return text

    try:
        body = urllib.parse.urlencode({
            "client": "gtx",
            "sl": "auto",
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://translate.googleapis.com/translate_a/single",
            data=body,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data and isinstance(data, list) and data[0]:
            translated = "".join(seg[0] for seg in data[0] if seg and seg[0])
            if translated and translated.strip():
                return translated.strip()
    except Exception:
        pass

    res = GoogleTranslator(source='auto', target=target_lang).translate(text)
    if res and res.strip() and not ("Error 500" in res or "Server Error" in res or "That’s an error" in res or "too many requests" in res.lower()):
        return res.strip()
    raise RuntimeError(f"Translation failed or rate-limited for '{target_lang}'")


def _split_chunks(text: str, max_chars: int = 3500) -> List[str]:
    """Splits text into chunks <= max_chars along newline boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > max_chars and current:
            chunks.append("".join(current))
            current = []
            current_len = 0
        if len(line) > max_chars:
            # Hard-split oversized single line
            for i in range(0, len(line), max_chars):
                chunks.append(line[i:i + max_chars])
        else:
            current.append(line)
            current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def translate_text_summary(summary_text: str, target_lang: str = "hi", output_dir: str = "outputs/multimodal") -> Dict[str, Any]:
    """
    Translates text summary into specified target language.
    Caches translations persistently in cache/translations/ to prevent redundant API calls.
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

    print(f"[Translation] Translating text summary into '{lang}'...")
    max_retries = 2
    translated_text = None
    last_error = None

    for attempt in range(max_retries):
        try:
            chunks = _split_chunks(summary_text, max_chars=3500)
            translated_parts = [_gtx_translate(ch, lang) for ch in chunks]
            candidate = "\n".join(translated_parts).strip()
            if candidate:
                translated_text = candidate
                break
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
    Batch translates a list of text strings efficiently with persistent caching.
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
        # Group into chunks of <= 20 items and <= 3200 chars
        chunks: List[List[int]] = []
        cur_batch: List[int] = []
        cur_len = 0
        for pos, txt in enumerate(uncached_texts):
            clean_len = len(txt.replace("\n", " ")) + 1
            if cur_batch and (len(cur_batch) >= 20 or cur_len + clean_len > 3200):
                chunks.append(cur_batch)
                cur_batch = []
                cur_len = 0
            cur_batch.append(pos)
            cur_len += clean_len
        if cur_batch:
            chunks.append(cur_batch)

        for batch_positions in chunks:
            sub_texts = [uncached_texts[p] for p in batch_positions]
            joined = "\n".join(t.replace("\n", " ").strip() for t in sub_texts)
            batch_ok = False
            try:
                translated_joined = _gtx_translate(joined, lang)
                split_lines = [line.strip() for line in translated_joined.split("\n")]
                if len(split_lines) == len(sub_texts):
                    for j, pos in enumerate(batch_positions):
                        orig_idx = uncached_indices[pos]
                        orig_txt = sub_texts[j]
                        clean_tr = split_lines[j] if split_lines[j] else orig_txt
                        results[orig_idx] = clean_tr
                        _TRANSLATION_CACHE[get_text_key(orig_txt, lang)] = clean_tr
                    batch_ok = True
            except Exception as e_batch:
                print(f"[Translation Batch Warning] Batch request failed ({e_batch}), falling back to item-by-item...")

            if not batch_ok:
                for pos in batch_positions:
                    orig_idx = uncached_indices[pos]
                    orig_txt = uncached_texts[pos]
                    try:
                        tr_item = _gtx_translate(orig_txt, lang)
                        results[orig_idx] = tr_item
                        _TRANSLATION_CACHE[get_text_key(orig_txt, lang)] = tr_item
                    except Exception:
                        results[orig_idx] = orig_txt

        _save_cache()

    return [res if res is not None else texts[i] for i, res in enumerate(results)]

if __name__ == "__main__":
    test_summary = "Machine learning is a subset of artificial intelligence enabling automated learning from data."
    print("Hindi Translation:", translate_text_summary(test_summary, "hi"))
    print("Kannada Translation:", translate_text_summary(test_summary, "kn"))
