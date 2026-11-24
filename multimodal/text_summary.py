import os
import sys
import json
import re
from typing import Dict, Any, List

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def summarize_transcript_text(
    transcript_text: str,
    output_dir: str = "outputs/multimodal",
    base_name: str = "summary",
    target_compression_ratio: float = 0.4,
    target_duration_sec: float = None
) -> Dict[str, Any]:
    """
    Summarizes transcript text into coherent, narration-ready prose.
    Dynamically targets word count to match target narration duration (~2.2 words/sec)
    using abstractive pipeline / TF-IDF sentence frequency ranking fallback.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not transcript_text or not transcript_text.strip():
        return {
            "success": False,
            "stage": "text_summary",
            "error": "Cannot summarize empty transcript.",
            "text_summary": None
        }

    clean_text = transcript_text.strip()
    words_all = re.findall(r'\w+', clean_text.lower())
    total_transcript_words = len(words_all)
    
    if total_transcript_words < 3:
        return {
            "success": False,
            "stage": "text_summary",
            "error": "Speech transcription quality is too low to generate a reliable text summary.",
            "text_summary": None
        }

    # Split text into clean sentences or clauses (including Indic danda U+0964)
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', clean_text) if len(s.strip()) > 3]
    
    # Fallback: if transcript lacks sentence punctuation, split by commas/semicolons/clause boundaries
    if len(raw_sentences) <= 1 and total_transcript_words > 25:
        clauses = [c.strip() for c in re.split(r'[,;:]+|\b(and|but|so|because|then|where|which|that|though|when|while)\b', clean_text, flags=re.IGNORECASE) if c and len(c.strip()) > 5]
        # Further chunk long clauses if needed (max ~15 words per chunk)
        sentences = []
        for cl in clauses:
            w_list = cl.split()
            if len(w_list) > 20:
                for i in range(0, len(w_list), 15):
                    sentences.append(" ".join(w_list[i:i+15]))
            elif len(w_list) >= 3:
                sentences.append(cl)
        if not sentences:
            sentences = [clean_text]
    else:
        sentences = raw_sentences if raw_sentences else [clean_text]

    # Determine approximate target summary word count based on narration duration (~2.2 words/sec)
    if target_duration_sec and target_duration_sec > 0:
        target_words = int(target_duration_sec * 2.2)
        # Cap target words between 25 words and 90% of transcript
        target_words = max(25, min(int(total_transcript_words * 0.9), target_words))
    else:
        target_words = max(25, int(total_transcript_words * target_compression_ratio))

    print(f"[Text Summary] Transcript Words: {total_transcript_words} | Approx Target Summary Words: {target_words} (~{target_duration_sec or 'N/A'}s target duration)")

    summary_text = None
    abstractive_used = False

    # Check if transcript is non-English (has non-ASCII Indic/foreign characters)
    is_non_english = len([ch for ch in clean_text if ord(ch) > 127]) > (len(clean_text) * 0.15)

    # Attempt HuggingFace Abstractive Summarization Pipeline if available and English text
    if not is_non_english:
        try:
            from transformers import pipeline
            print("[Text Summary] Attempting Abstractive Transformer Summarization...")
            summarizer = pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
            max_len = max(30, min(150, target_words))
            min_len = max(15, int(max_len * 0.5))
            res = summarizer(clean_text[:3000], max_length=max_len, min_length=min_len, do_sample=False)
            if res and len(res) > 0 and "summary_text" in res[0]:
                summary_text = res[0]["summary_text"].strip()
                abstractive_used = True
                print("[Text Summary] Abstractive summarization successful.")
        except Exception as e:
            print(f"[Text Summary] Abstractive pipeline fallback: {e}. Using narration-ready frequency summarization...")
    else:
        print("[Text Summary] Non-English script detected. Using narration-ready frequency summarization...")

    # Fallback / Direct Narration-Ready Extractive Summarization
    if not summary_text:
        word_freq = {}
        stop_words = {"the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "of", "for", "with", "it", "this", "that", "you", "we", "are", "they"}
        for w in words_all:
            if w not in stop_words and len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1

        sentence_scores = []
        for idx, s in enumerate(sentences):
            s_words = re.findall(r'\w+', s.lower())
            score = sum(word_freq.get(w, 0) for w in s_words)
            norm_score = score / max(1, len(s_words))
            sentence_scores.append((norm_score, idx, s))

        # Select sentences until target_words is accumulated
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        selected = []
        acc_words = 0

        for score, idx, s in sentence_scores:
            sw_count = len(re.findall(r'\w+', s))
            selected.append((idx, s))
            acc_words += sw_count
            if acc_words >= target_words:
                break

        # Re-sort chronologically
        selected.sort(key=lambda x: x[0])
        summary_sentences = [s for _, s in selected]

        # Clean filler words for narration-readiness
        summary_sentences = [re.sub(r'^(okay|guys|now|so|then|means|question|well)\b,?\s*', '', s, flags=re.IGNORECASE) for s in summary_sentences]
        summary_sentences = [s[0].upper() + s[1:] if len(s) > 1 else s for s in summary_sentences]
        summary_text = " ".join(summary_sentences)

    summary_word_count = len(re.findall(r'\w+', summary_text.lower()))
    compression_ratio_pct = round((summary_word_count / max(1, total_transcript_words)) * 100, 1)

    bullet_points = [f"• {s.strip()}" for s in re.split(r'(?<=[.!?])\s+', summary_text) if len(s.strip()) > 3]

    summary_path = os.path.join(output_dir, f"{base_name}_text_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[Text Summary Success] Summary Words: {summary_word_count} ({compression_ratio_pct}% of original) | File: {summary_path}")

    return {
        "success": True,
        "stage": "text_summary",
        "abstractive": abstractive_used,
        "text_summary": summary_text,
        "bullet_points": bullet_points,
        "stats": {
            "transcript_word_count": total_transcript_words,
            "summary_word_count": summary_word_count,
            "compression_ratio_pct": compression_ratio_pct
        },
        "summary_file": summary_path
    }

if __name__ == "__main__":
    test_text = "Artificial intelligence and machine learning are revolutionizing video analysis. Deep learning models can process frame sequences efficiently. Automated video summarization saves time for content creators. High density visual feature representations preserve core narrative continuity."
    res = summarize_transcript_text(test_text)
    print("Extractive Summary Result:", json.dumps(res, indent=2))
