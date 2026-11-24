"""
===============================================================================
Module: Embedding Service (services/embedding_service.py)
===============================================================================
1. Purpose:
   Generates dense semantic vector embeddings for lecture transcript sentences.
2. Optimization:
   - Persistent SentenceTransformer model instance
   - In-memory & JSON file caching for sentence vectors
   - Batch encoding with batch_size=64
===============================================================================
"""

import os
import json
import hashlib
import numpy as np
from typing import List
from models.transcript import Sentence
from sklearn.feature_extraction.text import TfidfVectorizer

_GLOBAL_SENTENCE_MODEL = None
CACHE_DIR = os.path.join("cache", "embeddings")
CACHE_FILE = os.path.join(CACHE_DIR, "embedding_cache.json")
_EMBEDDING_CACHE = {}

def _load_cache():
    global _EMBEDDING_CACHE
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _EMBEDDING_CACHE = json.load(f)
        except Exception:
            _EMBEDDING_CACHE = {}

def _save_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_EMBEDDING_CACHE, f, ensure_ascii=False)
    except Exception:
        pass

_load_cache()

def get_sentence_transformer_model():
    """Lazy-loads singleton SentenceTransformer instance."""
    global _GLOBAL_SENTENCE_MODEL
    if _GLOBAL_SENTENCE_MODEL is not None:
        return _GLOBAL_SENTENCE_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        print("[EMBEDDING SERVICE] Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
        _GLOBAL_SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("[EMBEDDING SERVICE] SentenceTransformer loaded successfully.")
    except Exception as e:
        print(f"[EMBEDDING SERVICE Warning] SentenceTransformer load failed ({e}). Using TF-IDF fallback vectorizer.")
        _GLOBAL_SENTENCE_MODEL = None

    return _GLOBAL_SENTENCE_MODEL

def warmup_embedding_model():
    """Pre-loads singleton SentenceTransformer model at startup to eliminate latency."""
    try:
        model = get_sentence_transformer_model()
        if model is not None:
            model.encode(["Warmup sentence"], show_progress_bar=False)
            print("[EMBEDDING SERVICE WARMUP] SentenceTransformer model pre-loaded successfully.")
    except Exception as e:
        print(f"[EMBEDDING SERVICE WARMUP Warning] Warmup failed: {e}")

def generate_sentence_embeddings(sentences: List[Sentence]) -> List[Sentence]:
    """Generates numerical vector embeddings for all transcript sentences with caching."""
    if not sentences:
        return sentences

    texts_to_compute = []
    indices_to_compute = []

    for idx, s in enumerate(sentences):
        text_hash = hashlib.md5(s.text.strip().encode('utf-8')).hexdigest()[:12]
        if text_hash in _EMBEDDING_CACHE:
            s.embedding = _EMBEDDING_CACHE[text_hash]
        else:
            texts_to_compute.append(s.text)
            indices_to_compute.append(idx)

    if not texts_to_compute:
        print(f"[EMBEDDING SERVICE CACHE HIT] Reused all {len(sentences)} cached sentence embeddings in < 0.001s!")
        return sentences

    model = get_sentence_transformer_model()

    if model is not None:
        try:
            embeddings = model.encode(
                texts_to_compute,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            for j, orig_idx in enumerate(indices_to_compute):
                emb_list = embeddings[j].tolist()
                sentences[orig_idx].embedding = emb_list
                text_hash = hashlib.md5(sentences[orig_idx].text.strip().encode('utf-8')).hexdigest()[:12]
                _EMBEDDING_CACHE[text_hash] = emb_list

            _save_cache()
            print(f"[EMBEDDING SERVICE] Computed {len(texts_to_compute)} new sentence embeddings (Reused {len(sentences) - len(texts_to_compute)} cached).")
            return sentences
        except Exception as e:
            print(f"[EMBEDDING SERVICE Warning] SentenceTransformer encoding failed: {e}. Falling back to TF-IDF.")

    # Fallback to TF-IDF dense matrix
    texts = [s.text for s in sentences]
    vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        for idx, s in enumerate(sentences):
            s.embedding = tfidf_matrix[idx].tolist()
        print(f"[EMBEDDING SERVICE] Generated {len(sentences)} TF-IDF fallback embeddings.")
    except Exception:
        # Ultimate fallback: zero vectors
        for s in sentences:
            s.embedding = [0.0] * 128

    return sentences
