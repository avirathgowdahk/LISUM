"""
===============================================================================
Module: Topic Segmentation Service (services/topic_service.py)
===============================================================================
1. Purpose:
   Segments long lecture transcripts into semantic topics/sections and optimizes
   summary selection to ensure balanced topic coverage across the lecture.
2. Input:
   List of Sentence objects with embeddings.
3. Processing:
   - Detects semantic shift boundaries using sliding-window embedding similarity
   - Assigns topic_id to every sentence (Topic 1, Topic 2, ..., Topic K)
   - Groups topic clusters with representative titles/keywords
4. Algorithm/Model:
   TextTiling-style semantic boundary detection via embedding cosine similarity drops.
5. Output:
   Sentences populated with topic_id and list of topic summaries.
6. Data Stored:
   Sentence.topic_id, list of topic section metadata.
7. Failure Cases:
   Short lectures (< 10 sentences) default to single topic cluster.
8. Evaluation:
   Ensures final summary contains top sentences from every major lecture topic section.
9. Integration:
   Feeds topic-segmented content to summarization_service.py and segment_service.py.
===============================================================================
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from models.transcript import Sentence
from utils.similarity_utils import compute_cosine_similarity

import re

def generate_academic_topic_title(topic_id: int, sentences: List[Sentence]) -> str:
    """Generates a clean, professional academic title for a topic section."""
    if not sentences:
        return f"Topic {topic_id}"

    if topic_id == 1:
        txt = sentences[0].text.lower()
        if any(w in txt for w in ["welcome", "today", "session", "discuss", "intro", "course", "lecture"]):
            return "Introduction & Overview"

    categories = [s.educational_category for s in sentences if getattr(s, "educational_category", None)]
    most_common_cat = max(set(categories), key=categories.count) if categories else None

    all_text = " ".join([s.text for s in sentences])
    words = re.findall(r'\b[A-Za-z]{4,}\b', all_text)
    stop = {"this", "that", "with", "from", "they", "have", "will", "what", "when", "where", "which", "would", "about", "there", "their", "today", "session", "going", "every", "points"}
    freq = {}
    for w in words:
        wl = w.lower()
        if wl not in stop:
            freq[wl] = freq.get(wl, 0) + 1

    top_words = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:3]
    top_words_title = " ".join([w.capitalize() for w in top_words])

    if most_common_cat == "definition":
        prefix = "Definitions & Concepts"
    elif most_common_cat == "formula":
        prefix = "Formulas & Equations"
    elif most_common_cat == "example":
        prefix = "Examples & Applications"
    elif most_common_cat == "concept":
        prefix = "Core Concepts"
    else:
        prefix = None

    if prefix and top_words_title:
        return f"{prefix}: {top_words_title}"
    elif top_words_title:
        return f"{top_words_title}"
    else:
        top_sent = max(sentences, key=lambda x: x.importance_score)
        t_str = top_sent.text.strip()
        if len(t_str) > 50:
            t_str = t_str[:47] + "..."
        return t_str

def segment_lecture_topics(sentences: List[Sentence], target_num_topics: int = None) -> Tuple[List[Sentence], List[Dict[str, Any]]]:
    """Segments lecture transcript into semantic topic sections using embedding similarity shifts."""
    if not sentences:
        return sentences, []

    num_sents = len(sentences)
    if num_sents <= 5:
        for s in sentences:
            s.topic_id = 1
        return sentences, [{"topic_id": 1, "title": "Lecture Content", "sentence_count": num_sents, "start_time": sentences[0].start_time, "end_time": sentences[-1].end_time}]

    if target_num_topics is None:
        target_num_topics = max(2, min(8, int(num_sents / 8)))

    # Compute adjacent sentence similarity curve
    sim_scores = []
    for i in range(num_sents - 1):
        if sentences[i].embedding is not None and sentences[i+1].embedding is not None:
            sim = compute_cosine_similarity(sentences[i].embedding, sentences[i+1].embedding)
        else:
            sim = 0.5
        sim_scores.append(sim)

    # Find boundary indices with lowest local similarity
    sorted_sim_indices = np.argsort(sim_scores)
    boundary_indices = sorted(sorted_sim_indices[:target_num_topics - 1] + 1)

    curr_topic = 1
    topic_sections: Dict[int, List[Sentence]] = {1: []}

    for idx, s in enumerate(sentences):
        if idx in boundary_indices:
            curr_topic += 1
            topic_sections[curr_topic] = []

        s.topic_id = curr_topic
        topic_sections[curr_topic].append(s)

    # Generate topic metadata summary
    topic_metadata = []
    for tid, s_list in topic_sections.items():
        if not s_list:
            continue
        start_ts = s_list[0].start_time
        end_ts = s_list[-1].end_time

        top_sent = max(s_list, key=lambda x: x.importance_score)
        clean_title = generate_academic_topic_title(tid, s_list)

        topic_metadata.append({
            "topic_id": tid,
            "title": clean_title,
            "start_time": start_ts,
            "end_time": end_ts,
            "sentence_count": len(s_list),
            "top_sentence_id": top_sent.sentence_id,
            "key_point": top_sent.text
        })

    print(f"[TOPIC SERVICE] Divided lecture into {len(topic_metadata)} semantic topic sections.")
    return sentences, topic_metadata
