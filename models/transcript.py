"""
===============================================================================
Module: Transcript Models (models/transcript.py)
===============================================================================
1. Purpose:
   Defines the core data structures for timestamped speech transcription,
   sentence-level segmentation, and lecture text representation.
2. Input:
   Raw ASR outputs (Whisper segment dictionaries with start/end timestamps and word tokens).
3. Processing:
   Encapsulates transcript segments and sentence blocks with precise start_time,
   end_time, confidence, text content, and numerical embedding vectors.
4. Algorithm/Model:
   Python Dataclasses with JSON serialization / deserialization helpers.
5. Output:
   Structured TranscriptSegment and Sentence data objects.
6. Data Stored:
   segment_id, sentence_id, start_time, end_time, text, confidence, words, embedding, importance_score, topic_id.
7. Failure Cases:
   Missing timestamps default to 0.0; empty text handles gracefully.
8. Evaluation:
   Ensures non-overlapping timestamps and duration match.
9. Integration:
   Central data structure used across STT, text processing, embedding, importance scoring, and summary alignment.
===============================================================================
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float

@dataclass
class TranscriptSegment:
    segment_id: int
    start_time: float
    end_time: float
    text: str
    confidence: float = 1.0
    words: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": round(self.start_time, 2),
            "end": round(self.end_time, 2),
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "words": self.words
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=d.get("segment_id", d.get("id", 0)),
            start_time=float(d.get("start_time", d.get("start", 0.0))),
            end_time=float(d.get("end_time", d.get("end", 0.0))),
            text=d.get("text", ""),
            confidence=float(d.get("confidence", 1.0)),
            words=d.get("words", [])
        )

@dataclass
class Sentence:
    sentence_id: int
    segment_id: int
    text: str
    start_time: float
    end_time: float
    importance_score: float = 0.0
    educational_category: str = "general" # definition, concept, explanation, formula, example, conclusion, noise
    topic_id: int = 0
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "segment_id": self.segment_id,
            "text": self.text,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "importance_score": round(self.importance_score, 4),
            "educational_category": self.educational_category,
            "topic_id": self.topic_id
        }
