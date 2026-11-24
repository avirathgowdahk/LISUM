"""
===============================================================================
Module: Summary Models (models/summary.py)
===============================================================================
1. Purpose:
   Defines data representations for generated lecture summaries (extractive,
   abstractive, hybrid) and statement-level semantic alignment to timestamped transcripts.
2. Input:
   Generated summary sentences, target language, summary ratio, source transcript mappings.
3. Processing:
   Maps every abstractive/extractive summary statement back to exact source transcript
   segments with calculated cosine similarity scores and temporal start/end bounds.
4. Algorithm/Model:
   Dataclass schema enforcing traceability from summary back to source video timestamps.
5. Output:
   Summary and SummaryAlignment data structures.
6. Data Stored:
   summary_id, lecture_id, summary_type, language, text, statements (statement_id, source_segments, start_time, end_time, similarity_score).
7. Failure Cases:
   Low-confidence matches fallback to closest temporal segment or extractive sentence.
8. Evaluation:
   Ensures 100% of summary statements have verifiable timestamp boundaries.
9. Integration:
   Used by video segment extractor, subtitle builder, API responses, and frontend timeline.
===============================================================================
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class SummaryStatement:
    statement_id: int
    text: str
    source_sentence_ids: List[int] = field(default_factory=list)
    source_segment_ids: List[int] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "text": self.text,
            "source_sentence_ids": self.source_sentence_ids,
            "source_segment_ids": self.source_segment_ids,
            "start": round(self.start_time, 2),
            "end": round(self.end_time, 2),
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "similarity": round(self.similarity_score, 4),
            "similarity_score": round(self.similarity_score, 4)
        }

@dataclass
class LectureSummary:
    summary_id: str
    lecture_id: str
    text: str
    summary_type: str = "hybrid" # extractive, abstractive, hybrid
    language: str = "en"
    statements: List[SummaryStatement] = field(default_factory=list)
    topic_bullets: List[Dict[str, Any]] = field(default_factory=list)
    structured_sections: List[Dict[str, Any]] = field(default_factory=list)
    topic_timeline: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "lecture_id": self.lecture_id,
            "summary": self.text,
            "summary_type": self.summary_type,
            "language": self.language,
            "statements": [s.to_dict() for s in self.statements],
            "topic_bullets": self.topic_bullets,
            "structured_sections": self.structured_sections,
            "topic_timeline": self.topic_timeline
        }
