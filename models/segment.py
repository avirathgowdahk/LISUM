"""
===============================================================================
Module: Video Segment Models (models/segment.py)
===============================================================================
1. Purpose:
   Represents video segment clips, temporal bounds, importance rank, and source sentence links.
2. Input:
   Time intervals (start_time, end_time) from timestamp alignment or knapsack selection.
3. Processing:
   Encapsulates duration calculations, clip file references, and deduplication logic.
4. Output:
   VideoClip objects.
===============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class VideoClip:
    clip_id: str
    lecture_id: str
    start_time: float
    end_time: float
    importance_score: float = 0.0
    source_sentence_ids: List[int] = field(default_factory=list)
    file_path: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "lecture_id": self.lecture_id,
            "start": round(self.start_time, 2),
            "end": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "importance": round(self.importance_score, 4),
            "source_sentence_ids": self.source_sentence_ids,
            "file_path": self.file_path
        }
