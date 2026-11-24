"""
===============================================================================
Module: Job State Models (models/job.py)
===============================================================================
1. Purpose:
   Defines processing job lifecycle states, progress tracking, SSE messages, and errors.
2. Processing Stages:
   UPLOADED -> VALIDATING -> EXTRACTING_AUDIO -> PREPROCESSING_AUDIO -> TRANSCRIBING
   -> POST_PROCESSING_TRANSCRIPT -> ANALYZING_TRANSCRIPT -> SCORING_IMPORTANCE
   -> GENERATING_SUMMARY -> ALIGNING_SUMMARY -> SELECTING_SEGMENTS -> EXTRACTING_SEGMENTS
   -> MERGING_VIDEO -> GENERATING_CAPTIONS -> TRANSLATING -> FINALIZING -> COMPLETED / FAILED
===============================================================================
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import time

class ProcessingStage:
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    PREPROCESSING_AUDIO = "PREPROCESSING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    POST_PROCESSING_TRANSCRIPT = "POST_PROCESSING_TRANSCRIPT"
    ANALYZING_TRANSCRIPT = "ANALYZING_TRANSCRIPT"
    SCORING_IMPORTANCE = "SCORING_IMPORTANCE"
    GENERATING_SUMMARY = "GENERATING_SUMMARY"
    ALIGNING_SUMMARY = "ALIGNING_SUMMARY"
    SELECTING_SEGMENTS = "SELECTING_SEGMENTS"
    EXTRACTING_SEGMENTS = "EXTRACTING_SEGMENTS"
    MERGING_VIDEO = "MERGING_VIDEO"
    GENERATING_CAPTIONS = "GENERATING_CAPTIONS"
    TRANSLATING = "TRANSLATING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

STAGE_PROGRESS_MAP = {
    ProcessingStage.UPLOADED: 0,
    ProcessingStage.VALIDATING: 5,
    ProcessingStage.EXTRACTING_AUDIO: 10,
    ProcessingStage.PREPROCESSING_AUDIO: 15,
    ProcessingStage.TRANSCRIBING: 25,
    ProcessingStage.POST_PROCESSING_TRANSCRIPT: 40,
    ProcessingStage.ANALYZING_TRANSCRIPT: 50,
    ProcessingStage.SCORING_IMPORTANCE: 55,
    ProcessingStage.GENERATING_SUMMARY: 65,
    ProcessingStage.ALIGNING_SUMMARY: 72,
    ProcessingStage.SELECTING_SEGMENTS: 78,
    ProcessingStage.EXTRACTING_SEGMENTS: 84,
    ProcessingStage.MERGING_VIDEO: 90,
    ProcessingStage.GENERATING_CAPTIONS: 94,
    ProcessingStage.TRANSLATING: 97,
    ProcessingStage.FINALIZING: 99,
    ProcessingStage.COMPLETED: 100,
    ProcessingStage.FAILED: 0
}

@dataclass
class ProcessingJob:
    job_id: str
    lecture_id: str
    stage: str = ProcessingStage.UPLOADED
    progress: int = 0
    message: str = "Job initialized"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None

    def update_stage(self, stage: str, message: str, custom_progress: Optional[int] = None):
        self.stage = stage
        self.message = message
        self.progress = custom_progress if custom_progress is not None else STAGE_PROGRESS_MAP.get(stage, self.progress)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "lecture_id": self.lecture_id,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result
        }
