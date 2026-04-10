"""
api/schemas.py
--------------
Pydantic models for request validation and response serialisation.
FastAPI uses these to auto-generate OpenAPI documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Request Models ───────────────────────────────────────────────────────────

class JobDescriptionRequest(BaseModel):
    """Payload for providing a plain-text job description."""
    job_description: str = Field(
        ...,
        min_length=20,
        description="The full text of the job description to screen against.",
        example=(
            "We are looking for a Senior Data Scientist proficient in Python, "
            "machine learning, TensorFlow, and SQL with 5+ years of experience."
        ),
    )
    top_n: Optional[int] = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top candidates to return.",
    )


class FeedbackRequest(BaseModel):
    """Payload for submitting user feedback on a candidate score."""
    score_id: int
    manual_score: float = Field(..., ge=0, le=10)
    notes: Optional[str] = None


# ─── Response Models ──────────────────────────────────────────────────────────

class KeywordOverlap(BaseModel):
    common_keyword_count: int
    sample_keywords: List[str]
    jd_keyword_count: int
    resume_keyword_count: int


class CandidateResponse(BaseModel):
    rank: int
    candidate_name: str
    filename: str
    category: Optional[str] = Field(None, description="Resume category based on folder structure")
    
    # NEW: Include current manual score if it exists
    score_id: int
    manual_score: Optional[float] = None
    feedback_notes: Optional[str] = None
    
    tfidf_score: float          = Field(description="Baseline TF-IDF cosine similarity [0-1]")
    semantic_score: float       = Field(description="Semantic embedding similarity [0-1]")
    hybrid_score: float         = Field(description="Weighted hybrid score [0-1]")
    skill_match_ratio: float    = Field(description="Fraction of JD skills found in resume [0-1]")
    matched_skills: List[str]   = Field(description="Skills present in both resume and JD")
    missing_skills: List[str]   = Field(description="JD skills absent from resume")
    extra_skills: List[str]     = Field(description="Skills candidate has beyond JD requirements")
    keyword_overlap: KeywordOverlap


class AnalysisResponse(BaseModel):
    status: str
    job_description_preview: str        = Field(description="First 200 chars of the JD")
    total_resumes_processed: int
    top_candidates: List[CandidateResponse]
    result_file: Optional[str] = None   # path where JSON was saved


class UploadResponse(BaseModel):
    status: str
    uploaded_files: List[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
