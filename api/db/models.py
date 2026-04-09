"""
api/db/models.py
----------------
SQLAlchemy ORM models for the AI Resume Screening System.
"""

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Text, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db.session import Base

class Candidate(Base):
    """
    Represents a job candidate and their resume.
    """
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    original_name: Mapped[str] = mapped_column(String(255))
    
    # Text content
    raw_text: Mapped[str] = mapped_column(Text)
    clean_text: Mapped[str] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    rankings: Mapped[List["RankingScore"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Candidate(name='{self.original_name}')>"


class JobAnalysis(Base):
    """
    Represents a specific Job Description used for a ranking run.
    """
    __tablename__ = "job_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Text content
    raw_text: Mapped[str] = mapped_column(Text)
    clean_text: Mapped[str] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    scores: Mapped[List["RankingScore"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<JobAnalysis(id={self.id})>"


class RankingScore(Base):
    """
    Stores the matching scores and explainability data for a candidate
    against a specific job analysis.
    """
    __tablename__ = "ranking_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("job_analyses.id"))
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    
    # Scores
    rank: Mapped[int] = mapped_column(index=True)
    tfidf_score: Mapped[float] = mapped_column(Float)
    semantic_score: Mapped[float] = mapped_column(Float)
    hybrid_score: Mapped[float] = mapped_column(Float, index=True)
    skill_match_ratio: Mapped[float] = mapped_column(Float)

    # Explainability (stored as JSON)
    matched_skills: Mapped[List[str]] = mapped_column(JSON)
    missing_skills: Mapped[List[str]] = mapped_column(JSON)
    extra_skills: Mapped[List[str]] = mapped_column(JSON)
    
    # Additional keyword data
    keyword_overlap: Mapped[dict] = mapped_column(JSON)

    # Relationships
    candidate: Mapped["Candidate"] = relationship(back_populates="rankings")
    analysis: Mapped["JobAnalysis"] = relationship(back_populates="scores")

    def __repr__(self) -> str:
        return f"<RankingScore(candidate_id={self.candidate_id}, score={self.hybrid_score})>"
