"""
models/ranker.py
----------------
Orchestrates the full candidate scoring and ranking pipeline.

Responsibilities:
  1. Receive cleaned resume texts + raw resume texts + cleaned JD.
  2. Run TF-IDF scoring (baseline).
  3. Run semantic scoring (advanced).
  4. Compute a weighted hybrid score.
  5. Run skill extraction and overlap analysis for explainability.
  6. Sort candidates by hybrid score and return a structured result list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Set

from config.settings import (
    TFIDF_WEIGHT,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    TOP_N_CANDIDATES,
    MIN_SCORE_THRESHOLD,
)
from models.tfidf_scorer import compute_tfidf_scores
from models.semantic_scorer import compute_semantic_scores
from feature_engineering.skill_extractor import extract_skills, compute_skill_overlap
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    """
    Structured output for a single candidate analysis.
    
    Attributes:
        rank (int): Final position in the ranking [1-N].
        candidate_name (str): Human-readable name derived from filename.
        filename (str): The original filename stored on disk.
        tfidf_score (float): Cosine similarity from TF-IDF model.
        semantic_score (float): Cosine similarity from Sentence-Transformer.
        hybrid_score (float): Weighted average of TF-IDF and Semantic scores.
    """
    rank:              int
    candidate_name:    str
    filename:          str
    tfidf_score:       float
    semantic_score:    float
    hybrid_score:      float
    skill_match_ratio: float
    matched_skills:    List[str] = field(default_factory=list)
    missing_skills:    List[str] = field(default_factory=list)
    extra_skills:      List[str] = field(default_factory=list)
    keyword_overlap:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass instance to a JSON-serializable dictionary."""
        return asdict(self)


# ─── Ranker ───────────────────────────────────────────────────────────────────

class CandidateRanker:
    """
    End-to-end candidate ranking engine.
    
    This class orchestrates the entire scoring pipeline, combining two
    different similarity metrics with skill-based extraction logic.
    """

    def rank(
        self,
        job_description_clean: str,
        job_description_raw: str,
        resumes_clean: List[str],
        resumes_raw: List[str],
        filenames: List[str],
        top_n: int = TOP_N_CANDIDATES,
    ) -> List[Dict[str, Any]]:
        """
        Score and rank all candidates for a given job description.

        Args:
            job_description_clean (str): Preprocessed JD text.
            job_description_raw (str): Original JD text (for semantic embeddings).
            resumes_clean (List[str]): List of preprocessed resume texts.
            resumes_raw (List[str]): List of original resume texts.
            filenames (List[str]): Corresponding filenames for the resumes.
            top_n (int): Max number of candidates to return.

        Returns:
            List[Dict[str, Any]]: Sorted results list with scores and skill chips.
        """
        n: int = len(resumes_clean)
        if n == 0:
            logger.warning("No resumes provided to ranker.")
            return []

        logger.info(f"Ranking {n} candidates …")

        # 1. Compute similarity vectors
        tfidf_scores: List[float] = compute_tfidf_scores(job_description_clean, resumes_clean)
        semantic_scores: List[float] = compute_semantic_scores(job_description_raw, resumes_raw)

        # 2. Extract JD skills once
        jd_skills: Set[str] = extract_skills(job_description_raw, method="hybrid")
        jd_words: Set[str] = set(job_description_clean.split())

        # 3. Build and score candidates
        results: List[CandidateResult] = []
        for idx in range(n):
            
            # Skill & Keyword Overlap
            resume_skills = extract_skills(resumes_raw[idx], method="hybrid")
            overlap = compute_skill_overlap(resume_skills, jd_skills)
            resume_words = set(resumes_clean[idx].split())
            common_words = jd_words & resume_words

            # Build result object
            result = CandidateResult(
                rank=0,
                candidate_name=_name_from_filename(filenames[idx]),
                filename=filenames[idx],
                tfidf_score=tfidf_scores[idx],
                semantic_score=semantic_scores[idx],
                hybrid_score=0.0,  # Calculated below
                skill_match_ratio=overlap["match_ratio"],
                matched_skills=overlap["matched_skills"],
                missing_skills=overlap["missing_skills"],
                extra_skills=overlap["extra_skills"],
                keyword_overlap={
                    "common_keyword_count": len(common_words),
                    "sample_keywords": sorted(common_words)[:20],
                    "jd_keyword_count": len(jd_words),
                    "resume_keyword_count": len(resume_words),
                },
            )
            
            # Hybrid Calculation (Now with 3 pillars!)
            result.hybrid_score = round(
                TFIDF_WEIGHT * tfidf_scores[idx] + 
                SEMANTIC_WEIGHT * semantic_scores[idx] +
                SKILL_WEIGHT * result.skill_match_ratio,
                6
            )
            
            results.append(result)

        # 4. Sort and Filter
        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        results = [r for r in results if r.hybrid_score >= MIN_SCORE_THRESHOLD]

        # 5. Finalize Ranks
        for rank_idx, result in enumerate(results[:top_n], start=1):
            result.rank = rank_idx

        logger.info(f"Ranking complete. Top score: {results[0].hybrid_score:.4f}")
        return [r.to_dict() for r in results[:top_n]]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _name_from_filename(filename: str) -> str:
    """
    Convert a filename with unique prefix into a human-readable candidate name.
    
    Example: "a1b2c3d4_kailas_nair_resume.pdf" → "Kailas Nair"

    Args:
        filename (str): The filename string.

    Returns:
        str: The extracted name, title-cased.
    """
    from pathlib import Path
    stem: str = Path(filename).stem
    
    # Strip leading UUID hex prefix (32 chars + underscore) if present
    if len(stem) > 33 and stem[32] == "_":
        stem = stem[33:]
        
    # Formatting
    name: str = stem.replace("_", " ").replace("-", " ").title()
    
    # Remove common artifacts
    for suffix in [" Resume", " Cv", " Application"]:
        name = name.replace(suffix, "")
        
    return name.strip() or filename
