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

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

from config.settings import (
    TFIDF_WEIGHT,
    SEMANTIC_WEIGHT,
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
    """Structured output for a single candidate."""
    rank:            int
    candidate_name:  str
    filename:        str
    tfidf_score:     float
    semantic_score:  float
    hybrid_score:    float
    skill_match_ratio: float
    matched_skills:  List[str] = field(default_factory=list)
    missing_skills:  List[str] = field(default_factory=list)
    extra_skills:    List[str] = field(default_factory=list)
    keyword_overlap: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Ranker ───────────────────────────────────────────────────────────────────

class CandidateRanker:
    """
    End-to-end candidate ranking engine.

    Usage::
        ranker = CandidateRanker()
        results = ranker.rank(
            job_description_clean  = "...",
            job_description_raw    = "...",
            resumes_clean          = ["...", "..."],
            resumes_raw            = ["...", "..."],
            filenames              = ["alice.pdf", "bob.pdf"],
        )
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
        Score and rank all candidates.

        Args:
            job_description_clean: Fully preprocessed JD (for TF-IDF & skill matching).
            job_description_raw:   Original JD text (for semantic model).
            resumes_clean:         Preprocessed resume texts.
            resumes_raw:           Original resume texts (for semantic model).
            filenames:             Resume filenames (one per resume).
            top_n:                 Maximum candidates to return.

        Returns:
            Sorted list of candidate result dicts (best match first).
        """
        n = len(resumes_clean)
        if n == 0:
            logger.warning("No resumes provided to ranker.")
            return []

        logger.info(f"Ranking {n} candidates …")

        # ── 1. TF-IDF similarity (baseline) ──────────────────────────────────
        tfidf_scores = compute_tfidf_scores(job_description_clean, resumes_clean)

        # ── 2. Semantic similarity (advanced) ────────────────────────────────
        semantic_scores = compute_semantic_scores(job_description_raw, resumes_raw)

        # ── 3. Hybrid score (weighted average) ───────────────────────────────
        hybrid_scores = [
            round(TFIDF_WEIGHT * t + SEMANTIC_WEIGHT * s, 6)
            for t, s in zip(tfidf_scores, semantic_scores)
        ]

        # ── 4. Skill extraction ───────────────────────────────────────────────
        jd_skills = extract_skills(job_description_raw, method="hybrid")

        # ── 5. Build per-candidate results ────────────────────────────────────
        results: List[CandidateResult] = []
        for idx in range(n):
            resume_skills = extract_skills(resumes_raw[idx], method="hybrid")
            overlap       = compute_skill_overlap(resume_skills, jd_skills)
            
            # Simple keyword overlap: words in both cleaned texts
            jd_words      = set(job_description_clean.split())
            resume_words  = set(resumes_clean[idx].split())
            common_words  = jd_words & resume_words

            # Derive candidate display name from filename
            candidate_name = _name_from_filename(filenames[idx])

            result = CandidateResult(
                rank=0,  # filled in after sorting
                candidate_name=candidate_name,
                filename=filenames[idx],
                tfidf_score=tfidf_scores[idx],
                semantic_score=semantic_scores[idx],
                hybrid_score=hybrid_scores[idx],
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
            results.append(result)

        # ── 6. Sort by hybrid score desc, apply threshold, assign rank ────────
        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        results = [r for r in results if r.hybrid_score >= MIN_SCORE_THRESHOLD]

        for rank_idx, result in enumerate(results[:top_n], start=1):
            result.rank = rank_idx

        ranked = [r.to_dict() for r in results[:top_n]]
        logger.info(f"Ranking complete. Top score: {ranked[0]['hybrid_score']:.4f}")
        return ranked


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _name_from_filename(filename: str) -> str:
    """
    Convert a filename to a human-readable candidate name.
    Example: "a3f1_john_doe_resume.pdf" → "John Doe"
    """
    from pathlib import Path
    stem = Path(filename).stem
    # Strip leading UUID hex prefix (32 hex chars + underscore)
    if len(stem) > 33 and stem[32] == "_":
        stem = stem[33:]
    # Replace underscores/dashes with spaces and title-case
    name = stem.replace("_", " ").replace("-", " ").title()
    # Remove common suffixes
    for suffix in [" Resume", " Cv", " Application"]:
        name = name.replace(suffix, "")
    return name.strip() or filename
