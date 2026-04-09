"""
tests/test_ranker.py
---------------------
Integration tests for models/ranker.py
Tests the full end-to-end ranking pipeline (TF-IDF + semantic + skills).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ranker import CandidateRanker, _name_from_filename


# ─── CandidateRanker Tests ────────────────────────────────────────────────────

class TestCandidateRanker:

    @pytest.fixture(autouse=True)
    def setup(
        self,
        sample_jd_raw,
        sample_jd_clean,
        resume_list_raw,
        resume_list_clean,
        filenames,
    ):
        """Run the ranker once; reuse results across tests in this class."""
        ranker = CandidateRanker()
        self.results = ranker.rank(
            job_description_clean=sample_jd_clean,
            job_description_raw=sample_jd_raw,
            resumes_clean=resume_list_clean,
            resumes_raw=resume_list_raw,
            filenames=filenames,
            top_n=10,
        )

    # ── Structure tests ───────────────────────────────────────────────────────

    def test_returns_list(self):
        assert isinstance(self.results, list)

    def test_correct_count(self):
        """Should return at most top_n=10 but at least 1 result."""
        assert 1 <= len(self.results) <= 10

    def test_result_has_required_keys(self):
        required = {
            "rank", "candidate_name", "filename",
            "tfidf_score", "semantic_score", "hybrid_score",
            "skill_match_ratio", "matched_skills",
            "missing_skills", "extra_skills", "keyword_overlap",
        }
        for r in self.results:
            assert required.issubset(r.keys()), (
                f"Missing keys: {required - r.keys()}"
            )

    # ── Ranking order tests ───────────────────────────────────────────────────

    def test_ranks_are_sequential(self):
        ranks = [r["rank"] for r in self.results]
        assert ranks == list(range(1, len(self.results) + 1))

    def test_sorted_descending_by_hybrid_score(self):
        scores = [r["hybrid_score"] for r in self.results]
        assert scores == sorted(scores, reverse=True), (
            "Results are not sorted by hybrid_score descending"
        )

    def test_best_candidate_is_most_relevant(self):
        """
        Alice (index 0) has the most JD-aligned skills; she should rank first
        or at least above Dave (index 2) who mostly has web-dev skills.
        """
        alice_rank = next(
            r["rank"] for r in self.results
            if "alice" in r["filename"].lower()
        )
        dave_rank = next(
            r["rank"] for r in self.results
            if "dave" in r["filename"].lower()
        )
        assert alice_rank < dave_rank, (
            f"Alice (rank {alice_rank}) should outrank Dave (rank {dave_rank})"
        )

    # ── Score boundary tests ──────────────────────────────────────────────────

    def test_hybrid_scores_in_range(self):
        for r in self.results:
            assert 0.0 <= r["hybrid_score"] <= 1.0, (
                f"hybrid_score {r['hybrid_score']} out of [0, 1]"
            )

    def test_tfidf_scores_in_range(self):
        for r in self.results:
            assert 0.0 <= r["tfidf_score"] <= 1.0

    def test_semantic_scores_in_range(self):
        for r in self.results:
            assert 0.0 <= r["semantic_score"] <= 1.0

    def test_skill_match_ratio_in_range(self):
        for r in self.results:
            assert 0.0 <= r["skill_match_ratio"] <= 1.0

    # ── Explainability tests ──────────────────────────────────────────────────

    def test_matched_skills_is_list(self):
        for r in self.results:
            assert isinstance(r["matched_skills"], list)

    def test_missing_skills_is_list(self):
        for r in self.results:
            assert isinstance(r["missing_skills"], list)

    def test_extra_skills_is_list(self):
        for r in self.results:
            assert isinstance(r["extra_skills"], list)

    def test_keyword_overlap_has_count(self):
        for r in self.results:
            kw = r["keyword_overlap"]
            assert "common_keyword_count" in kw
            assert isinstance(kw["common_keyword_count"], int)

    def test_candidate_name_is_string(self):
        for r in self.results:
            assert isinstance(r["candidate_name"], str)
            assert len(r["candidate_name"]) > 0

    # ── Edge case tests ───────────────────────────────────────────────────────

    def test_top_n_respected(self, sample_jd_raw, sample_jd_clean,
                              resume_list_raw, resume_list_clean, filenames):
        ranker = CandidateRanker()
        results = ranker.rank(
            job_description_clean=sample_jd_clean,
            job_description_raw=sample_jd_raw,
            resumes_clean=resume_list_clean,
            resumes_raw=resume_list_raw,
            filenames=filenames,
            top_n=1,
        )
        assert len(results) == 1

    def test_empty_resumes_returns_empty(self, sample_jd_raw, sample_jd_clean):
        ranker = CandidateRanker()
        results = ranker.rank(
            job_description_clean=sample_jd_clean,
            job_description_raw=sample_jd_raw,
            resumes_clean=[],
            resumes_raw=[],
            filenames=[],
        )
        assert results == []


# ─── _name_from_filename helper tests ────────────────────────────────────────

class TestNameFromFilename:

    def test_plain_filename(self):
        assert _name_from_filename("john_doe.txt") == "John Doe"

    def test_strips_resume_suffix(self):
        name = _name_from_filename("jane_smith_resume.pdf")
        assert "Resume" not in name
        assert "Jane" in name

    def test_uuid_prefixed_filename(self):
        # 32-char hex prefix + underscore
        name = _name_from_filename("a" * 32 + "_alice_chen_resume.txt")
        assert "Alice" in name
        assert "Chen" in name

    def test_hyphen_separator(self):
        name = _name_from_filename("bob-jones.pdf")
        assert "Bob" in name
        assert "Jones" in name

    def test_fallback_on_empty_stem(self):
        # Should not raise
        result = _name_from_filename(".pdf")
        assert isinstance(result, str)
