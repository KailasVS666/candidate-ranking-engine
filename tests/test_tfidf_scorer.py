"""
tests/test_tfidf_scorer.py
---------------------------
Unit tests for models/tfidf_scorer.py
Tests TF-IDF vectorisation and cosine similarity scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.tfidf_scorer import TFIDFScorer, compute_tfidf_scores


class TestTFIDFScorer:
    """Tests for the TFIDFScorer class."""

    def test_returns_correct_number_of_scores(
        self, sample_jd_clean, resume_list_clean
    ):
        """One score per resume must be returned."""
        scores = compute_tfidf_scores(sample_jd_clean, resume_list_clean)
        assert len(scores) == len(resume_list_clean)

    def test_scores_are_floats(self, sample_jd_clean, resume_list_clean):
        scores = compute_tfidf_scores(sample_jd_clean, resume_list_clean)
        assert all(isinstance(s, float) for s in scores)

    def test_scores_bounded_zero_to_one(self, sample_jd_clean, resume_list_clean):
        """Cosine similarity must be in [0, 1]."""
        scores = compute_tfidf_scores(sample_jd_clean, resume_list_clean)
        for s in scores:
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1] range"

    def test_identical_text_scores_one(self):
        """A resume identical to the JD should score 1.0 (or very close)."""
        text = "python machine learning data science sql"
        scores = compute_tfidf_scores(text, [text])
        assert scores[0] == pytest.approx(1.0, abs=0.01)

    def test_completely_different_text_scores_low(self):
        """A resume with zero overlap with the JD should score 0.0."""
        jd = "python machine learning deep neural networks"
        resume = "cooking recipes baking kitchen ingredients flour sugar"
        scores = compute_tfidf_scores(jd, [resume])
        assert scores[0] == pytest.approx(0.0, abs=0.05)

    def test_better_resume_scores_higher(
        self, sample_jd_clean, resume_list_clean
    ):
        """
        Alice (resume_list_clean[0]) has far more JD keyword overlap than
        Dave (resume_list_clean[2]). Alice should score higher.
        """
        scores = compute_tfidf_scores(sample_jd_clean, resume_list_clean)
        # Alice (index 0) should beat Dave (index 2)
        assert scores[0] > scores[2], (
            f"Expected Alice ({scores[0]:.4f}) > Dave ({scores[2]:.4f})"
        )

    def test_single_resume(self, sample_jd_clean, sample_resume_clean):
        """Single-resume list should work without errors."""
        scores = compute_tfidf_scores(sample_jd_clean, [sample_resume_clean])
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_empty_resume_scores_zero(self, sample_jd_clean):
        """An empty resume string should score 0 (no overlap)."""
        scores = compute_tfidf_scores(sample_jd_clean, [""])
        assert scores[0] == pytest.approx(0.0, abs=0.01)

    def test_multiple_resumes_consistency(self, sample_jd_clean, resume_list_clean):
        """Running twice on the same data must produce identical results."""
        scores_1 = compute_tfidf_scores(sample_jd_clean, resume_list_clean)
        scores_2 = compute_tfidf_scores(sample_jd_clean, resume_list_clean)
        for s1, s2 in zip(scores_1, scores_2):
            assert s1 == pytest.approx(s2, abs=1e-9)

    def test_scorer_instance_reuse(self, sample_jd_clean, resume_list_clean):
        """TFIDFScorer instance should be usable directly."""
        scorer = TFIDFScorer()
        scores = scorer.fit_transform(sample_jd_clean, resume_list_clean)
        assert len(scores) == len(resume_list_clean)
