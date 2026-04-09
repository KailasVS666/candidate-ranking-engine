"""
tests/test_skill_extractor.py
------------------------------
Unit tests for feature_engineering/skill_extractor.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feature_engineering.skill_extractor import (
    rule_based_extraction,
    compute_skill_overlap,
)


SAMPLE_RESUME = """
Alice Chen is a Senior Data Scientist with 7 years of experience.
She is proficient in Python, SQL, PostgreSQL, TensorFlow, PyTorch,
Docker, Kubernetes, AWS, and Apache Spark.
She has also worked extensively with NLP, BERT, and Hugging Face.
"""

SAMPLE_JD = """
We need a Data Scientist with Python, SQL, Machine Learning,
TensorFlow, Docker, and experience on AWS.
"""


class TestRuleBasedExtraction:
    """Tests for keyword-matching skill extractor."""

    def test_finds_python(self):
        skills = rule_based_extraction(SAMPLE_RESUME)
        assert "Python" in skills

    def test_finds_sql(self):
        skills = rule_based_extraction(SAMPLE_RESUME)
        assert "SQL" in skills

    def test_finds_tensorflow(self):
        skills = rule_based_extraction(SAMPLE_RESUME)
        assert "TensorFlow" in skills

    def test_finds_docker(self):
        skills = rule_based_extraction(SAMPLE_RESUME)
        assert "Docker" in skills

    def test_returns_set(self):
        result = rule_based_extraction(SAMPLE_RESUME)
        assert isinstance(result, set)

    def test_empty_text_returns_empty_set(self):
        result = rule_based_extraction("")
        assert isinstance(result, set)
        assert len(result) == 0

    def test_no_false_positives_for_r(self):
        # "R" as a skill should not match inside longer words like "architecture"
        text = "software architecture design"
        skills = rule_based_extraction(text)
        assert "R" not in skills


class TestComputeSkillOverlap:
    """Tests for the skill overlap / explainability helper."""

    def setup_method(self):
        self.resume_skills = {"Python", "SQL", "TensorFlow", "Docker", "Spark"}
        self.jd_skills     = {"Python", "SQL", "Machine Learning", "TensorFlow", "AWS"}

    def test_matched_skills(self):
        result = compute_skill_overlap(self.resume_skills, self.jd_skills)
        assert "Python"     in result["matched_skills"]
        assert "SQL"        in result["matched_skills"]
        assert "TensorFlow" in result["matched_skills"]

    def test_missing_skills(self):
        result = compute_skill_overlap(self.resume_skills, self.jd_skills)
        assert "Machine Learning" in result["missing_skills"]
        assert "AWS"              in result["missing_skills"]

    def test_extra_skills(self):
        result = compute_skill_overlap(self.resume_skills, self.jd_skills)
        assert "Docker" in result["extra_skills"]
        assert "Spark"  in result["extra_skills"]

    def test_match_ratio_between_0_and_1(self):
        result = compute_skill_overlap(self.resume_skills, self.jd_skills)
        assert 0.0 <= result["match_ratio"] <= 1.0

    def test_perfect_match(self):
        result = compute_skill_overlap(self.jd_skills, self.jd_skills)
        assert result["match_ratio"] == 1.0
        assert len(result["missing_skills"]) == 0

    def test_no_match(self):
        result = compute_skill_overlap({"Java"}, {"Python"})
        assert result["match_ratio"] == 0.0
        assert len(result["matched_skills"]) == 0

    def test_empty_jd_skills(self):
        result = compute_skill_overlap({"Python"}, set())
        assert result["match_ratio"] == 0.0

    def test_returns_sorted_lists(self):
        result = compute_skill_overlap(self.resume_skills, self.jd_skills)
        assert result["matched_skills"] == sorted(result["matched_skills"])
        assert result["missing_skills"] == sorted(result["missing_skills"])
