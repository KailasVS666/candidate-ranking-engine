"""
tests/test_text_cleaner.py
---------------------------
Unit tests for data_processing/text_cleaner.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_processing.text_cleaner import clean_text, tokenize, extract_sentences


class TestCleanText:
    """Tests for the main clean_text() function."""

    def test_basic_lowercasing(self):
        result = clean_text("Python Machine Learning")
        assert result == result.lower()

    def test_removes_urls(self):
        text = "Visit https://example.com for more info."
        result = clean_text(text)
        assert "https" not in result
        assert "example" not in result

    def test_removes_email(self):
        text = "Contact us at info@company.org"
        result = clean_text(text)
        assert "@" not in result

    def test_removes_punctuation(self):
        text = "Python, SQL, and Machine-Learning!"
        result = clean_text(text)
        assert "," not in result
        assert "!" not in result

    def test_removes_stopwords(self):
        text = "the candidate has strong skills in machine learning"
        result = clean_text(text)
        assert "the" not in result.split()
        assert "has" not in result.split()
        assert "in" not in result.split()

    def test_handles_empty_string(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_handles_unicode(self):
        text = "Résumé with naïve content café"
        result = clean_text(text)
        # After NFKD + ASCII — should not raise
        assert isinstance(result, str)

    def test_preserves_technical_terms(self):
        text = "Python SQL TensorFlow scikit-learn pandas"
        result = clean_text(text)
        # Core terms should survive (lemmatised / lowercased)
        assert "python" in result
        assert "sql" in result
        assert "tensorflow" in result

    def test_returns_string(self):
        assert isinstance(clean_text("hello world"), str)


class TestTokenize:
    def test_returns_list(self):
        tokens = tokenize("Python SQL machine learning")
        assert isinstance(tokens, list)

    def test_tokens_are_strings(self):
        tokens = tokenize("data science nlp")
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_input(self):
        assert tokenize("") == []


class TestExtractSentences:
    def test_splits_on_period(self):
        text = "I know Python. I use SQL daily. I love NLP."
        sentences = extract_sentences(text)
        assert len(sentences) >= 2

    def test_returns_list(self):
        assert isinstance(extract_sentences("Hello world."), list)

    def test_empty_string(self):
        result = extract_sentences("")
        assert isinstance(result, list)
