"""
models/tfidf_scorer.py
-----------------------
Baseline similarity scorer using TF-IDF + Cosine Similarity.

How it works:
  1. Fit a TfidfVectorizer on a corpus = [job_description] + [all resumes].
  2. Transform the job description into a TF-IDF vector.
  3. Transform each resume into its TF-IDF vector.
  4. Compute cosine similarity between each resume vector and the JD vector.
  5. Return scores in [0, 1].

TF-IDF captures *term frequency* weighted by *inverse document frequency*,
meaning rare but important words contribute more to the similarity score.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity        # type: ignore

from config.settings import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from utils.logger import get_logger

logger = get_logger(__name__)


class TFIDFScorer:
    """
    Fits a shared TF-IDF vocabulary across the job description and all
    resumes, then computes cosine similarity for each resume.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,       # replace TF with 1 + log(TF) to dampen high freq
            strip_accents="unicode",
            analyzer="word",
        )

    def fit_transform(
        self,
        job_description: str,
        resumes: List[str],
    ) -> List[float]:
        """
        Fit the vectorizer and compute similarity scores.

        Args:
            job_description: Cleaned JD text.
            resumes:         List of cleaned resume texts.

        Returns:
            List of cosine similarity scores [0, 1] — one per resume.
        """
        corpus = [job_description] + resumes
        logger.info(f"TF-IDF fitting on corpus size={len(corpus)}")

        # Fit on the entire corpus (JD + resumes) for a shared vocabulary
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # First row = JD vector; remaining rows = resume vectors
        jd_vector      = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        # Cosine similarity returns shape (n_resumes, 1)
        similarities = cosine_similarity(resume_vectors, jd_vector).flatten()
        scores = [round(float(s), 6) for s in similarities]

        logger.info(
            f"TF-IDF scores → min={min(scores):.4f}, "
            f"max={max(scores):.4f}, mean={np.mean(scores):.4f}"
        )
        return scores


def compute_tfidf_scores(
    job_description: str,
    resumes: List[str],
) -> List[float]:
    """
    Convenience function — creates a TFIDFScorer and returns scores.

    Args:
        job_description: Cleaned job description text.
        resumes:         List of cleaned resume texts.

    Returns:
        List[float] – cosine similarity score per resume.
    """
    scorer = TFIDFScorer()
    return scorer.fit_transform(job_description, resumes)
