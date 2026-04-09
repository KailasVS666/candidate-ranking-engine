"""
models/semantic_scorer.py
--------------------------
Advanced similarity scorer using Sentence-Transformers (semantic embeddings).

Why this is better than TF-IDF:
  • TF-IDF treats "Python developer" and "software engineer proficient in Python"
    as very different strings. Sentence embeddings capture the *meaning*, so
    semantically similar phrases score highly even without exact word overlap.

Model used: "all-MiniLM-L6-v2"
  • Small (80 MB), fast, runs locally — no API calls.
  • Strong multilingual performance on STS tasks.
  • Output: 384-dim normalised embeddings.

Cosine similarity on L2-normalised vectors == dot product, which is 
numerically equivalent and very fast.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from config.settings import SENTENCE_TRANSFORMER_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """
    Load and cache the SentenceTransformer model.
    The @lru_cache ensures the heavy model is loaded only once per process.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info(f"Loading SentenceTransformer: {SENTENCE_TRANSFORMER_MODEL}")
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        logger.info("SentenceTransformer ready.")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Semantic scoring will be skipped. "
            "Install with: pip install sentence-transformers"
        )
        return None


def compute_semantic_scores(
    job_description: str,
    resumes: List[str],
) -> List[float]:
    """
    Encode the JD and all resumes, then return cosine similarity scores.

    Args:
        job_description: Raw or lightly preprocessed JD text.
                         (Semantic models work best on natural language —
                          do NOT pass heavily cleaned text.)
        resumes:         List of raw / lightly preprocessed resume texts.

    Returns:
        List[float] – semantic similarity score [0, 1] per resume.
        Returns list of zeros if sentence-transformers is unavailable.
    """
    model = _load_model()

    if model is None:
        logger.warning("Returning zero semantic scores (model unavailable).")
        return [0.0] * len(resumes)

    # Encode all texts at once for efficiency (batched on GPU if available)
    all_texts  = [job_description] + resumes
    logger.info(f"Encoding {len(all_texts)} texts with SentenceTransformer …")
    embeddings = model.encode(
        all_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise → cosine == dot product
        show_progress_bar=False,
        batch_size=32,
    )

    jd_embedding      = embeddings[0:1]        # shape (1, 384)
    resume_embeddings = embeddings[1:]          # shape (n_resumes, 384)

    # Dot product on normalised vectors == cosine similarity
    similarities = (resume_embeddings @ jd_embedding.T).flatten()  # shape (n_resumes,)

    # Clip to [0, 1] – cosine can be slightly negative for opposite meanings
    scores = [round(float(max(0.0, s)), 6) for s in similarities]

    logger.info(
        f"Semantic scores → min={min(scores):.4f}, "
        f"max={max(scores):.4f}, mean={np.mean(scores):.4f}"
    )
    return scores
