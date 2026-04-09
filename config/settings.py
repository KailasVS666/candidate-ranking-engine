"""
config/settings.py
------------------
Central configuration for the AI Resume Screening & Candidate Ranking System.
Adjust these values to tune the system behaviour without touching core logic.
"""

import os
from pathlib import Path
from typing import Tuple

# ─── Project Root ────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# ─── Unified Storage Strategy ────────────────────────────────────────────────
# All mutable data lives here to simplify Docker volume mapping and backups.
STORAGE_DIR: Path = BASE_DIR / "storage"
UPLOAD_DIR: Path  = STORAGE_DIR / "uploads"          # Raw uploaded PDFs / text files
PROCESSED_DIR: Path = STORAGE_DIR / "processed"      # Cleaned text artefacts
RESULTS_DIR: Path   = STORAGE_DIR / "results"        # Ranking output JSONs
LOG_DIR: Path       = STORAGE_DIR / "logs"

# Ensure all storage paths exist on startup
for _dir in [UPLOAD_DIR, PROCESSED_DIR, RESULTS_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Text Preprocessing ──────────────────────────────────────────────────────
REMOVE_STOPWORDS: bool   = True
REMOVE_PUNCTUATION: bool = True
LOWERCASE: bool          = True
MIN_TOKEN_LENGTH: int    = 2       # Discard tokens shorter than this

# ─── Skill Extraction ────────────────────────────────────────────────────────
# Path to the predefined skills list (one skill per line)
SKILLS_FILE: Path = BASE_DIR / "data" / "skills_list.txt"

# spaCy model to use for NLP-based extraction
SPACY_MODEL: str = "en_core_web_sm"

# ─── Scoring & Similarity ────────────────────────────────────────────────────
# Weights for the hybrid score (must sum to 1.0)
TFIDF_WEIGHT: float    = 0.40
SEMANTIC_WEIGHT: float = 0.60

# Sentence-Transformers model (runs 100% locally — no AI API calls)
SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

# TF-IDF vectoriser settings
TFIDF_MAX_FEATURES: int     = 5000
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)   # Unigrams + bigrams

# ─── Ranking ─────────────────────────────────────────────────────────────────
TOP_N_CANDIDATES: int      = 10        # How many candidates to return by default
MIN_SCORE_THRESHOLD: float = 0.0       # Drop candidates below this hybrid score

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST: str   = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int   = int(os.getenv("API_PORT", 8000))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

# ─── Frontend ────────────────────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE: str = "AI Resume Screening System"
STREAMLIT_LAYOUT: str      = "wide"

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE: Path  = LOG_DIR / "app.log"
