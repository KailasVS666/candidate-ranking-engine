"""
config/settings.py
------------------
Central configuration for the AI Resume Screening & Candidate Ranking System.
Adjust these values to tune the system behaviour without touching core logic.
"""

import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ─── Storage Paths ───────────────────────────────────────────────────────────
UPLOAD_DIR       = BASE_DIR / "uploads"          # raw uploaded PDFs / text files
PROCESSED_DIR    = BASE_DIR / "processed"        # cleaned text artefacts
RESULTS_DIR      = BASE_DIR / "results"          # ranking output JSONs
LOG_DIR          = BASE_DIR / "logs"

# Create directories on import so they always exist
for _dir in [UPLOAD_DIR, PROCESSED_DIR, RESULTS_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Text Preprocessing ──────────────────────────────────────────────────────
REMOVE_STOPWORDS  = True
REMOVE_PUNCTUATION = True
LOWERCASE         = True
MIN_TOKEN_LENGTH  = 2       # discard tokens shorter than this

# ─── Skill Extraction ────────────────────────────────────────────────────────
# Path to the predefined skills list (one skill per line)
SKILLS_FILE = BASE_DIR / "data" / "skills_list.txt"

# spaCy model to use for NLP-based extraction
SPACY_MODEL = "en_core_web_sm"

# ─── Scoring & Similarity ────────────────────────────────────────────────────
# Weights for the hybrid score  (must sum to 1.0)
TFIDF_WEIGHT     = 0.40
SEMANTIC_WEIGHT  = 0.60

# Sentence-Transformers model (runs 100 % locally — no API calls)
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# TF-IDF vectoriser settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams

# ─── Ranking ─────────────────────────────────────────────────────────────────
TOP_N_CANDIDATES  = 10        # how many candidates to return by default
MIN_SCORE_THRESHOLD = 0.0     # drop candidates below this hybrid score

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# ─── Frontend ────────────────────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "AI Resume Screening System"
STREAMLIT_LAYOUT     = "wide"

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE   = LOG_DIR / "app.log"
