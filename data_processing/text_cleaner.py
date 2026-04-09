"""
data_processing/text_cleaner.py
--------------------------------
Cleans and normalises raw text extracted from resumes / job descriptions.

Pipeline:
  1. Unicode normalisation (NFKD → ASCII-compatible)
  2. Lowercase
  3. Remove URLs and email addresses
  4. Remove punctuation and special characters
  5. Stopword removal (NLTK)
  6. Tokenisation
  7. Lemmatisation (NLTK WordNetLemmatizer)
  8. Short-token filtering
"""

import re
import unicodedata
from typing import List, Set, Any

from utils.logger import get_logger
from config.settings import (
    REMOVE_STOPWORDS,
    REMOVE_PUNCTUATION,
    LOWERCASE,
    MIN_TOKEN_LENGTH,
)

logger = get_logger(__name__)

# ── Lazy-load NLTK resources ──────────────────────────────────────────────────
_STOPWORDS: Set[str] | None = None
_LEMMATIZER: Any = None


def _get_stopwords() -> Set[str]:
    """
    Lazy-load NLTK stopwords. Downloads them if not present.

    Returns:
        Set of English stopwords.
    """
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk  # type: ignore
        try:
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            logger.info("Downloading NLTK stopwords …")
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _get_lemmatizer() -> Any:
    """
    Lazy-load NLTK WordNetLemmatizer. Downloads wordnet if not present.

    Returns:
        An instance of WordNetLemmatizer.
    """
    global _LEMMATIZER
    if _LEMMATIZER is None:
        import nltk  # type: ignore
        try:
            from nltk.stem import WordNetLemmatizer
            _LEMMATIZER = WordNetLemmatizer()
            # Trigger a lookup to confirm wordnet is available
            _LEMMATIZER.lemmatize("testing")
        except LookupError:
            logger.info("Downloading NLTK wordnet …")
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.stem import WordNetLemmatizer
            _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


# ─── Public API ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full cleaning pipeline returning a single cleaned string suitable for
    both TF-IDF vectorisation and skill extraction.

    Args:
        text (str): Raw text from a resume or job description.

    Returns:
        str: Lowercased, stopword-free, lemmatised string.
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 4. Lowercase
    if LOWERCASE:
        text = text.lower()

    # 5. Remove punctuation / special characters (keep spaces & letters/digits)
    # SMART FIX: We preserve '+' and '#' for C++, C#, and '.' for .NET
    if REMOVE_PUNCTUATION:
        text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenise
    tokens = text.split()

    # 8. Filter short tokens
    tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH]

    # 9. Stopword removal
    if REMOVE_STOPWORDS:
        sw = _get_stopwords()
        tokens = [t for t in tokens if t not in sw]

    # 10. Lemmatise
    lemmatizer = _get_lemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    """
    Tokenise *text* after cleaning and return individual tokens.
    Useful for skill extraction and keyword overlap analysis.

    Args:
        text (str): Raw string to tokenize.

    Returns:
        List[str]: Cleaned tokens.
    """
    cleaned = clean_text(text)
    return cleaned.split()


def extract_sentences(text: str) -> List[str]:
    """
    Split raw text into sentences for sentence-embedding models.
    Keeps sentences as-is (not cleaned) because semantic models
    work better on natural language.

    Args:
        text (str): Raw text to split.

    Returns:
        List[str]: List of sentence strings.
    """
    # Simple sentence splitter – good enough for resumes
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]
