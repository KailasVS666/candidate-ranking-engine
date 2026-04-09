"""
data_processing/pdf_extractor.py
---------------------------------
Extracts raw text from PDF files using pdfplumber (primary) with a
PyMuPDF (fitz) fallback for scanned or complex layouts.
"""

from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(filepath: str | Path) -> str:
    """
    Extract all text from *filepath*.

    Strategy:
      1. Try pdfplumber  → works well for most digital PDFs.
      2. Fall back to PyMuPDF (fitz) if pdfplumber yields nothing.
      3. Return empty string if both fail (caller must handle this case).

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        Extracted text as a single string (pages joined by newlines).
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"PDF not found: {filepath}")
        return ""

    # ── Attempt 1: pdfplumber ─────────────────────────────────────────────────
    text = _extract_with_pdfplumber(filepath)
    if text.strip():
        logger.debug(f"pdfplumber extracted {len(text)} chars from {filepath.name}")
        return text

    # ── Attempt 2: PyMuPDF ───────────────────────────────────────────────────
    logger.warning(f"pdfplumber returned empty text for {filepath.name}; trying PyMuPDF …")
    text = _extract_with_pymupdf(filepath)
    if text.strip():
        logger.debug(f"PyMuPDF extracted {len(text)} chars from {filepath.name}")
        return text

    logger.error(f"Both extractors failed for {filepath.name}.")
    return ""


# ─── Private helpers ──────────────────────────────────────────────────────────

def _extract_with_pdfplumber(filepath: Path) -> str:
    """Extract text page-by-page using pdfplumber."""
    try:
        import pdfplumber  # type: ignore
        pages_text: list[str] = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
        return "\n".join(pages_text)
    except ImportError:
        logger.warning("pdfplumber not installed — skipping.")
        return ""
    except Exception as exc:
        logger.warning(f"pdfplumber failed ({exc})")
        return ""


def _extract_with_pymupdf(filepath: Path) -> str:
    """Extract text page-by-page using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore  (PyMuPDF)
        pages_text: list[str] = []
        with fitz.open(str(filepath)) as doc:
            for page in doc:
                pages_text.append(page.get_text("text"))
        return "\n".join(pages_text)
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed — skipping.")
        return ""
    except Exception as exc:
        logger.warning(f"PyMuPDF failed ({exc})")
        return ""


def extract_text_from_txt(filepath: str | Path) -> str:
    """
    Read plain text / Markdown resume files.

    Args:
        filepath: Path to a .txt or .md file.

    Returns:
        File contents as a string, or empty string on error.
    """
    filepath = Path(filepath)
    try:
        return filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Failed reading text file {filepath}: {exc}")
        return ""
