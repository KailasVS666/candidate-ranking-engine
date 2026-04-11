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
    Extract text from PDF with layout-aware fallback logic.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"PDF not found: {filepath}")
        return ""

    # 1. Attempt Primary: pdfplumber (best for clean text structure)
    text = _extract_with_pdfplumber(filepath)
    
    # 2. Heuristic check: 
    # If text is empty OR suspiciously short (< 200 chars), it might be 
    # a multi-column layout that pdfplumber failed to parse.
    if len(text.strip()) > 200:
        logger.info(f"pdfplumber extracted {len(text)} chars from {filepath.name}")
        return text

    # 3. Fallback: PyMuPDF (best for layout sorting)
    logger.info(f"pdfplumber yielded thin results ({len(text)} chars); falling back to PyMuPDF...")
    text_fallback = _extract_with_pymupdf(filepath)
    
    # Return the one that gave us more meaningful data
    if len(text_fallback.strip()) > len(text.strip()):
        logger.info(f"PyMuPDF extracted {len(text_fallback)} chars from {filepath.name}")
        return text_fallback
    
    if text.strip():
        return text

    logger.error(f"HARD EXTRACTION FAIL: No text found for {filepath.name}")
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
                # IMPORTANT: sort=True handles multi-column layouts
                pages_text.append(page.get_text("text", sort=True))
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
