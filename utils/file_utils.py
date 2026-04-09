"""
utils/file_utils.py
-------------------
Utility helpers for safe file I/O operations used across the project.
"""

import json
import shutil
import uuid
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def generate_unique_filename(original_name: str) -> str:
    """
    Prefix the original filename with a UUID4 to avoid collisions.
    Example: 'resume.pdf' → 'a3f1..._resume.pdf'
    """
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix
    return f"{uuid.uuid4().hex}_{stem}{suffix}"


def save_upload(file_bytes: bytes, filename: str, dest_dir: Path) -> Path:
    """
    Write raw bytes to dest_dir/<filename> and return the full path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(file_bytes)
    logger.info(f"Saved upload → {dest_path}")
    return dest_path


def save_text(text: str, filename: str, dest_dir: Path) -> Path:
    """
    Write a UTF-8 text string to dest_dir/<filename>.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_text(text, encoding="utf-8")
    return dest_path


def save_json(data: dict | list, filename: str, dest_dir: Path) -> Path:
    """
    Serialise *data* as pretty-printed JSON and save it.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved JSON  → {dest_path}")
    return dest_path


def load_json(filepath: Path) -> dict | list:
    """Load and return a JSON file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def cleanup_dir(directory: Path) -> None:
    """Remove all files inside *directory* (non-recursive)."""
    if directory.exists():
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
        logger.info(f"Cleaned up directory: {directory}")
