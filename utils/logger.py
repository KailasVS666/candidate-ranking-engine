"""
utils/logger.py
---------------
Centralised logging setup.
Import `get_logger(__name__)` from any module to get a properly
configured logger that writes to both the console and a rotating file.
"""

import logging
import logging.handlers
from pathlib import Path

from config.settings import LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with:
      - StreamHandler  → console output
      - RotatingFileHandler → logs/app.log (max 5 MB, 3 backups)
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when the module is imported twice
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Rotating file handler ─────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
