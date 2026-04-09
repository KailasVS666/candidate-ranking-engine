"""
feature_engineering/skill_extractor.py
----------------------------------------
Two-strategy skill extraction:

  A. Rule-based  – fast keyword matching against a curated skills list.
  B. NLP-based   – spaCy pattern matching using a PhraseMatcher for
                   precision multi-word phrase detection.

The public entry point is `extract_skills(text, method="hybrid")`.
"hybrid" returns the union of both approaches.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Set, List, Tuple, Any, Dict

from config.settings import SKILLS_FILE, SPACY_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Load skills list ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_skills_list() -> List[str]:
    """
    Load skills from SKILLS_FILE (one per line).
    Results are cached so the file is read only once per process.

    Returns:
        List[str]: List of skill strings.
    """
    skills_path = Path(SKILLS_FILE)
    if not skills_path.exists():
        logger.warning(f"Skills file not found: {skills_path}")
        return []
    skills = [
        line.strip()
        for line in skills_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info(f"Loaded {len(skills)} skills from {skills_path.name}")
    return skills


# ─── A. Rule-Based Extraction ─────────────────────────────────────────────────

def rule_based_extraction(text: str) -> Set[str]:
    """
    Match every skill from the predefined list against *text* using
    whole-word, case-insensitive regex.

    Args:
        text (str): Input text (resume or JD).

    Returns:
        Set[str]: Set of matched skill strings (original casing from skills list).
    """
    skills = _load_skills_list()
    text_lower = text.lower()
    matched: Set[str] = set()

    for skill in skills:
        # SMART FIX: Standard \b (word boundary) fails for skills ending in symbols like C++
        # We use lookarounds to ensure the skill is not surrounded by alphanumeric characters.
        # This allows matching "C++" but not "R" in "React".
        escaped_skill = re.escape(skill.lower())
        pattern = rf"(?<![a-z0-9]){escaped_skill}(?![a-z0-9])"
        
        if re.search(pattern, text_lower):
            matched.add(skill)

    logger.debug(f"Rule-based: found {len(matched)} skills")
    return matched


# ─── B. NLP-Based Extraction (spaCy PhraseMatcher) ───────────────────────────

@lru_cache(maxsize=1)
def _get_spacy_matcher() -> Tuple[Any, Any]:
    """
    Build and cache a spaCy PhraseMatcher loaded with every skill phrase.

    Returns:
        Tuple[nlp, matcher]: Loaded spaCy model and PhraseMatcher.
    """
    try:
        import spacy  # type: ignore
        from spacy.matcher import PhraseMatcher  # type: ignore
    except ImportError:
        logger.warning("spaCy not installed – NLP extraction unavailable.")
        return None, None

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        logger.warning(
            f"spaCy model '{SPACY_MODEL}' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )
        return None, None

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    skills = _load_skills_list()
    patterns = list(nlp.pipe(skills))
    matcher.add("SKILLS", patterns)
    logger.info("spaCy PhraseMatcher ready.")
    return nlp, matcher


def nlp_based_extraction(text: str) -> Set[str]:
    """
    Use spaCy PhraseMatcher to extract skills. Falls back to an empty
    set if spaCy or the model is not available.

    Args:
        text (str): Input text.

    Returns:
        Set[str]: Set of matched skill strings.
    """
    nlp, matcher = _get_spacy_matcher()
    if nlp is None:
        return set()

    skills = _load_skills_list()
    skills_lower = {s.lower(): s for s in skills}

    doc = nlp(text[:1_000_000])  # spaCy limit guard
    matches = matcher(doc)
    matched: Set[str] = set()
    for _, start, end in matches:
        span_text = doc[start:end].text.lower()
        if span_text in skills_lower:
            matched.add(skills_lower[span_text])

    logger.debug(f"NLP-based: found {len(matched)} skills")
    return matched


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_skills(
    text: str,
    method: Literal["rule", "nlp", "hybrid"] = "hybrid",
) -> Set[str]:
    """
    Extract skills from *text* using the specified *method*.

    Args:
        text (str):  Raw or lightly cleaned resume / JD text.
        method (str): "rule"   → keyword matching only
                      "nlp"    → spaCy PhraseMatcher only
                      "hybrid" → union of both (default)

    Returns:
        Set[str]: Set of extracted skill strings.
    """
    if method == "rule":
        return rule_based_extraction(text)
    elif method == "nlp":
        return nlp_based_extraction(text)
    else:  # "hybrid"
        rule_skills = rule_based_extraction(text)
        nlp_skills  = nlp_based_extraction(text)
        combined = rule_skills | nlp_skills
        logger.info(
            f"Hybrid extraction → rule={len(rule_skills)}, "
            f"nlp={len(nlp_skills)}, combined={len(combined)}"
        )
        return combined


def compute_skill_overlap(
    resume_skills: Set[str],
    jd_skills: Set[str],
) -> Dict[str, Any]:
    """
    Compare resume skills against job-description skills.

    Args:
        resume_skills (Set[str]): Skills extracted from resume.
        jd_skills (Set[str]): Skills required by the JD.

    Returns:
        Dict[str, Any] with keys:
          - matched_skills  : skills present in both resume and JD
          - missing_skills  : skills required by JD but absent in resume
          - extra_skills    : skills candidate has beyond JD requirements
          - match_ratio     : len(matched) / len(jd_skills)  [0-1]
    """
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    extra   = resume_skills - jd_skills
    ratio   = len(matched) / max(len(jd_skills), 1)

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "extra_skills":   sorted(extra),
        "match_ratio":    round(ratio, 4),
    }
