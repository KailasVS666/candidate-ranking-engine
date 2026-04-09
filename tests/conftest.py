"""
tests/conftest.py
-----------------
Shared pytest fixtures available to every test module automatically.

Fixtures defined here:
  - sample_resume_raw       : raw text of a typical resume
  - sample_resume_clean     : cleaned version of the same text
  - sample_jd_raw           : raw job description text
  - sample_jd_clean         : cleaned job description text
  - resume_list_raw         : list of 3 varied raw resumes
  - resume_list_clean       : list of 3 varied cleaned resumes
  - filenames               : matching filename list
  - async_client            : HTTPX async test client for FastAPI
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_processing.text_cleaner import clean_text


# ─── Text fixtures ────────────────────────────────────────────────────────────

RESUME_A = """
Alice Chen — Senior Data Scientist
Python, SQL, PostgreSQL, TensorFlow, PyTorch, scikit-learn, pandas, NumPy,
Docker, Kubernetes, AWS, GCP, NLP, BERT, Hugging Face, Airflow, Spark, MLOps.
7 years of experience building production ML systems.
M.S. Computer Science — Stanford University.
"""

RESUME_B = """
Bob Martinez — Data Scientist
Python, SQL, scikit-learn, pandas, NumPy, Machine Learning, Statistics, Flask.
4 years of experience. B.S. Mathematics.
"""

RESUME_C = """
Dave Wilson — Software Engineer (transitioning to Data Science)
JavaScript, React, Node.js, HTML, CSS, Python (basic), SQL (basic).
Completed Coursera ML Specialisation.
"""

JD = """
Senior Data Scientist needed with 5+ years experience.
Required: Python, SQL, Machine Learning, TensorFlow, Docker, AWS, NLP.
Nice-to-have: Spark, Airflow, Kubernetes, Hugging Face.
"""


@pytest.fixture(scope="session")
def sample_resume_raw() -> str:
    return RESUME_A.strip()


@pytest.fixture(scope="session")
def sample_resume_clean(sample_resume_raw) -> str:
    return clean_text(sample_resume_raw)


@pytest.fixture(scope="session")
def sample_jd_raw() -> str:
    return JD.strip()


@pytest.fixture(scope="session")
def sample_jd_clean(sample_jd_raw) -> str:
    return clean_text(sample_jd_raw)


@pytest.fixture(scope="session")
def resume_list_raw() -> list[str]:
    return [RESUME_A.strip(), RESUME_B.strip(), RESUME_C.strip()]


@pytest.fixture(scope="session")
def resume_list_clean(resume_list_raw) -> list[str]:
    return [clean_text(r) for r in resume_list_raw]


@pytest.fixture(scope="session")
def filenames() -> list[str]:
    return [
        "alice_chen_resume.txt",
        "bob_martinez_resume.txt",
        "dave_wilson_resume.txt",
    ]


# ─── FastAPI async test client ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def async_client():
    """
    Returns an HTTPX test client bound to the FastAPI app.
    Use this instead of a real HTTP server so tests run without starting uvicorn.
    """
    import httpx
    from api.main import app

    with httpx.Client(app=app, base_url="http://testserver") as client:
        yield client
