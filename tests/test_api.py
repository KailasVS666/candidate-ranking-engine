"""
tests/test_api.py
-----------------
FastAPI endpoint tests using HTTPX's synchronous test client.
These tests do NOT require a running uvicorn server.

Endpoints tested:
  GET  /              → health check
  POST /upload_resume → file upload
  POST /analyze       → full ranking
  GET  /results       → list results
  DELETE /clear       → clear session
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# FastAPI test client (sync) via httpx transport
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ─── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_RESUME_TEXT = b"""
Alice Chen - Senior Data Scientist
Skills: Python, SQL, TensorFlow, Docker, AWS, Machine Learning, NLP
Experience: 7 years building ML pipelines
Education: M.S. Computer Science, Stanford
"""

SAMPLE_JD = (
    "Looking for a Senior Data Scientist with Python, SQL, Machine Learning, "
    "TensorFlow, Docker, and AWS experience. 5+ years required."
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _clear_session():
    """Reset server-side resume list between tests."""
    client.delete("/clear")


# ─── Health check ─────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_status_ok(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_has_status_field(self):
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_response_has_version(self):
        data = client.get("/").json()
        assert "version" in data

    def test_models_loaded_key_present(self):
        data = client.get("/").json()
        assert "models_loaded" in data
        assert "tfidf" in data["models_loaded"]


# ─── Upload Resume ────────────────────────────────────────────────────────────

class TestUploadResume:

    def setup_method(self):
        _clear_session()

    def test_upload_txt_file_succeeds(self):
        response = client.post(
            "/upload_resume",
            files=[("files", ("alice.txt", io.BytesIO(SAMPLE_RESUME_TEXT), "text/plain"))],
        )
        assert response.status_code == 200

    def test_upload_response_structure(self):
        response = client.post(
            "/upload_resume",
            files=[("files", ("alice.txt", io.BytesIO(SAMPLE_RESUME_TEXT), "text/plain"))],
        )
        data = response.json()
        assert "status" in data
        assert "uploaded_files" in data
        assert "message" in data

    def test_upload_status_is_success(self):
        response = client.post(
            "/upload_resume",
            files=[("files", ("alice.txt", io.BytesIO(SAMPLE_RESUME_TEXT), "text/plain"))],
        )
        assert response.json()["status"] == "success"

    def test_upload_multiple_files(self):
        resume_b = b"Bob Martinez - Data Scientist. Python, SQL, scikit-learn."
        response = client.post(
            "/upload_resume",
            files=[
                ("files", ("alice.txt", io.BytesIO(SAMPLE_RESUME_TEXT), "text/plain")),
                ("files", ("bob.txt",   io.BytesIO(resume_b),            "text/plain")),
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["uploaded_files"]) == 2

    def test_unsupported_file_type_rejected(self):
        response = client.post(
            "/upload_resume",
            files=[("files", ("resume.docx", io.BytesIO(b"data"), "application/octet-stream"))],
        )
        assert response.status_code == 415


# ─── Analyze ─────────────────────────────────────────────────────────────────

class TestAnalyze:

    def setup_method(self):
        _clear_session()
        # Pre-upload one resume before each test
        client.post(
            "/upload_resume",
            files=[("files", ("alice.txt", io.BytesIO(SAMPLE_RESUME_TEXT), "text/plain"))],
        )

    def test_analyze_succeeds(self):
        response = client.post("/analyze", data={"job_description": SAMPLE_JD})
        assert response.status_code == 200

    def test_analyze_response_has_top_candidates(self):
        data = client.post("/analyze", data={"job_description": SAMPLE_JD}).json()
        assert "top_candidates" in data
        assert isinstance(data["top_candidates"], list)
        assert len(data["top_candidates"]) >= 1

    def test_analyze_candidate_has_required_fields(self):
        data = client.post("/analyze", data={"job_description": SAMPLE_JD}).json()
        candidate = data["top_candidates"][0]
        for field in [
            "rank", "candidate_name", "filename",
            "tfidf_score", "semantic_score", "hybrid_score",
            "skill_match_ratio", "matched_skills",
            "missing_skills", "extra_skills",
        ]:
            assert field in candidate, f"Missing field: {field}"

    def test_analyze_rank_starts_at_one(self):
        data = client.post("/analyze", data={"job_description": SAMPLE_JD}).json()
        assert data["top_candidates"][0]["rank"] == 1

    def test_analyze_scores_bounded(self):
        data = client.post("/analyze", data={"job_description": SAMPLE_JD}).json()
        for c in data["top_candidates"]:
            assert 0.0 <= c["hybrid_score"] <= 1.0

    def test_analyze_without_upload_returns_400(self):
        _clear_session()
        response = client.post("/analyze", data={"job_description": SAMPLE_JD})
        assert response.status_code == 400

    def test_analyze_empty_jd_returns_422(self):
        response = client.post("/analyze", data={"job_description": "  "})
        assert response.status_code == 422

    def test_analyze_respects_top_n(self):
        # Upload 1 resume, ask for top_n=5 → should return 1
        data = client.post(
            "/analyze",
            data={"job_description": SAMPLE_JD, "top_n": 5},
        ).json()
        assert len(data["top_candidates"]) <= 5

    def test_result_file_saved(self):
        data = client.post("/analyze", data={"job_description": SAMPLE_JD}).json()
        assert data.get("result_file") is not None


# ─── Results ─────────────────────────────────────────────────────────────────

class TestResults:

    def test_list_results_returns_200(self):
        response = client.get("/results")
        assert response.status_code == 200

    def test_list_results_has_result_files_key(self):
        data = client.get("/results").json()
        assert "result_files" in data
        assert isinstance(data["result_files"], list)

    def test_get_nonexistent_result_returns_404(self):
        response = client.get("/results/nonexistent_file.json")
        assert response.status_code == 404


# ─── Clear ───────────────────────────────────────────────────────────────────

class TestClear:

    def test_clear_returns_200(self):
        response = client.delete("/clear")
        assert response.status_code == 200

    def test_clear_response_has_status(self):
        data = client.delete("/clear").json()
        assert "status" in data
        assert data["status"] == "cleared"

    def test_after_clear_analyze_returns_400(self):
        client.delete("/clear")
        response = client.post("/analyze", data={"job_description": SAMPLE_JD})
        assert response.status_code == 400
