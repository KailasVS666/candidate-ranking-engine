"""
api/routes.py
-------------
FastAPI route definitions.

Endpoints:
  GET  /               → health/status check
  POST /upload_resume  → upload one or more PDF/TXT resume files
  POST /analyze        → run ranking against a job description
  GET  /results        → list previously saved result JSONs
  GET  /results/{name} → retrieve a specific result file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.schemas import (
    AnalysisResponse,
    CandidateResponse,
    HealthResponse,
    JobDescriptionRequest,
    KeywordOverlap,
    UploadResponse,
)
from config.settings import UPLOAD_DIR, PROCESSED_DIR, RESULTS_DIR
from data_processing.pdf_extractor import extract_text_from_pdf, extract_text_from_txt
from data_processing.text_cleaner import clean_text
from models.ranker import CandidateRanker
from utils.file_utils import generate_unique_filename, save_upload, save_json
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# In-memory store of uploaded resume info (reset on server restart)
# In production, use a DB or cache (Redis/PostgreSQL).
_uploaded_resumes: List[dict] = []  # [{filename, raw_text, clean_text}]


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return service status and model availability."""
    # Try importing optional deps to report availability
    tfidf_ok = True  # always available (scikit-learn)
    try:
        from sentence_transformers import SentenceTransformer  # noqa
        semantic_ok = True
    except ImportError:
        semantic_ok = False

    try:
        import spacy  # noqa
        spacy_ok = True
    except ImportError:
        spacy_ok = False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded={
            "tfidf": tfidf_ok,
            "sentence_transformers": semantic_ok,
            "spacy": spacy_ok,
        },
    )


# ─── Upload Resumes ───────────────────────────────────────────────────────────

@router.post("/upload_resume", response_model=UploadResponse, tags=["Resumes"])
async def upload_resume(files: List[UploadFile] = File(...)):
    """
    Upload one or more resume files (PDF or TXT).

    The files are saved to the uploads/ directory and their extracted
    text is stored in memory for the next /analyze call.
    """
    global _uploaded_resumes
    uploaded_names: List[str] = []

    for upload in files:
        original_name = upload.filename or "resume.pdf"
        suffix = Path(original_name).suffix.lower()

        if suffix not in {".pdf", ".txt", ".md"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{suffix}'. Upload PDF or TXT files.",
            )

        # Save raw bytes to uploads/
        raw_bytes = await upload.read()
        unique_name = generate_unique_filename(original_name)
        saved_path = save_upload(raw_bytes, unique_name, UPLOAD_DIR)

        # Extract text
        if suffix == ".pdf":
            raw_text = extract_text_from_pdf(saved_path)
        else:
            raw_text = extract_text_from_txt(saved_path)

        if not raw_text.strip():
            logger.warning(f"Empty text extracted from {original_name}")
            raw_text = ""

        clean = clean_text(raw_text)

        _uploaded_resumes.append({
            "filename":   unique_name,
            "original":   original_name,
            "raw_text":   raw_text,
            "clean_text": clean,
        })
        uploaded_names.append(original_name)
        logger.info(f"Uploaded & processed: {original_name} → {unique_name}")

    return UploadResponse(
        status="success",
        uploaded_files=uploaded_names,
        message=f"{len(uploaded_names)} resume(s) uploaded. Call POST /analyze next.",
    )


# ─── Analyse / Rank ───────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(
    job_description: str = Form(..., description="Full text of the job description"),
    top_n: int = Form(default=10, ge=1, le=100),
):
    """
    Rank all uploaded resumes against the given job description.

    Expects at least one resume to have been uploaded via POST /upload_resume.
    Returns a ranked list with explainable scores.
    """
    global _uploaded_resumes

    if not _uploaded_resumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resumes uploaded. Call POST /upload_resume first.",
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Job description cannot be empty.",
        )

    logger.info(f"Analyzing {len(_uploaded_resumes)} resumes …")

    jd_clean = clean_text(job_description)

    ranker = CandidateRanker()
    ranked = ranker.rank(
        job_description_clean=jd_clean,
        job_description_raw=job_description,
        resumes_clean=[r["clean_text"] for r in _uploaded_resumes],
        resumes_raw=[r["raw_text"]   for r in _uploaded_resumes],
        filenames=[r["filename"]    for r in _uploaded_resumes],
        top_n=top_n,
    )

    # Persist results to disk
    import uuid as _uuid
    result_filename = f"results_{_uuid.uuid4().hex[:8]}.json"
    save_json(
        {"job_description": job_description, "rankings": ranked},
        result_filename,
        RESULTS_DIR,
    )

    # Build response
    top_candidates = [
        CandidateResponse(
            rank=c["rank"],
            candidate_name=c["candidate_name"],
            filename=c["filename"],
            tfidf_score=c["tfidf_score"],
            semantic_score=c["semantic_score"],
            hybrid_score=c["hybrid_score"],
            skill_match_ratio=c["skill_match_ratio"],
            matched_skills=c["matched_skills"],
            missing_skills=c["missing_skills"],
            extra_skills=c["extra_skills"],
            keyword_overlap=KeywordOverlap(**c["keyword_overlap"]),
        )
        for c in ranked
    ]

    return AnalysisResponse(
        status="success",
        job_description_preview=job_description[:200],
        total_resumes_processed=len(_uploaded_resumes),
        top_candidates=top_candidates,
        result_file=result_filename,
    )


# ─── Clear Session ────────────────────────────────────────────────────────────

@router.delete("/clear", tags=["Resumes"])
async def clear_session():
    """
    Clear all uploaded resumes from the current session.
    """
    global _uploaded_resumes
    count = len(_uploaded_resumes)
    _uploaded_resumes = []
    return {"status": "cleared", "removed_count": count}


# ─── List Results ─────────────────────────────────────────────────────────────

@router.get("/results", tags=["Results"])
async def list_results():
    """List all saved result JSON files."""
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return {"result_files": [f.name for f in files]}


@router.get("/results/{filename}", tags=["Results"])
async def get_result(filename: str):
    """Retrieve a specific result file by name."""
    path = RESULTS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Result '{filename}' not found.")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))
