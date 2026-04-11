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

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Depends
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, delete

from api.schemas import (
    AnalysisResponse,
    CandidateResponse,
    FeedbackRequest,
    HealthResponse,
    JobDescriptionRequest,
    KeywordOverlap,
    UploadResponse,
)
from api.db.session import get_db
from api.db.models import Candidate, JobAnalysis, RankingScore
from config.settings import UPLOAD_DIR, PROCESSED_DIR, RESULTS_DIR
from data_processing.pdf_extractor import extract_text_from_pdf, extract_text_from_txt
from data_processing.text_cleaner import clean_text
from models.ranker import CandidateRanker
from models.vector_store import VectorStoreManager
from utils.file_utils import generate_unique_filename, save_upload, save_json
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
vector_store = VectorStoreManager()


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/", response_model=HealthResponse, tags=["System"])
async def health_check(db: Session = Depends(get_db)):
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
async def upload_resume(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload one or more resume files (PDF or TXT).

    The files are saved to the uploads/ directory and their metadata
    is stored permanently in the database.
    """
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

        # Save to Database
        db_candidate = Candidate(
            filename=unique_name,
            original_name=original_name,
            raw_text=raw_text,
            clean_text=clean
        )
        db.add(db_candidate)
        uploaded_names.append(original_name)
        
        # 4. Add to Vector Store
        vector_store.add_resumes([clean], [unique_name])
        
        logger.info(f"Uploaded & stored in DB: {original_name} → {unique_name}")

    db.commit()

    return UploadResponse(
        status="success",
        uploaded_files=uploaded_names,
        message=f"{len(uploaded_names)} resume(s) uploaded and saved to DB. Call POST /analyze next.",
    )


# ─── Analyse / Rank ───────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(
    job_description: str = Form(..., description="Full text of the job description"),
    top_n: int = Form(default=10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Rank all resumes in the database against the given job description.
    """
    # 1. Fetch all candidates from DB
    candidates = db.execute(select(Candidate)).scalars().all()

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resumes found in database. Call POST /upload_resume first.",
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Job description cannot be empty.",
        )

    logger.info(f"Analyzing {len(candidates)} resumes against JD …")

    jd_clean = clean_text(job_description)

    # 2. Run Ranking Engine
    ranker = CandidateRanker()
    ranked_dicts = ranker.rank(
        job_description_clean=jd_clean,
        job_description_raw=job_description,
        resumes_clean=[c.clean_text for c in candidates],
        resumes_raw=[c.raw_text for c in candidates],
        filenames=[c.filename for c in candidates],
        top_n=top_n,
    )

    # 3. Persist Analysis and individual scores to DB
    db_analysis = JobAnalysis(
        raw_text=job_description,
        clean_text=jd_clean
    )
    db.add(db_analysis)
    db.flush()  # Get analysis.id

    top_candidates_responses = []
    
    # Mapping of results back to DB candidates
    candidate_map = {c.filename: c for c in candidates}

    for res in ranked_dicts:
        candidate = candidate_map[res["filename"]]
        
        # Save scores to DB
        db_score = RankingScore(
            analysis_id=db_analysis.id,
            candidate_id=candidate.id,
            rank=res["rank"],
            tfidf_score=res["tfidf_score"],
            semantic_score=res["semantic_score"],
            hybrid_score=res["hybrid_score"],
            skill_match_ratio=res["skill_match_ratio"],
            matched_skills=res["matched_skills"],
            missing_skills=res["missing_skills"],
            extra_skills=res["extra_skills"],
            keyword_overlap=res["keyword_overlap"]
        )
        db.add(db_score)
        db.flush()  # Generate score.id for the response
        
        # Prepare response model
        top_candidates_responses.append(
            CandidateResponse(
                rank=res["rank"],
                candidate_name=res["candidate_name"],
                filename=res["filename"],
                category=candidate.category,
                score_id=db_score.id,
                manual_score=None,
                feedback_notes=None,
                tfidf_score=res["tfidf_score"],
                semantic_score=res["semantic_score"],
                hybrid_score=res["hybrid_score"],
                skill_match_ratio=res["skill_match_ratio"],
                matched_skills=res["matched_skills"],
                missing_skills=res["missing_skills"],
                extra_skills=res["extra_skills"],
                keyword_overlap=KeywordOverlap(**res["keyword_overlap"]),
            )
        )

    db.commit()

    return AnalysisResponse(
        status="success",
        job_description_preview=job_description[:200],
        total_resumes_processed=len(candidates),
        top_candidates=top_candidates_responses,
        result_file=f"db_analysis_{db_analysis.id}",
    )


@router.post("/feedback", tags=["Analysis"])
async def submit_feedback(data: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Update a ranking score with manual user feedback (1-10 rating).
    This data is used to train future machine learning models.
    """
    score = db.get(RankingScore, data.score_id)
    if not score:
        raise HTTPException(status_code=404, detail="Ranking record not found.")

    score.manual_score = data.manual_score
    score.feedback_notes = data.notes
    db.commit()

    return {"status": "success", "message": f"Rating of {data.manual_score}/10 saved."}


# ─── Clear All Data ───────────────────────────────────────────────────────────

@router.delete("/clear", tags=["System"])
async def clear_data(db: Session = Depends(get_db)):
    """
    Clear all candidates, analyses, and scores from the database.
    Does NOT delete the actual PDF/TXT files from storage/uploads.
    """
    db.execute(delete(RankingScore))
    db.execute(delete(JobAnalysis))
    db.execute(delete(Candidate))
    db.commit()
    
    # Also clear the vector store
    vector_store.clear()
    
    return {"status": "cleared", "message": "All database records and vector embeddings removed."}


# ─── Sync & Ingestion ───────────────────────────────────────────────────────

@router.post("/sync", tags=["System"])
async def sync_resumes(db: Session = Depends(get_db)):
    """
    Recursively scan UPLOAD_DIR for PDFs and TXTs not yet in the database.
    Assigns category based on parent folder name.
    """
    # 1. Get existing filenames in DB
    existing = set(db.execute(select(Candidate.filename)).scalars().all())
    
    # 2. Scan Filesystem
    added_count = 0
    # rglob for common resume extensions
    for path in UPLOAD_DIR.rglob("*"):
        if path.is_dir() or path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
            
        # Unique filename is relative to UPLOAD_DIR for our logic
        rel_path = path.relative_to(UPLOAD_DIR)
        unique_name = str(rel_path).replace("\\", "/") # standardize to forward slashes
        
        if unique_name in existing:
            continue
            
        # Determine Category (parent folder name if it's not UPLOAD_DIR itself)
        category = "General"
        if path.parent != UPLOAD_DIR:
            category = path.parent.name
            
        # Extraction
        try:
            if path.suffix.lower() == ".pdf":
                raw_text = extract_text_from_pdf(path)
            else:
                raw_text = extract_text_from_txt(path)
                
            clean = clean_text(raw_text)
            
            # Save to DB
            db_candidate = Candidate(
                filename=unique_name,
                original_name=path.name,
                raw_text=raw_text,
                clean_text=clean,
                category=category
            )
            db.add(db_candidate)
            added_count += 1
            logger.info(f"Sync discovered: {unique_name} (Category: {category})")
        except Exception as e:
            logger.error(f"Failed to ingest during sync: {path}. Error: {e}")

    db.commit()
    
    # 3. Update Vector Store with all new candidates
    if added_count > 0:
        new_candidates = db.execute(
            select(Candidate).where(Candidate.filename.notin_(existing))
        ).scalars().all()
        
        vector_store.add_resumes(
            texts=[c.clean_text for c in new_candidates],
            filenames=[c.filename for c in new_candidates]
        )
        
    return {
        "status": "success", 
        "added_count": added_count,
        "message": f"Discovered and ingested {added_count} new candidates."
    }


# ─── List Results ─────────────────────────────────────────────────────────────

@router.get("/results", tags=["Results"])
async def list_results(db: Session = Depends(get_db)):
    """List all saved analysis runs from the database."""
    analyses = db.execute(select(JobAnalysis).order_by(JobAnalysis.created_at.desc())).scalars().all()
    return {
        "result_files": [
            f"Analysis #{a.id} - {a.created_at.strftime('%Y-%m-%d %H:%M')}" 
            for a in analyses
        ],
        "analysis_ids": [a.id for a in analyses]
    }


@router.get("/results/{analysis_id}", tags=["Results"])
async def get_result(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieve the full score details for a specific analysis run."""
    analysis = db.get(JobAnalysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")
        
    scores = db.execute(
        select(RankingScore, Candidate.original_name)
        .join(Candidate, RankingScore.candidate_id == Candidate.id)
        .where(RankingScore.analysis_id == analysis_id)
        .order_by(RankingScore.rank)
    ).all()
    
    return {
        "job_description": analysis.raw_text,
        "rankings": [
            {
                "rank": s.RankingScore.rank,
                "candidate_name": s.original_name,
                "hybrid_score": s.RankingScore.hybrid_score,
                # ... other fields if needed ...
            } for s in scores
        ]
    }
@router.get("/resumes/{filename:path}", tags=["Resumes"])
async def get_resume_file(filename: str):
    """
    Serve the raw resume file (PDF or TXT) for previewing.
    Includes security checks to prevent path traversal.
    """
    # Security: Ensure the filename doesn't contain path traversal outside UPLOAD_DIR
    # BUT allow internal subdirectories (e.g. Engineering/file.pdf)
    if ".." in filename:
        raise HTTPException(status_code=400, detail="Path traversal not allowed.")
    
    # Resolve relative to UPLOAD_DIR
    file_path = (UPLOAD_DIR / filename).resolve()
    
    # Verify it is still inside UPLOAD_DIR
    if not str(file_path).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")
    
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="Resume file not found.")

    # Determine content type
    content_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    
    return FileResponse(
        path=file_path,
        media_type=content_type,
        filename=filename
    )
