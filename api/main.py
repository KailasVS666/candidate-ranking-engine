"""
api/main.py
-----------
FastAPI application entry point.

Start the server with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
or run the convenience script:
    python run_api.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import API_HOST, API_PORT
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Application factory ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="AI Resume Screening & Candidate Ranking API",
        description=(
            "Production-quality resume screening system using classical NLP "
            "(TF-IDF, Sentence-Transformers, spaCy). No external LLM APIs."
        ),
        version="1.0.0",
        contact={
            "name": "AI Screening Team",
            "url": "https://github.com/your-repo/candidate-ranking-engine",
        },
        license_info={"name": "MIT"},
    )

    # ── CORS (allow Streamlit frontend on any port during development) ─────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # Restrict in production!
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register routes ────────────────────────────────────────────────────────
    app.include_router(router)

    @app.on_event("startup")
    async def on_startup():
        logger.info(f"🚀 API server starting on {API_HOST}:{API_PORT}")

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("API server shutting down.")

    return app


app = create_app()
