"""
scripts/demo.py
---------------
End-to-end demonstration script — no browser, no API, just Python.

Run with:
    python scripts/demo.py

What it does:
  1. Loads sample resumes from data/sample_resumes/
  2. Uses a hardcoded sample job description
  3. Runs the full ranking pipeline
  4. Prints a formatted ranking table to the console
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_processing.pdf_extractor import extract_text_from_txt
from data_processing.text_cleaner import clean_text
from models.ranker import CandidateRanker

# ─── Sample job description ───────────────────────────────────────────────────
SAMPLE_JD = """
Senior Data Scientist

We are looking for an experienced Senior Data Scientist to join our AI team.

Requirements:
- 5+ years of experience in Machine Learning and Data Science
- Proficiency in Python, pandas, NumPy, scikit-learn
- Experience with deep learning frameworks: TensorFlow or PyTorch
- Strong knowledge of SQL and PostgreSQL
- Familiarity with cloud platforms: AWS or Google Cloud (GCP)
- Experience with model deployment and MLOps practices
- Knowledge of NLP and Natural Language Processing
- Experience with Docker and Kubernetes for containerisation
- Strong statistical and mathematical foundations
- Excellent communication and teamwork skills

Nice to have:
- Experience with Apache Spark or Kafka
- Familiarity with Airflow for pipeline orchestration
- Knowledge of Transformers / BERT / Hugging Face
- Experience with A/B testing and experimentation frameworks
"""


def load_sample_resumes() -> list[dict]:
    """Load all .txt resumes from data/sample_resumes/."""
    sample_dir = ROOT / "data" / "sample_resumes"
    if not sample_dir.exists():
        print(f"⚠  Sample resumes not found at {sample_dir}")
        print("   Run: python scripts/generate_sample_data.py  first.")
        return []

    resumes = []
    for path in sorted(sample_dir.glob("*.txt")):
        raw = extract_text_from_txt(path)
        clean = clean_text(raw)
        resumes.append({
            "filename": path.name,
            "raw_text": raw,
            "clean_text": clean,
        })
    return resumes


def print_banner(text: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def run_demo() -> None:
    print_banner("🤖 AI Resume Screening System — Demo")

    # Load resumes
    resumes = load_sample_resumes()
    if not resumes:
        return
    print(f"\n  Loaded {len(resumes)} sample resume(s)")

    # Clean JD
    jd_clean = clean_text(SAMPLE_JD)

    # Run ranker
    print("\n  Running ranking pipeline …")
    ranker = CandidateRanker()
    results = ranker.rank(
        job_description_clean=jd_clean,
        job_description_raw=SAMPLE_JD,
        resumes_clean=[r["clean_text"] for r in resumes],
        resumes_raw=[r["raw_text"]    for r in resumes],
        filenames=[r["filename"]     for r in resumes],
        top_n=10,
    )

    # ── Print results table ───────────────────────────────────────────────────
    print_banner("📊 Ranking Results")
    header = f"{'Rank':<5} {'Candidate':<22} {'Hybrid':>8} {'TF-IDF':>8} {'Semantic':>9} {'Skill%':>7}"
    print(f"\n  {header}")
    print("  " + "-" * 62)

    for c in results:
        row = (
            f"{c['rank']:<5} "
            f"{c['candidate_name']:<22} "
            f"{c['hybrid_score']:>7.1%} "
            f"{c['tfidf_score']:>8.1%} "
            f"{c['semantic_score']:>9.1%} "
            f"{c['skill_match_ratio']:>6.1%}"
        )
        print(f"  {row}")

    # ── Detail for top candidate ──────────────────────────────────────────────
    if results:
        top = results[0]
        print_banner(f"🥇 Top Candidate Detail: {top['candidate_name']}")
        print(f"\n  Hybrid Score  : {top['hybrid_score']:.4f}")
        print(f"  TF-IDF Score  : {top['tfidf_score']:.4f}")
        print(f"  Semantic Score: {top['semantic_score']:.4f}")
        print(f"  Skill Match   : {top['skill_match_ratio']:.1%}")
        print(f"\n  ✅ Matched Skills ({len(top['matched_skills'])}):")
        print(f"     {', '.join(top['matched_skills']) or 'None'}")
        print(f"\n  ❌ Missing Skills ({len(top['missing_skills'])}):")
        print(f"     {', '.join(top['missing_skills']) or 'None'}")
        print(f"\n  ➕ Extra Skills ({len(top['extra_skills'])}):")
        print(f"     {', '.join(top['extra_skills'][:10]) or 'None'}")
        kw = top["keyword_overlap"]
        print(f"\n  🔑 Keyword overlap: {kw['common_keyword_count']} common keywords")

    print("\n  ✅ Demo complete!\n")


if __name__ == "__main__":
    run_demo()
