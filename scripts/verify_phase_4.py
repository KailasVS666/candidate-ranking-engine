import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ranker import CandidateRanker
from data_processing.text_cleaner import clean_text

def verify_phase_4():
    print("\n--- PHASE 4: RANKING ORCHESTRATION VERIFICATION ---")
    
    jd_raw = "Senior Data Scientist proficient in Python, SQL, and Machine Learning."
    resume_raw_1 = "Senior Data Scientist with expertise in Python, SQL, and Deep Learning models."
    resume_raw_2 = "Junior Data Analyst with Python and SQL basics."
    
    # Pre-process
    jd_clean = clean_text(jd_raw)
    res_raw = [resume_raw_1, resume_raw_2]
    res_clean = [clean_text(r) for r in res_raw]
    filenames = ["alice_senior.txt", "bob_junior.txt"]
    
    # 1. Run Master Ranker
    ranker = CandidateRanker()
    results = ranker.rank(
        job_description_clean=jd_clean,
        job_description_raw=jd_raw,
        resumes_clean=res_clean,
        resumes_raw=res_raw,
        filenames=filenames,
        top_n=5
    )
    
    print("\nRanking Results:")
    for res in results:
        print(f"Rank {res['rank']}: {res['candidate_name']} | Hybrid Score: {res['hybrid_score']}")
    
    # Simple checks
    assert results[0]['candidate_name'] == "Alice Senior", "Ranking failed: Alice should be #1"
    assert results[0]['hybrid_score'] > results[1]['hybrid_score'], "Hybrid score should be descending"
    
    print("\n✅ Phase 4: Ranking Orchestration (Hybrid Scoring + Sorting) successfully verified.")

if __name__ == "__main__":
    verify_phase_4()
