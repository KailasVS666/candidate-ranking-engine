import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.tfidf_scorer import compute_tfidf_scores
from models.semantic_scorer import compute_semantic_scores
from data_processing.text_cleaner import clean_text

def verify_phase_3():
    print("\n--- PHASE 3: SCORING MODEL VERIFICATION ---")
    
    jd_raw = "Looking for a Data Scientist with experience in Machine Learning and Python."
    resume_raw_1 = "Experience as a Data Scientist building Machine Learning models in Python."
    resume_raw_2 = "Web developer proficient in React and Node.js."
    
    # 1. Prepare data
    jd_clean = clean_text(jd_raw)
    res_clean = [clean_text(resume_raw_1), clean_text(resume_raw_2)]
    
    # 2. Test TF-IDF Scorer
    print("\n[TF-IDF SCORER TEST]")
    tfidf_scores = compute_tfidf_scores(jd_clean, res_clean)
    print(f"Scores: {tfidf_scores}")
    
    assert tfidf_scores[0] > tfidf_scores[1], "TF-IDF score logic failed: relevant resume should score higher"
    print("✅ TF-IDF Scoring matches keyword relevance.")
    
    # 3. Test Semantic Scorer
    print("\n[SEMANTIC SCORER TEST]")
    semantic_scores = compute_semantic_scores(jd_raw, [resume_raw_1, resume_raw_2])
    print(f"Scores: {semantic_scores}")
    
    assert semantic_scores[0] > semantic_scores[1], "Semantic score logic failed: relevant resume should score higher"
    print("✅ Semantic Scoring captures meaning correctly.")
    
    print("\n✅ Phase 3: Both scoring models (TF-IDF and Semantic) successfully verified.")

if __name__ == "__main__":
    verify_phase_3()
