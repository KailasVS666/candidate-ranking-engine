import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feature_engineering.skill_extractor import extract_skills

def verify_phase_2():
    print("\n--- PHASE 2: SKILL EXTRACTION VERIFICATION ---")
    
    sample_text = """
    We are looking for a candidate with expertise in Python, SQL, and Machine Learning. 
    Experience with Docker and Kubernetes is a plus. 
    Must be proficient in BERT and Transformers.
    """
    
    print(f"Sample Text:\n{sample_text.strip()}")
    
    # 1. Test Hybrid Extraction
    skills = extract_skills(sample_text, method="hybrid")
    
    print("\nExtracted Skills (Hybrid):")
    print(sorted(list(skills)))
    
    # Simple checks
    expected_skills = {"Python", "SQL", "Machine Learning", "Docker", "Kubernetes", "BERT", "Transformers"}
    found_expected = expected_skills.intersection(skills)
    
    print(f"\nFound {len(found_expected)} out of {len(expected_skills)} expected keywords.")
    
    assert "Python" in skills, "Failed to extract Python"
    assert "BERT" in skills, "Failed to extract BERT"
    
    print("\n✅ Phase 2: Skill extraction successfully verified keyword matching and hybrid logic.")

if __name__ == "__main__":
    verify_phase_2()
