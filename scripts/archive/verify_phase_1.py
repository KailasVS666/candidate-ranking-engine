import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_processing.text_cleaner import clean_text

def verify_phase_1():
    print("\n--- PHASE 1: TEXT PROCESSING VERIFICATION ---")
    
    sample_text = """
    Alice Chen (alice.chen@example.com) 
    Senior Data Scientist | https://linkedin.com/in/alice
    Expert in Python, Machine Learning, and NLP. 
    Worked with Résumés and café-style data!
    """
    
    print(f"Original Text:\n{sample_text.strip()}")
    
    cleaned = clean_text(sample_text)
    
    print("\nCleaned Text:")
    print(cleaned)
    
    # Simple checks
    assert "@" not in cleaned, "Email not removed"
    assert "https" not in cleaned, "URL not removed"
    assert cleaned == cleaned.lower(), "Not lowercased"
    assert "resume" in cleaned or "resumes" in cleaned, "Unicode mapping failed"
    
    print("\n✅ Phase 1: Text processing successfully verified normalization, URL/Email removal, and cleaning.")

if __name__ == "__main__":
    verify_phase_1()
