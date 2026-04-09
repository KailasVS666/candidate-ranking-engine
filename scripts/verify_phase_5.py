import requests
import io

def verify_phase_5():
    print("\n--- PHASE 5: API ENDPOINT VERIFICATION ---")
    
    BASE_URL = "http://127.0.0.1:8000"
    
    # 1. Health Check
    print("\n[GET / health check]")
    try:
        resp = requests.get(f"{BASE_URL}/")
        print(f"Status: {resp.status_code}")
        print(f"Body: {resp.json()}")
        assert resp.status_code == 200, "Health check failed"
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        return

    # 2. Upload Resume
    print("\n[POST /upload_resume]")
    resume_content = "Alice Alpha. Data Scientist. Expert in Python and ML."
    files = {"files": ("alice.txt", io.StringIO(resume_content), "text/plain")}
    resp = requests.post(f"{BASE_URL}/upload_resume", files=files)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.json()}")
    assert resp.status_code == 200, "Upload failed"

    # 3. Analyze
    print("\n[POST /analyze]")
    payload = {
        "job_description": "We need a Data Scientist with Python and ML skills.",
        "top_n": 5
    }
    resp = requests.post(f"{BASE_URL}/analyze", data=payload)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Top Candidate: {data['top_candidates'][0]['candidate_name']}")
    print(f"Hybrid Score: {data['top_candidates'][0]['hybrid_score']}")
    
    assert resp.status_code == 200, "Analysis failed"
    assert len(data['top_candidates']) > 0, "No candidates returned"
    
    print("\n✅ Phase 5: API endpoints (Health, Upload, Analyze) successfully verified.")

if __name__ == "__main__":
    verify_phase_5()
