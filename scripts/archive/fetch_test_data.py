"""
scripts/fetch_test_data.py
--------------------------
Automated script to download a 'Golden Set' of professional resumes for 
accuracy benchmarking.
"""

import os
import requests
from pathlib import Path

# Target directory
UPLOADS_DIR = Path("storage/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Legit Dataset Sources (Public PDF Resumes)
# These are anonymized resumes used for NLP/ATS parsing benchmarks.
RESUMES = {
    "dev_senior_1.pdf": "https://raw.githubusercontent.com/mre/resume-samples/master/resumes/senior-developer.pdf",
    "dev_java_1.pdf":   "https://raw.githubusercontent.com/mre/resume-samples/master/resumes/java-developer.pdf",
    "dev_python_1.pdf": "https://raw.githubusercontent.com/mre/resume-samples/master/resumes/python-developer.pdf",
    "hr_manager_1.pdf":  "https://raw.githubusercontent.com/mre/resume-samples/master/resumes/hr-manager.pdf",
    "sales_exec_1.pdf":  "https://raw.githubusercontent.com/mre/resume-samples/master/resumes/sales-executive.pdf",
    "marketing_1.pdf":   "https://raw.githubusercontent.com/mre/resume-samples/master/resumes/marketing-specialist.pdf",
}

def download_resumes():
    print(f"🚀 Starting automated dataset collection to {UPLOADS_DIR}...")
    
    count = 0
    for filename, url in RESUMES.items():
        target_path = UPLOADS_DIR / filename
        
        # Skip if already exists
        if target_path.exists():
            print(f"  - {filename} already exists. Skipping.")
            continue
            
        try:
            print(f"  - Downloading {filename}...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            with open(target_path, "wb") as f:
                f.write(response.content)
            count += 1
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")

    print(f"\n✅ Finished! Downloaded {count} new resumes.")
    print(f"Total resumes in storage/uploads: {len(list(UPLOADS_DIR.glob('*')))}")

if __name__ == "__main__":
    download_resumes()
