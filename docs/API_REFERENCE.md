# API Reference (FastAPI)

The API provides a set of endpoints for uploading resumes and analyzing them against a job description. Auto-generated Swagger docs are available at **http://127.0.0.1:8000/docs**.

---

## 📡 Endpoints

### 1. Health & Status
`GET /`
- **Description**: Returns the system status and confirms that ML models (spaCy, Transformers) are loaded correctly.
- **Response**:
  ```json
  {
      "status": "ok",
      "models_loaded": ["spacy", "sentence_transformers"],
      "api_version": "1.0.0"
  }
  ```

### 2. Resume Upload
`POST /upload_resume`
- **Description**: Uploads one or more PDF or text resumes to the local storage.
- **Payload**: `files` (Multipart form-data).
- **Response**:
  ```json
  {
      "message": "2 resume(s) uploaded.",
      "filenames": ["alice_chen.pdf", "bob_martinez.txt"]
  }
  ```

### 3. Analysis & Ranking
`POST /analyze`
- **Description**: Ranks all currently uploaded resumes against a specific Job Description.
- **Payload**:
  - `job_description`: (string, required)
  - `top_n`: (integer, optional, default: 10)
- **Response**:
  ```json
  {
      "job_description": "...",
      "top_candidates": [
          {
              "rank": 1,
              "candidate_name": "Alice Chen",
              "hybrid_score": 0.825,
              "matched_skills": ["Python", "SQL"],
              "missing_skills": ["AWS"],
              "extra_skills": ["Docker", "Kubernetes"]
          }
      ]
  }
  ```

### 4. Results Management
`GET /results`
- **Description**: Lists all saved JSON results in the `results/` folder.

`GET /results/{filename}`
- **Description**: Retrieves the full JSON details of a specific analysis run.

`DELETE /clear`
- **Description**: Clears the current session's uploaded resumes from memory.

---

## 🛠 Integration Example (Python)

```python
import requests

# 1. Upload
with open("resume.txt", "rb") as f:
    requests.post("http://localhost:8000/upload_resume", files={"files": f})

# 2. Analyze
resp = requests.post(
    "http://localhost:8000/analyze",
    data={"job_description": "Senior Data Scientist...", "top_n": 5}
)
print(resp.json()["top_candidates"])
```
