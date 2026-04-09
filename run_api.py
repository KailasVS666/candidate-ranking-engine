"""
run_api.py
----------
Convenience script to launch the FastAPI server.

Usage:
    python run_api.py
"""

import uvicorn
from config.settings import API_HOST, API_PORT, API_RELOAD

if __name__ == "__main__":
    print(f"Starting API server at http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level="info",
    )
