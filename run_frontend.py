"""
run_frontend.py
---------------
Convenience script to launch the Streamlit frontend.

Usage:
    python run_frontend.py
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", "8501"],
        check=True,
    )
