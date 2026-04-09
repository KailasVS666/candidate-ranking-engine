# Use a lightweight Python base image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLP models (one-time step)
COPY scripts/setup_nlp_models.py ./scripts/
RUN python scripts/setup_nlp_models.py

# Copy project files
COPY . .

# Create storage directories
RUN mkdir -p uploads processed results logs

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command (will be overridden by docker-compose)
CMD ["python", "run_api.py"]
