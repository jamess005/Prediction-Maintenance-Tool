FROM python:3.12-slim

WORKDIR /app

# System deps for scikit-learn / xgboost wheel builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source code and data
COPY src/ src/
COPY data/ data/

# Create output directories
RUN mkdir -p outputs/models outputs/figures outputs/reports

# Copy pre-trained model if available (optional — can retrain via API)
COPY outputs/models/ outputs/models/

# Expose the API port
EXPOSE 8000

ENV PYTHONPATH=/app/src

# Run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
