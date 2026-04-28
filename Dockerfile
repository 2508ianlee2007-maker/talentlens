# ── TalentLens Dockerfile ──────────────────────────────────────────────────
FROM python:3.11-slim

# System deps needed by faiss-cpu, sentence-transformers, and pypdf
# curl is required for the health check
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker can cache the layer
COPY requirements.txt .

# Install Python dependencies (CPU-only torch via extra index)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
# (avoids cold-start delay on first run in production)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY app.py backend.py ./

# Streamlit config — disable the "Are you sure you want to leave?" prompt
# and bind to 0.0.0.0 so Render can reach it on port 10000
RUN mkdir -p /root/.streamlit && printf '\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 10000\n\
address = "0.0.0.0"\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

EXPOSE 10000

# Health-check so Render knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:10000/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
