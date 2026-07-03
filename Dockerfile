# ── Stage 1: Build environment ─────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Copy project files ─────────────────────────────────────────────
COPY . .

# ── Expose port ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Run FastAPI server ───────────────────────────────────────────────────────
# Models are pre-trained and included in the repo (models/*.joblib)
# No retraining needed — just start the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
