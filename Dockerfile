# Portable image for the FastAPI backend (api.py).
# Works on any container host: Render, Railway, Google Cloud Run, Fly.io,
# Hugging Face Docker Spaces, etc. Torch is intentionally excluded
# (retrieval.mode: id), so this image stays small and builds fast on free tiers.

FROM python:3.12-slim

# Avoid .pyc files and buffer issues in containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Hosts inject $PORT at runtime; bind to it. Shell form so $PORT expands.
# Single worker keeps the in-memory session store coherent (see README:
# use Redis + multiple workers for real production scale).
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
