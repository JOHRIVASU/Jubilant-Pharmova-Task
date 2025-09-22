FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=on PIP_NO_CACHE_DIR=off PIP_ROOT_USER_ACTION=ignore
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config libgl1 libglib2.0-0 libpng-dev libjpeg-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m pip install -U pip && pip install -r requirements.txt && pip install prometheus-client
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python - <<'PY'\nimport sys, urllib.request, os\nurl = f\"http://127.0.0.1:{os.getenv('PORT','8000')}/api/v1/health\"\ntry:\n    with urllib.request.urlopen(url, timeout=3) as r:\n        sys.exit(0 if r.status==200 else 1)\nexcept Exception:\n    sys.exit(1)\nPY
ENV PORT=8000
CMD ["python","-m","uvicorn","genai_assistant_split.backend.serve:app","--host","0.0.0.0","--port","8000","--workers","1"]
