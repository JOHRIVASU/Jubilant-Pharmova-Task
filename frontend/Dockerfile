FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=on PIP_NO_CACHE_DIR=off PIP_ROOT_USER_ACTION=ignore
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m pip install -U pip && pip install -r requirements.txt
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python - <<'PY'\nimport sys, urllib.request, os\nurl = f\"http://127.0.0.1:{os.getenv('PORT','8501')}\"\ntry:\n    with urllib.request.urlopen(url, timeout=3) as r:\n        sys.exit(0 if r.status==200 else 1)\nexcept Exception:\n    sys.exit(1)\nPY
ENV PORT=8501
CMD ["streamlit","run","genai_assistant_split/frontend/app.py","--server.port=8501","--server.address=0.0.0.0"]
