# GenAI Assistant — Split Frontend/Backend

## Layout
```
genai_assistant_split/
├─ requirements.txt
├─ backend/
│  └─ main.py # FastAPI app (uvicorn backend.main:api --reload)
   └─ server.py
   └─ docker 
       
└─ frontend/
   └─ app.py # Streamlit UI (streamlit run frontend/app.py)
   └─ docker  
```

## Quickstart

1) Create a Python venv (recommended) and install deps:
```
pip install -r requirements.txt
```

2) Create a `.env` at the **project root** (same level as requirements.txt) with:
```
OPENAI_API_KEY=sk-...           # your key
BACKEND_URL=http://localhost:8000
```

3) Run backend:
```
uvicorn backend.main:api --reload --port 8000
```

4) Run frontend (in another terminal):
```
streamlit run frontend/app.py
```

5) In the Streamlit UI:
- Click **Create Session** (left sidebar)
- Upload files and click **Ingest to Backend**
- Ask questions in **Chat Section**
- Build an **Agentic Report** and download the PDF

