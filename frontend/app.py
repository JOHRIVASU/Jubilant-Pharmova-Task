"""
GenAI Assistant — Frontend (Clean, no optional agentic toggles)
- Talks to FastAPI backend via HTTP
- 3 steps: create session -> ingest -> chat/report
- Branding + persistent chat history
"""

import os, io, json, time
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------- Page config & branding ----------------
st.set_page_config(page_title="GenAI Assistant — Vasu Johri", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
.small-muted {font-size: 0.85rem; opacity: 0.75;}
.badge {display:inline-block; padding: 0.25rem 0.5rem; border-radius: 0.4rem; border: 1px solid #444;}
.footer {margin-top: 1.5rem; padding-top: 0.6rem; border-top: 1px dashed #444; text-align:center; opacity:0.8;}
.code-chip {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; padding: 0.15rem 0.35rem; border: 1px solid #555; border-radius: 0.35rem;}
</style>
""", unsafe_allow_html=True)

title_left, title_right = st.columns([0.8, 0.2])
with title_left:
    st.title("🩺 GenAI Assistant")
    st.caption("Document Q&A • OCR • RAG • Report Builder")
with title_right:
    st.markdown('<div class="badge">Made by <b>Vasu Johri</b></div>', unsafe_allow_html=True)

# ---------------- State ----------------
if "session_id" not in st.session_state: st.session_state.session_id = None
if "ingested" not in st.session_state: st.session_state.ingested = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []  # list of {role, content}

# ---------------- Sidebar: Backend & Session ----------------
with st.sidebar:
    st.header("Backend")
    st.text_input("Backend URL", value=BACKEND_URL, key="backend_url")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create Session", use_container_width=True):
            try:
                resp = requests.post(f"{st.session_state.backend_url}/api/v1/session")
                st.session_state.session_id = resp.json()["session_id"]
                st.session_state.ingested = False
                st.session_state.chat_history = []
                st.success("Session created.")
            except Exception as e:
                st.error(f"Failed to create session: {e}")
    with c2:
        if st.button("Health", use_container_width=True):
            try:
                h = requests.get(f"{st.session_state.backend_url}/api/v1/health").json()
                st.info(h, icon="🩺")
            except Exception as e:
                st.error(f"Health check failed: {e}")
    st.caption(f"Session: `{st.session_state.session_id or '—'}`")

# ---------------- 1) Upload & Ingest ----------------
st.subheader("1) Upload & Ingest")
uploads = st.file_uploader(
    "PDF / DOCX / CSV / XLSX / Images",
    type=["pdf","docx","csv","xlsx","xls","png","jpg","jpeg","bmp","tiff","gif"],
    accept_multiple_files=True
)

col1, col2, col3 = st.columns(3)
with col1:
    ocr_img_default = st.checkbox("OCR for Images", value=True)
with col2:
    ocr_pdf_default = st.checkbox("OCR for PDF pages (no text)", value=True)
with col3:
    ocr_docx_default = st.checkbox("OCR for DOCX embedded images", value=True)

if st.button("Ingest to Backend", type="primary"):
    if not st.session_state.session_id:
        st.warning("Create a session first (left sidebar).")
    elif not uploads:
        st.warning("Please pick at least one file.")
    else:
        files_payload = [("files", (u.name, u.getvalue(), u.type or "application/octet-stream")) for u in uploads]
        data = {
            "session_id": st.session_state.session_id,
            # agentic flags removed; only core OCR toggles sent
            "ocr_img_default": json.dumps(bool(ocr_img_default)),
            "ocr_pdf_default": json.dumps(bool(ocr_pdf_default)),
            "ocr_docx_default": json.dumps(bool(ocr_docx_default)),
        }
        with st.spinner("Indexing documents..."):
            try:
                r = requests.post(
                    f"{st.session_state.backend_url}/api/v1/ingest",
                    data=data, files=files_payload, timeout=300
                )
                if r.ok:
                    j = r.json()
                    st.session_state.ingested = True
                    st.success(f"✅ Ingested • Chunks: {j.get('chunks')} • Uploads: {j.get('uploads')}")
                    st.caption("You can move to Chat now.")
                else:
                    st.error(f"Ingest failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Ingest error: {e}")

# ---------------- 2) Chat (with history) ----------------
st.subheader("2) Chat")
st.caption("Ask questions grounded in your uploaded documents.")

# Render existing history
with st.container():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat input at the bottom
user_q = st.chat_input("Type your question…")
if user_q:
    if not st.session_state.session_id:
        st.warning("Create a session first.")
    elif not st.session_state.ingested:
        st.warning("Please ingest files first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)
        # Call backend
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    r = requests.post(
                        f"{st.session_state.backend_url}/api/v1/chat",
                        json={"session_id": st.session_state.session_id, "message": user_q}
                    )
                    if r.ok:
                        ans = r.json().get("answer", "(no answer)")
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                        st.write(ans)
                    else:
                        st.error(f"Chat failed: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"Chat error: {e}")

# ---------------- 3) Build Report ----------------
# ---------------- 3) Build Report ----------------
st.subheader("3) Build Report")
title = st.text_input("Report Title", value="Structured Document Report")
sections = st.multiselect(
    "Sections",
    ["Introduction","About","Brief","Patient Tables","Graphs","Summary"],
    default=["Introduction","About","Brief","Patient Tables","Graphs","Summary"]
)
user_goal = st.text_area(
    "Focus / Goal",
    value="Create a clean, concise report summarizing key content of the uploaded files.",
    height=90
)
temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

sec_queries = {}
for sec in sections:
    sec_queries[sec] = st.text_input(f"Query/keywords for '{sec}'", value=sec, key=f"q_{sec}")

if st.button("Generate PDF"):
    if not st.session_state.session_id:
        st.warning("Create a session first.")
    elif not st.session_state.ingested:
        st.warning("Please ingest files first.")
    else:
        payload = {
            "session_id": st.session_state.session_id,
            "title": title,
            "sections": sections,
            "sec_queries": sec_queries,
            "user_goal": user_goal,
            "temperature": float(temperature),
            # 🔑 force agentic path
            "use_agentic_plan": True
        }
        with st.spinner("Assembling report (Agentic Flow)…"):
            try:
                r = requests.post(
                    f"{st.session_state.backend_url}/api/v1/report",
                    json=payload, stream=True, timeout=600
                )
                if r.ok:
                    st.download_button("📄 Download Report PDF", data=r.content,
                                       file_name="report.pdf", mime="application/pdf")
                    st.success("Report generated (Agentic).")
                else:
                    st.error(f"Report failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Report error: {e}")

# ---------------- Footer ----------------
st.markdown(
    '<div class="footer small-muted">© GenAI Assistant • Built with FastAPI + Streamlit • Made by <b>Vasu Johri</b></div>',
    unsafe_allow_html=True
)
