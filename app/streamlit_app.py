"""
Professional Streamlit frontend — calls FastAPI only.

Usage:
    uv run streamlit run app/streamlit_app.py
"""

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"
HEALTH   = "http://localhost:8000/health"

st.set_page_config(
    page_title="FinanceRAG · AI Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Base */
.stApp { background: #000814; color: #e2eaf5; }
section[data-testid="stSidebar"] {
    background: #00091f;
    border-right: 1px solid #0a2a52;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #000d1f 0%, #001840 50%, #000d1f 100%);
    border: 1px solid #0a2a52;
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 13px;
    color: #5c8bb5;
    margin: 4px 0 0 0;
}
.header-badge {
    background: #0d2d5e;
    border: 1px solid #1565c0;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px;
    color: #64b5f6;
    font-weight: 600;
}

/* Status pills */
.status-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid;
}
.pill-green { background:#001a0d; border-color:#1b5e20; color:#69f0ae; }
.pill-blue  { background:#000d1f; border-color:#0d47a1; color:#64b5f6; }
.pill-amber { background:#1a1100; border-color:#e65100; color:#ffb74d; }
.pill-red   { background:#1a0000; border-color:#7f0000; color:#ef9a9a; }

/* Chat messages */
.stChatMessage {
    background: #00091f !important;
    border: 1px solid #0a2a52 !important;
    border-radius: 10px !important;
}

/* Chunk card */
.chunk-card {
    background: linear-gradient(160deg, #000d1f 0%, #00122e 100%);
    border: 1px solid #0a2a52;
    border-left: 4px solid #1565c0;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: all 0.2s;
    box-shadow: 0 2px 16px rgba(21,101,192,0.10);
}
.chunk-card:hover {
    border-left-color: #2196f3;
    box-shadow: 0 4px 24px rgba(33,150,243,0.22);
    transform: translateX(2px);
}
.chunk-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 6px;
}
.chunk-meta { font-size: 12px; color: #4a7fa8; display: flex; align-items: center; gap: 8px; }
.chunk-text { font-size: 14px; color: #b8d0ea; line-height: 1.75; }

/* Citation badge */
.cite-badge {
    background: #0d2d5e;
    border: 1px solid #1565c0;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 11px;
    color: #64b5f6;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Score bar */
.score-wrap { display: flex; align-items: center; gap: 8px; }
.score-bar-bg {
    background: #001433;
    border-radius: 4px;
    height: 6px;
    width: 80px;
    overflow: hidden;
    border: 1px solid #0a2a52;
}
.score-bar-fill { height: 100%; border-radius: 4px; }

/* Source tag */
.source-tag {
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
}
.tag-text  { background:#001a33; border:1px solid #0d47a1; color:#64b5f6; }
.tag-table { background:#001a1a; border:1px solid #006064; color:#4dd0e1; }
.tag-struct{ background:#1a001a; border:1px solid #4a0072; color:#ce93d8; }

/* LLM placeholder */
.llm-box {
    background: linear-gradient(135deg, #000d1f, #00122e);
    border: 1px dashed #1565c0;
    border-radius: 12px;
    padding: 26px;
    text-align: center;
    margin-bottom: 22px;
}
.llm-box .icon { font-size: 32px; margin-bottom: 8px; }
.llm-box h4 { color: #64b5f6; margin: 0 0 6px 0; font-size: 16px; }
.llm-box p  { color: #4a7fa8; font-size: 13px; margin: 0; }

/* Citations summary */
.citations-box {
    background: #000d1f;
    border: 1px solid #0a2a52;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
}
.citations-box h5 { color: #5c8bb5; font-size: 12px; text-transform: uppercase;
                    letter-spacing: 1px; margin: 0 0 10px 0; }
.cite-item { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.cite-num  { background: #0d2d5e; border: 1px solid #1565c0; border-radius: 4px;
             padding: 1px 8px; font-size: 11px; color: #64b5f6; font-weight: 700;
             min-width: 28px; text-align: center; }
.cite-info { font-size: 12px; color: #4a7fa8; }

/* Sidebar */
.sidebar-section { margin-bottom: 20px; }
.sidebar-section h4 { color: #1e88e5; font-size: 13px; text-transform: uppercase;
                       letter-spacing: 1px; margin-bottom: 12px; }
.sidebar-stat {
    background: #000d1f;
    border: 1px solid #0a2a52;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-label { font-size: 12px; color: #4a7fa8; }
.stat-value { font-size: 14px; color: #64b5f6; font-weight: 600; }

/* Inputs */
.stTextInput input, .stSelectbox select {
    background: #000d1f !important;
    border: 1px solid #0a2a52 !important;
    color: #c8dff5 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus { border-color: #1565c0 !important; }
.stSlider .stSlider { color: #1565c0 !important; }

/* Chat input */
.stChatInputContainer {
    background: #00091f !important;
    border-top: 1px solid #0a2a52 !important;
    padding: 12px !important;
}
[data-testid="stChatInput"] {
    background: #000d1f !important;
    border: 1px solid #0a2a52 !important;
    border-radius: 10px !important;
    color: #c8dff5 !important;
}

/* Divider */
hr { border-color: #0a2a52 !important; margin: 16px 0 !important; }

/* Button */
.stButton button {
    background: #0d2d5e !important;
    border: 1px solid #1565c0 !important;
    color: #64b5f6 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100%;
}
.stButton button:hover {
    background: #1565c0 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id"    not in st.session_state: st.session_state.session_id    = str(uuid.uuid4())
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "query_count"   not in st.session_state: st.session_state.query_count   = 0
if "total_chunks"  not in st.session_state: st.session_state.total_chunks  = 0

# ── Health check ──────────────────────────────────────────────────────────────
health = {}
try:
    health = requests.get(HEALTH, timeout=3).json()
except Exception:
    pass

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 FinanceRAG")
    st.caption("AI Research Assistant · v0.1")
    st.markdown("---")

    # Service status
    st.markdown('<div class="sidebar-section"><h4>Services</h4>', unsafe_allow_html=True)
    services = [
        ("PostgreSQL", health.get("postgres", False), "🗄️"),
        ("Redis",      health.get("redis",    False), "⚡"),
        ("Qdrant",     bool(health),                  "🔍"),
        ("LLM",        health.get("llm_ready",False), "🤖"),
    ]
    for name, ok, icon in services:
        label  = "Ready"   if ok else ("Coming soon" if name == "LLM" else "Offline")
        cls    = "pill-green" if ok else ("pill-amber" if name == "LLM" else "pill-red")
        dot    = "●"
        st.markdown(
            f'<div class="status-pill {cls}">{icon} {name} <span>{dot} {label}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Identity
    st.markdown('<div class="sidebar-section"><h4>Identity</h4>', unsafe_allow_html=True)
    tenant_id = st.text_input("Tenant ID", value="default_tenant", label_visibility="visible")
    user_id   = st.text_input("User ID",   value="default_user",   label_visibility="visible")
    st.caption(f"Session `{st.session_state.session_id[:12]}…`")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄  New Session"):
        st.session_state.session_id   = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.query_count  = 0
        st.session_state.total_chunks = 0
        st.rerun()

    st.markdown("---")

    # Settings
    st.markdown('<div class="sidebar-section"><h4>Retrieval</h4>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Session stats
    st.markdown('<div class="sidebar-section"><h4>Session Stats</h4>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-stat">
        <span class="stat-label">Queries</span>
        <span class="stat-value">{st.session_state.query_count}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">Chunks retrieved</span>
        <span class="stat-value">{st.session_state.total_chunks}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.caption("📖 Source: Corporate Finance — Vernimmen, 4th Ed.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <h1>📊 Corporate Finance Assistant</h1>
        <p>Powered by RAG · Semantic + Keyword Search · Multi-collection retrieval</p>
    </div>
    <div class="header-badge">● Retrieval Ready</div>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"], unsafe_allow_html=True)

# ── Query input ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about corporate finance…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.query_count += 1

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question":   question,
                        "tenant_id":  tenant_id,
                        "user_id":    user_id,
                        "session_id": st.session_state.session_id,
                        "top_k":      top_k,
                    },
                    headers={
                        "X-Tenant-Id":  tenant_id,
                        "X-User-Id":    user_id,
                        "X-Session-Id": st.session_state.session_id,
                    },
                    timeout=30,
                )
                data   = resp.json()
                chunks = data.get("chunks", [])
                cites  = data.get("citations", [])
                st.session_state.total_chunks += len(chunks)

                # LLM placeholder
                st.markdown("""
<div class="llm-box">
    <div class="icon">🤖</div>
    <h4>AI Answer</h4>
    <p>LLM will generate a synthesised answer here once the API key is configured.<br>
    Relevant sources are retrieved and ready below.</p>
</div>""", unsafe_allow_html=True)

                # Citations summary
                if cites:
                    cite_html = '<div class="citations-box"><h5>📎 Sources Retrieved</h5>'
                    for c in cites:
                        section = (c.get("section") or "—")[:55]
                        cite_html += f"""
<div class="cite-item">
    <span class="cite-num">[{c['number']}]</span>
    <span class="cite-info">Page {c.get('page','?')} &nbsp;·&nbsp; {section}</span>
</div>"""
                    cite_html += "</div>"
                    st.markdown(cite_html, unsafe_allow_html=True)

                # Chunk cards
                st.markdown(f"#### Retrieved Chunks &nbsp; <small style='color:#4a7fa8;font-weight:400'>({len(chunks)} results)</small>", unsafe_allow_html=True)

                for chunk in chunks:
                    score   = chunk["score"]
                    pct     = int(score * 100)
                    bar_col = "#29b6f6" if score >= 0.7 else "#0288d1" if score >= 0.5 else "#0d47a1"
                    page    = chunk.get("page", "?")
                    section = (chunk.get("section") or "")[:55]
                    source  = chunk.get("source", "text")
                    coll    = chunk.get("collection", "")

                    tag_cls  = "tag-table"  if source == "table" else "tag-struct" if "structure" in coll else "tag-text"
                    tag_lbl  = "📊 Table"   if source == "table" else "🗂️ Structure" if "structure" in coll else "📝 Text"

                    st.markdown(f"""
<div class="chunk-card">
    <div class="chunk-header">
        <div class="chunk-meta">
            <span class="cite-badge">[{chunk['citation_number']}]</span>
            <span>Page {page}</span>
            <span>·</span>
            <span>{section}</span>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <span class="source-tag {tag_cls}">{tag_lbl}</span>
            <div class="score-wrap">
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{pct}%;background:{bar_col};"></div>
                </div>
                <span style="font-size:12px;color:{bar_col};font-weight:600;">{score:.3f}</span>
            </div>
        </div>
    </div>
    <div class="chunk-text">{chunk['text'][:450]}{"…" if len(chunk['text']) > 450 else ""}</div>
</div>""", unsafe_allow_html=True)

                reply = f"Found **{len(chunks)} sources** for: *{question}*"
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

            except requests.exceptions.ConnectionError:
                st.error("⚠️ API server not reachable. Run: `uv run uvicorn app.main_api:app --reload --port 8000`")
            except Exception as e:
                st.error(f"Error: {e}")
