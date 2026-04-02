"""
app.py — Property Valuation Tool: Streamlit web UI.

Run with: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import base64
import streamlit as st
from config import get_config
from src.chat.engine import ChatEngine
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.registry import DocumentRegistry
from src.indexing.vector_store import get_vector_store

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Valuation Intelligence · Real Estate Co",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Hide Streamlit chrome (preserve sidebar toggle) ── */
    #MainMenu                          { visibility: hidden; }
    footer                             { visibility: hidden; }
    [data-testid="stHeader"]           { display: none !important; }
    .stDeployButton                    { display: none !important; }
    [data-testid="stToolbar"]          { display: none !important; }
    /* Keep the collapsed-sidebar arrow always reachable */
    [data-testid="collapsedControl"]   { visibility: visible !important; display: flex !important; }

    /* ── App background ───────────────────────────────────── */
    .stApp { background: #F1F4F8; }
    .main .block-container {
        padding: 2rem 2.5rem 4rem 2.5rem;
        max-width: 900px;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #1F2937 !important;
    }
    [data-testid="stSidebar"] * { color: #9CA3AF !important; }
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] b         { color: #F9FAFB !important; }
    [data-testid="stSidebar"] hr        { border-color: #1F2937 !important; }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #F9FAFB !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #1F2937 !important;
        border: 1px solid #374151 !important;
        color: #E5E7EB !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] .stCheckbox label { color: #D1D5DB !important; font-size: 0.82rem !important; }
    [data-testid="stSidebar"] .stCheckbox span  { color: #D1D5DB !important; }

    [data-testid="stSidebar"] .stButton > button {
        background: #1F2937 !important;
        color: #E5E7EB !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: all 0.15s;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #2563EB !important;
        border-color: #2563EB !important;
        color: #ffffff !important;
    }

    /* ── Chat messages ────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 2px 0 !important;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
        background: #1E3A5F !important;
        color: #EFF6FF !important;
        border-radius: 14px 14px 4px 14px !important;
        padding: 12px 18px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { color: #EFF6FF !important; }

    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px 14px 14px 4px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) span,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) strong {
        color: #0F172A !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"]:focus-within {
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }

    /* ── Expanders ────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
        margin: 4px 0 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        color: #1E3A5F !important;
        padding: 10px 14px !important;
    }
    [data-testid="stExpander"] summary:hover { background: #F8FAFC !important; }

    /* ── Confidence pill ─────────────────────────────────── */
    .conf-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.03em; margin: 8px 0 12px 0;
    }
    .conf-high   { background:#DCFCE7; color:#166534; border:1px solid #86EFAC; }
    .conf-medium { background:#FEF9C3; color:#854D0E; border:1px solid #FDE047; }
    .conf-low    { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
    .conf-none   { background:#F1F5F9; color:#475569; border:1px solid #CBD5E1; }

    /* ── Follow-ups ──────────────────────────────────────── */
    .followup-wrap {
        margin-top: 12px; padding: 12px 16px;
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    .followup-label {
        font-size: 0.68rem; font-weight: 700; color: #94A3B8;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
    }
    .followup-item { font-size: 0.82rem; color: #1E40AF; padding: 2px 0; }
    .followup-item::before { content: "→ "; color: #2563EB; font-weight: 600; }

    /* ── Starter cards ───────────────────────────────────── */
    div[data-testid="column"] .stButton > button {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #1E3A5F !important;
        font-size: 0.83rem !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 12px 16px !important;
        line-height: 1.45 !important;
        height: auto !important;
        white-space: normal !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        transition: all 0.15s !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        border-color: #2563EB !important;
        background: #EFF6FF !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.12) !important;
        color: #1E3A5F !important;
    }

    /* ── Debug chunk card ────────────────────────────────── */
    .chunk-card {
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 6px; padding: 10px 13px;
        font-size: 0.77rem; font-family: 'SFMono-Regular', Consolas, monospace;
        white-space: pre-wrap; word-break: break-word;
        color: #334155; line-height: 1.55; margin-bottom: 8px;
    }

    /* ── Dividers ────────────────────────────────────────── */
    hr { border-color: #E2E8F0 !important; margin: 16px 0 !important; }

    /* ── Spinner ─────────────────────────────────────────── */
    .stSpinner > div { border-top-color: #2563EB !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    for k, v in {"chat_history": [], "turns": [], "engine": None, "config": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def _load_engine():
    cfg = get_config()
    try:
        return ChatEngine(cfg), cfg, None
    except Exception as e:
        return None, cfg, str(e)


@st.cache_data(show_spinner=False)
def _load_pdf_b64(filename: str):
    pdf_path = os.path.join("data", "pdfs", filename)
    if not os.path.isfile(pdf_path):
        return None
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── UI helpers ────────────────────────────────────────────────────────────────

_CONF = {
    "high":                  ("conf-high",   "●", "High Confidence"),
    "medium":                ("conf-medium", "◑", "Medium Confidence"),
    "low":                   ("conf-low",    "○", "Low Confidence"),
    "insufficient_evidence": ("conf-none",   "✕", "Insufficient Evidence"),
}

def _confidence_html(level: str) -> str:
    cls, dot, label = _CONF.get(level.lower(), ("conf-none", "◑", level))
    return f'<span class="conf-pill {cls}">{dot}&ensp;{label}</span>'


def _render_citations(citations: list):
    if not citations:
        return

    with st.expander(f"📎  Source Evidence — {len(citations)} chunk(s)", expanded=False):
        tabs = st.tabs([f"E{c['evidence_num']}" for c in citations])
        for tab, c in zip(tabs, citations):
            with tab:
                type_lbl   = "▦ Table" if c["chunk_type"] == "table" else "¶ Narrative"
                type_color = "#7C3AED" if c["chunk_type"] == "table" else "#059669"
                score      = f"{c['rerank_score']:.3f}" if c.get("rerank_score") else "—"
                section    = c.get("section", "—") or "—"
                full_text  = c.get("full_text", c.get("preview", ""))
                is_table   = c["chunk_type"] == "table"
                accent     = "#7C3AED" if is_table else "#2563EB"
                bg         = "#F5F3FF" if is_table else "#EFF6FF"

                st.markdown(
                    f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;'
                    f'font-size:0.76rem;color:#64748B">'
                    f'<span><b style="color:#0F172A">File</b>&nbsp;{c["filename"]}</span>'
                    f'<span><b style="color:#0F172A">Page</b>&nbsp;{c["page"]}</span>'
                    f'<span style="color:{type_color};font-weight:600">{type_lbl}</span>'
                    f'<span><b style="color:#0F172A">Score</b>&nbsp;{score}</span>'
                    f'<span><b style="color:#0F172A">Section</b>&nbsp;{section}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p style="font-size:0.68rem;font-weight:700;color:#94A3B8;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 5px 0">Excerpt</p>'
                    f'<div style="background:{bg};border-left:3px solid {accent};'
                    f'border-radius:0 6px 6px 0;padding:10px 14px;font-size:0.81rem;'
                    f'line-height:1.7;color:#1E293B;white-space:pre-wrap;'
                    f'word-break:break-word;margin-bottom:14px">{full_text}</div>',
                    unsafe_allow_html=True,
                )
                pdf_b64 = _load_pdf_b64(c["filename"])
                if pdf_b64:
                    page_num = c.get("page", 1)
                    pdf_src  = f"data:application/pdf;base64,{pdf_b64}#page={page_num}"
                    st.markdown(
                        f'<p style="font-size:0.68rem;font-weight:700;color:#94A3B8;'
                        f'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 5px 0">'
                        f'Source Document — page {page_num}</p>'
                        f'<iframe src="{pdf_src}" width="100%" height="500px" '
                        f'style="border:1px solid #E2E8F0;border-radius:8px;display:block">'
                        f'</iframe>',
                        unsafe_allow_html=True,
                    )

        st.markdown(
            '<p style="font-size:0.68rem;color:#94A3B8;margin-top:10px">'
            'Answers are grounded exclusively in retrieved evidence. '
            'No external knowledge is used.</p>',
            unsafe_allow_html=True,
        )


def _render_debug(chunks: list):
    if not chunks:
        return
    with st.expander("🔬  Retrieval trace — full reranked context", expanded=False):
        st.caption("Chunks in reranked order. This is the exact context passed to the model.")
        for i, c in enumerate(chunks, 1):
            score = f"  rerank={c.rerank_score:.3f}" if c.rerank_score is not None else ""
            heading = f"  ·  §{c.section_heading}" if c.section_heading else ""
            st.markdown(
                f'<div style="font-size:0.75rem;font-weight:600;color:#1E3A5F;margin:8px 0 3px 0">'
                f'[{i}] {c.source_filename} · p.{c.page_num} · {c.chunk_type}{score}{heading}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="chunk-card">{c.text[:600]}</div>', unsafe_allow_html=True)


def _render_turn(turn: dict):
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        result = turn["result"]
        st.markdown(result.answer)
        st.markdown(_confidence_html(result.confidence), unsafe_allow_html=True)
        _render_citations(result.citations)
        _render_debug(turn.get("chunks", []))
        if result.follow_up_questions:
            fqs = "".join(f'<div class="followup-item">{fq}</div>' for fq in result.follow_up_questions)
            st.markdown(
                f'<div class="followup-wrap"><div class="followup-label">Suggested follow-ups</div>{fqs}</div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="padding:8px 0 16px 0">'
        '<div style="font-size:0.6rem;font-weight:700;color:#4B5563;letter-spacing:0.12em;text-transform:uppercase">Real Estate Co</div>'
        '<div style="font-size:1.05rem;font-weight:700;color:#F9FAFB;line-height:1.3;margin-top:2px">Valuation Intelligence</div>'
        '<div style="font-size:0.7rem;color:#4B5563;margin-top:3px">Hybrid RAG · Grounded Answers</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    engine, config, load_error = _load_engine()
    st.session_state.engine = engine
    st.session_state.config = config

    if load_error:
        st.error(f"Engine error: {load_error}")

    # Knowledge base status
    st.markdown("**Knowledge Base**")
    try:
        registry    = DocumentRegistry(config.registry_path, gcs_bucket=config.gcs_bucket)
        summary     = registry.summary()
        vs          = get_vector_store(config)
        chunk_count = vs.count()
        c1, c2 = st.columns(2)
        c1.metric("Documents", summary.get("indexed", 0))
        c2.metric("Chunks", chunk_count)
        if summary.get("failed", 0) > 0:
            st.warning(f"{summary['failed']} doc(s) failed ingestion.")
        if chunk_count == 0:
            st.warning("No chunks indexed. Run ingestion first.")
    except Exception as e:
        st.error(f"Status error: {e}")

    st.divider()

    # Ingestion controls
    st.markdown("**Document Ingestion**")
    force_cb = st.checkbox("Force reindex all", value=False)
    if st.button("↻  Re-ingest Documents", use_container_width=True):
        lines: list = []
        placeholder = st.empty()
        def _cb(msg):
            lines.append(msg)
            placeholder.text("\n".join(lines[-10:]))
        with st.spinner("Running ingestion pipeline..."):
            try:
                pipeline = IngestionPipeline(config, progress_callback=_cb)
                res = pipeline.run(force_reindex=force_cb)
                st.success(f"Done — {res['processed']} processed, {res['skipped']} skipped, {res['failed']} failed")
            except Exception as e:
                st.error(f"Ingestion error: {e}")
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # Retrieval filter
    st.markdown("**Retrieval Scope**")
    filter_choice = st.selectbox(
        "Content type",
        ["All documents", "Tables only", "Text only"],
        label_visibility="collapsed",
    )
    metadata_filter = None
    if filter_choice == "Tables only":
        metadata_filter = {"chunk_type": "table"}
    elif filter_choice == "Text only":
        metadata_filter = {"chunk_type": "text"}

    st.divider()
    if st.button("⊘  Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.turns = []
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────

# Header
st.markdown(
    '<div style="display:flex;align-items:center;justify-content:space-between;'
    'padding-bottom:20px;border-bottom:1px solid #E2E8F0;margin-bottom:24px">'
    '  <div>'
    '    <div style="font-size:0.65rem;font-weight:700;color:#94A3B8;letter-spacing:0.1em;text-transform:uppercase">Real Estate Co</div>'
    '    <div style="font-size:1.4rem;font-weight:700;color:#0F172A;line-height:1.2">Property Valuation Intelligence</div>'
    '  </div>'
    '  <div style="font-size:0.7rem;font-weight:600;color:#2563EB;background:#EFF6FF;'
    '  padding:5px 14px;border-radius:999px;border:1px solid #BFDBFE">'
    '  Grounded · Cited · Evidence-Backed</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Starter questions
if not st.session_state.turns:
    st.markdown(
        '<p style="font-size:0.7rem;font-weight:700;color:#94A3B8;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:10px">Suggested questions</p>',
        unsafe_allow_html=True,
    )
    _starters = [
        "What is the appraised value and cap rate for Rivergate Apartments?",
        "Which comparable sales support the 5.35% cap rate?",
        "What is the DSCR and LTV for the Rivergate loan?",
        "What are the key risk factors for the Rivergate property?",
        "What capital expenditures are recommended in the next 24 months?",
        "How far below market are the current in-place rents?",
    ]
    c1, c2 = st.columns(2)
    for i, q in enumerate(_starters):
        col = c1 if i % 2 == 0 else c2
        if col.button(q, key=f"s_{i}", use_container_width=True):
            st.session_state["_pending"] = q
            st.rerun()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

st.divider()

# Conversation history
for turn in st.session_state.turns:
    _render_turn(turn)

# ── Input + query handler ─────────────────────────────────────────────────────

pending     = st.session_state.pop("_pending", None)
user_input  = st.chat_input("Ask about cap rates, NOI, rent rolls, comps, underwriting metrics…")
active_query = pending or user_input

if active_query:
    _engine = st.session_state.engine
    _config = st.session_state.config

    if _engine is None:
        st.error("Engine not loaded — check your .env file and refresh.")
        st.stop()

    try:
        if get_vector_store(_config).count() == 0:
            st.warning("No documents indexed. Click **Re-ingest Documents** in the sidebar.")
            st.stop()
    except Exception:
        pass

    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating answer…"):
            try:
                result, top_chunks = _engine.ask_with_chunks(
                    query=active_query,
                    chat_history=st.session_state.chat_history,
                    metadata_filter=metadata_filter,
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.markdown(result.answer)
        st.markdown(_confidence_html(result.confidence), unsafe_allow_html=True)
        _render_citations(result.citations)
        _render_debug(top_chunks)

        if result.follow_up_questions:
            fqs = "".join(f'<div class="followup-item">{fq}</div>' for fq in result.follow_up_questions)
            st.markdown(
                f'<div class="followup-wrap"><div class="followup-label">Suggested follow-ups</div>{fqs}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.chat_history.append({"role": "user",      "content": active_query})
    st.session_state.chat_history.append({"role": "assistant", "content": result.answer})
    st.session_state.turns.append({"query": active_query, "result": result, "chunks": top_chunks})
    st.rerun()
