"""
app.py — Property Valuation Tool: Streamlit web UI.

Run with: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from config import get_config
from src.chat.engine import ChatEngine
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.registry import DocumentRegistry
from src.indexing.vector_store import VectorStore

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Real Estate Co · Valuation Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global reset ─────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide all Streamlit chrome */
    #MainMenu        { visibility: hidden; }
    footer           { visibility: hidden; }
    header           { visibility: hidden; }
    .stDeployButton  { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Page background ──────────────────────────────────── */
    .stApp {
        background: #f4f6f9;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0d1b2a !important;
        border-right: 1px solid #1a2d42;
    }
    [data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1e3048 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #7a99b8 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #c8d6e5 !important;
        font-size: 0.83rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #1a2d42 !important;
        border: 1px solid #2a4060 !important;
        color: #e0eaf4 !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: #1a3a5c !important;
        color: #e0eaf4 !important;
        border: 1px solid #2a5080 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1e4d78 !important;
        border-color: #3a7abf !important;
    }

    /* ── Brand header ─────────────────────────────────────── */
    .rec-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 24px 14px 24px;
        background: #ffffff;
        border-bottom: 1px solid #e0e6ef;
        margin: -1rem -1rem 0 -1rem;
    }
    .rec-logo {
        width: 38px; height: 38px;
        background: #0d1b2a;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; line-height: 1;
        flex-shrink: 0;
    }
    .rec-brand { display: flex; flex-direction: column; }
    .rec-company {
        font-size: 0.7rem; font-weight: 600;
        color: #7a8fa6; letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .rec-product {
        font-size: 1.15rem; font-weight: 700;
        color: #0d1b2a; line-height: 1.2;
    }
    .rec-divider {
        flex: 1;
    }
    .rec-badge {
        font-size: 0.68rem; font-weight: 500;
        color: #3a7abf;
        background: #e8f1fb;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid #c0d8f5;
    }

    /* ── Starter question cards ───────────────────────────── */
    .starter-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 12px 0 20px 0;
    }
    .starter-label {
        font-size: 0.75rem; font-weight: 600;
        color: #7a8fa6; letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 4px;
    }

    /* Override starter buttons to look like cards */
    div[data-testid="column"] .stButton > button {
        background: #ffffff !important;
        border: 1px solid #dde4ef !important;
        border-radius: 8px !important;
        color: #1a2d42 !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 10px 14px !important;
        line-height: 1.4 !important;
        height: auto !important;
        white-space: normal !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: all 0.15s !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        border-color: #3a7abf !important;
        background: #f0f6ff !important;
        box-shadow: 0 2px 8px rgba(58,122,191,0.12) !important;
    }

    /* ── Chat messages ────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 4px 0 !important;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
        background: #0d1b2a !important;
        color: #f0f4f8 !important;
        border-radius: 12px 12px 4px 12px !important;
        padding: 12px 16px !important;
    }

    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
        background: #ffffff !important;
        border: 1px solid #e0e6ef !important;
        border-radius: 12px 12px 12px 4px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #c8d4e4 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #3a7abf !important;
        box-shadow: 0 0 0 3px rgba(58,122,191,0.12) !important;
    }

    /* ── Confidence badge ─────────────────────────────────── */
    .conf-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin: 6px 0 10px 0;
    }
    .conf-high   { background: #e6f4ea; color: #1e6832; border: 1px solid #b7dfc2; }
    .conf-medium { background: #fef9e7; color: #7d5a00; border: 1px solid #f0d88a; }
    .conf-low    { background: #fdecea; color: #8b1a1a; border: 1px solid #f5b8b8; }
    .conf-none   { background: #f0f2f5; color: #4a5568; border: 1px solid #d0d6e0; }

    /* ── Citation cards ───────────────────────────────────── */
    .cite-wrap { margin-top: 4px; }
    .cite-card {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        background: #f8fafd;
        border: 1px solid #dde4ef;
        border-left: 3px solid #3a7abf;
        border-radius: 0 6px 6px 0;
        padding: 8px 12px;
        margin: 5px 0;
        font-size: 0.8rem;
        line-height: 1.5;
    }
    .cite-num {
        font-size: 0.68rem; font-weight: 700;
        color: #ffffff; background: #3a7abf;
        border-radius: 4px;
        padding: 1px 6px;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .cite-body { flex: 1; }
    .cite-meta {
        font-weight: 600; color: #1a2d42;
        display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
    }
    .cite-type-table { color: #6d28d9; font-weight: 500; font-size: 0.72rem; }
    .cite-type-text  { color: #1e6832; font-weight: 500; font-size: 0.72rem; }
    .cite-score { color: #7a8fa6; font-size: 0.72rem; }
    .cite-section { color: #7a8fa6; font-style: italic; font-size: 0.75rem; margin-top: 1px; }
    .cite-preview { color: #4a5568; font-size: 0.77rem; margin-top: 3px; }
    .cite-footer {
        font-size: 0.71rem; color: #9aabb8;
        border-top: 1px solid #e8edf4;
        padding-top: 6px; margin-top: 6px;
    }

    /* ── Debug chunk panel ────────────────────────────────── */
    .chunk-header {
        font-size: 0.78rem; font-weight: 600;
        color: #1a2d42; margin: 8px 0 3px 0;
    }
    .chunk-card {
        background: #f0f2f6;
        border: 1px solid #d4dae6;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.76rem;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        white-space: pre-wrap;
        word-break: break-word;
        color: #2d3748;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    /* ── Expanders ────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid #dde4ef !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        margin-top: 6px !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #1a2d42 !important;
    }

    /* ── Divider ──────────────────────────────────────────── */
    hr { border-color: #e0e6ef !important; margin: 12px 0 !important; }

    /* ── Sidebar brand block ──────────────────────────────── */
    .sb-brand {
        padding: 4px 0 12px 0;
    }
    .sb-company {
        font-size: 0.62rem !important; font-weight: 700 !important;
        color: #5a7fa0 !important; letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .sb-product {
        font-size: 1.0rem !important; font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1.25;
    }
    .sb-tagline {
        font-size: 0.71rem !important; color: #5a7fa0 !important;
        margin-top: 2px;
    }

    /* ── Follow-up suggestions ────────────────────────────── */
    .followup-wrap {
        margin-top: 10px;
        padding: 10px 14px;
        background: #f8fafd;
        border: 1px solid #dde4ef;
        border-radius: 8px;
    }
    .followup-label {
        font-size: 0.7rem; font-weight: 600;
        color: #7a8fa6; letter-spacing: 0.07em;
        text-transform: uppercase; margin-bottom: 5px;
    }
    .followup-item {
        font-size: 0.8rem; color: #2d5a8e;
        padding: 2px 0;
    }
    .followup-item::before { content: "→ "; color: #3a7abf; font-weight: 600; }

    /* ── Spinner ──────────────────────────────────────────── */
    .stSpinner > div { border-top-color: #3a7abf !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "chat_history": [],
        "turns": [],
        "engine": None,
        "config": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Engine loader (cached across reruns) ──────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def _load_engine():
    cfg = get_config()
    try:
        engine = ChatEngine(cfg)
        return engine, cfg, None
    except Exception as e:
        return None, cfg, str(e)


# ── Helpers ───────────────────────────────────────────────────────────────────

_CONF = {
    "high":                 ("conf-high",   "●", "High Confidence"),
    "medium":               ("conf-medium", "◑", "Medium Confidence"),
    "low":                  ("conf-low",    "○", "Low Confidence"),
    "insufficient_evidence":("conf-none",   "✕", "Insufficient Evidence"),
}

def _confidence_html(level: str) -> str:
    cls, dot, label = _CONF.get(level.lower(), ("conf-none", "◑", level))
    return f'<span class="conf-pill {cls}">{dot}&ensp;{label}</span>'


def _render_citations(citations: list, expanded: bool = False):
    if not citations:
        return
    with st.expander(f"📎  Source Evidence — {len(citations)} chunk(s) retrieved", expanded=expanded):
        st.markdown('<div class="cite-wrap">', unsafe_allow_html=True)
        for c in citations:
            type_cls  = "cite-type-table" if c["chunk_type"] == "table" else "cite-type-text"
            type_lbl  = "▦ Table" if c["chunk_type"] == "table" else "¶ Text"
            score_str = f'<span class="cite-score">· {c["rerank_score"]:.3f}</span>' if c.get("rerank_score") else ""
            section   = f'<div class="cite-section">{c["section"]}</div>' if c["section"] and c["section"] != "—" else ""
            st.markdown(
                f'<div class="cite-card">'
                f'  <div class="cite-num">E{c["evidence_num"]}</div>'
                f'  <div class="cite-body">'
                f'    <div class="cite-meta">'
                f'      <span>{c["filename"]}</span>'
                f'      <span>· p.{c["page"]}</span>'
                f'      <span class="{type_cls}">{type_lbl}</span>'
                f'      {score_str}'
                f'    </div>'
                f'    {section}'
                f'    <div class="cite-preview">{c["preview"]}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="cite-footer">Answers are grounded exclusively in the evidence above. '
            'No external knowledge is used.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def _render_debug(chunks: list):
    if not chunks:
        return
    with st.expander("🔬  Retrieval Trace — reranked context sent to model", expanded=False):
        st.caption(
            "Chunks shown in reranked order (highest relevance first). "
            "This is the exact context the model used to construct its answer."
        )
        for i, c in enumerate(chunks, 1):
            score_str = f"  rerank={c.rerank_score:.3f}" if c.rerank_score is not None else ""
            heading = f"  ·  §{c.section_heading}" if c.section_heading else ""
            st.markdown(
                f'<div class="chunk-header">[{i}] {c.source_filename} &nbsp;·&nbsp; '
                f'p.{c.page_num} &nbsp;·&nbsp; {c.chunk_type}{score_str}{heading}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="chunk-card">{c.text[:600]}</div>', unsafe_allow_html=True)


def _render_turn(turn: dict, turn_idx: int):
    with st.chat_message("user"):
        st.markdown(turn["query"])

    with st.chat_message("assistant"):
        result = turn["result"]
        st.markdown(result.answer)
        st.markdown(_confidence_html(result.confidence), unsafe_allow_html=True)
        _render_citations(result.citations, expanded=False)
        _render_debug(turn.get("chunks", []))

        if result.follow_up_questions:
            fqs = "".join(f'<div class="followup-item">{fq}</div>' for fq in result.follow_up_questions)
            st.markdown(
                f'<div class="followup-wrap">'
                f'<div class="followup-label">Suggested follow-ups</div>'
                f'{fqs}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-company">Real Estate Co</div>'
        '<div class="sb-product">Valuation Intelligence</div>'
        '<div class="sb-tagline">RAG · Hybrid Retrieval · Grounded Answers</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    engine, config, load_error = _load_engine()
    st.session_state.engine = engine
    st.session_state.config = config

    if load_error:
        st.error(f"Engine load failed: {load_error}")

    # ── System status ─────────────────────────────────────────────────────────
    st.markdown("**Knowledge Base**")

    try:
        registry = DocumentRegistry(config.registry_path)
        summary = registry.summary()
        vs = VectorStore(config.chroma_dir)
        chunk_count = vs.count()
        col1, col2 = st.columns(2)
        col1.metric("Documents", summary.get("indexed", 0))
        col2.metric("Chunks", chunk_count)
        if summary.get("failed", 0) > 0:
            st.warning(f"{summary['failed']} document(s) failed ingestion.")
        if chunk_count == 0:
            st.warning("No chunks indexed. Run ingestion first.")
    except Exception as e:
        st.error(f"Status error: {e}")

    st.divider()

    # ── Ingestion controls ────────────────────────────────────────────────────
    st.markdown("**Document Ingestion**")
    force_cb = st.checkbox("Force reindex all documents", value=False)

    if st.button("↻  Re-ingest Documents", use_container_width=True):
        progress_lines: list[str] = []
        log_placeholder = st.empty()

        def _progress(msg: str):
            progress_lines.append(msg)
            log_placeholder.text("\n".join(progress_lines[-12:]))

        with st.spinner("Running ingestion pipeline..."):
            try:
                pipeline = IngestionPipeline(config, progress_callback=_progress)
                res = pipeline.run(force_reindex=force_cb)
                st.success(
                    f"Done — {res['processed']} processed, "
                    f"{res['skipped']} skipped, {res['failed']} failed"
                )
            except Exception as e:
                st.error(f"Ingestion error: {e}")
                res = {}

        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ── Retrieval filter ──────────────────────────────────────────────────────
    st.markdown("**Retrieval Scope**")
    filter_choice = st.selectbox(
        "Filter by content type:",
        ["All documents", "Tables only", "Text only"],
        index=0,
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


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="rec-header">
    <div class="rec-logo">🏢</div>
    <div class="rec-brand">
        <div class="rec-company">Real Estate Co</div>
        <div class="rec-product">Property Valuation Intelligence</div>
    </div>
    <div class="rec-divider"></div>
    <div class="rec-badge">Grounded · Cited · Evidence-Backed</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# Starter questions (shown only on fresh session)
if not st.session_state.turns:
    st.markdown('<div class="starter-label">Suggested questions</div>', unsafe_allow_html=True)
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
        if col.button(q, key=f"starter_{i}", use_container_width=True):
            st.session_state["_pending"] = q
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

st.divider()

# Render conversation history
for idx, turn in enumerate(st.session_state.turns):
    _render_turn(turn, idx)

# ── Chat input + query handler ────────────────────────────────────────────────

pending = st.session_state.pop("_pending", None)
user_input = st.chat_input("Ask about cap rates, NOI, rent rolls, comps, underwriting...")
active_query = pending or user_input

if active_query:
    _engine = st.session_state.engine
    _config = st.session_state.config

    if _engine is None:
        st.error(
            "Engine not loaded. Check that OPENAI_API_KEY is set in .env "
            "and refresh the page."
        )
        st.stop()

    try:
        _vs = VectorStore(_config.chroma_dir)
        if _vs.count() == 0:
            st.warning(
                "No documents indexed yet. Run `python3 ingest.py` in your terminal "
                "or click **Re-ingest Documents** in the sidebar, then try again."
            )
            st.stop()
    except Exception:
        pass

    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                result, top_chunks = _engine.ask_with_chunks(
                    query=active_query,
                    chat_history=st.session_state.chat_history,
                    metadata_filter=metadata_filter,
                )
            except Exception as e:
                err_msg = str(e)
                if "api_key" in err_msg.lower() or "authentication" in err_msg.lower():
                    st.error(
                        "LLM authentication failed. Check that OPENAI_API_KEY "
                        "is set correctly in your .env file."
                    )
                elif "timeout" in err_msg.lower():
                    st.error("LLM request timed out. Please try again.")
                else:
                    st.error(f"Error: {err_msg}")
                st.stop()

        st.markdown(result.answer)
        st.markdown(_confidence_html(result.confidence), unsafe_allow_html=True)
        _render_citations(result.citations, expanded=True)
        _render_debug(top_chunks)

        if result.follow_up_questions:
            fqs = "".join(f'<div class="followup-item">{fq}</div>' for fq in result.follow_up_questions)
            st.markdown(
                f'<div class="followup-wrap">'
                f'<div class="followup-label">Suggested follow-ups</div>'
                f'{fqs}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.session_state.chat_history.append({"role": "user", "content": active_query})
    st.session_state.chat_history.append({"role": "assistant", "content": result.answer})
    st.session_state.turns.append({
        "query": active_query,
        "result": result,
        "chunks": top_chunks,
    })

    st.rerun()
