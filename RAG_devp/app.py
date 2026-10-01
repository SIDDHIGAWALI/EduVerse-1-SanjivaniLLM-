"""
Streamlit frontend for EduVerse-1 — Sanjivani University RAG Chatbot.

Connects to the existing backend (src/rag/pipeline.py, src/model_free/extractive_generator.py)
without modifying any existing source code.

Run:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Path setup (mirrors run_demo.py)
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "model"))

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduVerse — Sanjivani University AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS (Dark Theme with high contrast)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Font & Body ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0 !important;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #e2e8f0 !important;
}

/* ── Global text color fixes for Streamlit widgets ── */
.stMarkdown, p, span, h1, h2, h3, h4, h5, h6, label, div {
    color: #e2e8f0;
}

/* ── Header bar background ── */
[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(10px);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Main container ── */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 850px;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 1.8rem 1rem 1rem;
    margin-bottom: 1rem;
}
.hero-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c084fc, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-header p {
    color: #94a3b8 !important;
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* ── Chat bubbles ── */
.chat-bubble {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 1rem;
}
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar-user  { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; }
.avatar-bot   { background: linear-gradient(135deg, #0ea5e9, #06b6d4); color: white; }
.chat-content {
    flex: 1;
    padding: 0.85rem 1.1rem;
    border-radius: 14px;
    font-size: 0.95rem;
    line-height: 1.6;
}
.user-content {
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #f1f5f9 !important;
}
.bot-content {
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(14, 165, 233, 0.3);
    color: #f1f5f9 !important;
}

/* ── Source pills & cards ── */
.source-pill {
    display: inline-block;
    background: rgba(192, 132, 252, 0.2);
    border: 1px solid rgba(192, 132, 252, 0.4);
    color: #c084fc !important;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.chunk-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.chunk-score {
    font-size: 0.75rem;
    color: #94a3b8 !important;
    margin-bottom: 0.4rem;
}
.chunk-text {
    color: #cbd5e1 !important;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-ready {
    background: rgba(52, 211, 153, 0.2);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #34d399 !important;
}
.status-error {
    background: rgba(248, 113, 113, 0.2);
    border: 1px solid rgba(248, 113, 113, 0.4);
    color: #f87171 !important;
}

/* ── Input box styling ── */
.stTextInput input {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
}
.stTextInput input:focus {
    border-color: #818cf8 !important;
}

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "init_error" not in st.session_state:
    st.session_state.init_error = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ──────────────────────────────────────────────────────────────────────────────
# Backend loader (cached so it only runs once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_retriever(index_path: str, meta_path: str):
    from src.rag.retrieve import Retriever
    return Retriever(index_path, meta_path)


def get_retriever():
    if st.session_state.retriever is None and st.session_state.init_error is None:
        try:
            idx = str(ROOT / "data" / "sanjivani.index")
            meta = str(ROOT / "data" / "sanjivani_meta.jsonl")
            st.session_state.retriever = load_retriever(idx, meta)
        except Exception as exc:
            st.session_state.init_error = str(exc)
    return st.session_state.retriever


# ──────────────────────────────────────────────────────────────────────────────
# Query functions with strict grounding guardrails
# ──────────────────────────────────────────────────────────────────────────────
MIN_RELEVANCE_SCORE = 0.20  # Minimum similarity score required

def run_extractive(query: str, k: int = 3, min_score: float = MIN_RELEVANCE_SCORE) -> dict:
    """Run extractive RAG with strict relevance threshold."""
    from src.model_free.extractive_generator import extractive_answer

    retriever = get_retriever()
    if retriever is None:
        return {"error": st.session_state.init_error}

    t0 = time.perf_counter()
    chunks = retriever.retrieve(query, k=k, min_score=min_score)
    
    if not chunks:
        answer = "⚠️ I could not find relevant information in the Sanjivani University database for your question. Please verify your query or contact university administration."
    else:
        answer = extractive_answer(chunks)
        
    latency = time.perf_counter() - t0
    return {"answer": answer, "chunks": chunks, "latency": latency}


def run_neural(query: str, checkpoint: str, tokenizer_path: str, k: int = 3, max_new_tokens: int = 80, min_score: float = MIN_RELEVANCE_SCORE) -> dict:
    """Run neural (EduVerse-1) RAG with strict relevance threshold."""
    retriever = get_retriever()
    if retriever is None:
        return {"error": st.session_state.init_error}

    try:
        t0 = time.perf_counter()
        raw_chunks = retriever.retrieve(query, k=k, min_score=min_score)

        if not raw_chunks:
            latency = time.perf_counter() - t0
            return {
                "answer": "⚠️ I could not find relevant information in the Sanjivani University database for your question. EduVerse-1 will not generate answers without verified document context.",
                "chunks": [],
                "latency": latency,
            }

        from generate import EduVerseGenerator
        from src.rag.pipeline import RAGPipeline

        generator = EduVerseGenerator(checkpoint, tokenizer_path)
        pipeline = RAGPipeline(retriever, generate_fn=generator, k=k)
        result = pipeline.answer(query)
        latency = time.perf_counter() - t0

        return {"answer": result.answer, "chunks": result.retrieved_chunks, "latency": latency}

    except Exception as exc:
        return {"error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    mode = st.radio(
        "**Answer Mode**",
        ["Extractive (fast, no hallucination)", "Neural — EduVerse-1"],
        index=0,
        help="Extractive mode pulls text directly from docs. Neural uses the trained LLM.",
    )
    extractive_mode = mode.startswith("Extractive")

    st.markdown("")
    k_val = st.slider("**Retrieved chunks (k)**", min_value=1, max_value=6, value=3, step=1)
    min_score_val = st.slider(
        "**Relevance threshold**",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="Chunks with similarity score below this threshold are ignored to prevent answering out-of-scope questions.",
    )

    if not extractive_mode:
        st.divider()
        st.markdown("### 🧠 EduVerse-1 Settings")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="checkpoints/stage3_sanjivani/checkpoint.pt",
            placeholder="checkpoints/.../checkpoint.pt",
        )
        tokenizer_path = st.text_input(
            "Tokenizer path",
            value="data/eduverse_tokenizer.json",
            placeholder="data/eduverse_tokenizer.json",
        )
        max_new_tokens = st.slider("Max new tokens", 20, 200, 80, step=10)
    else:
        checkpoint_path = None
        tokenizer_path = None
        max_new_tokens = 80

    st.divider()

    # Status indicator
    retriever = get_retriever()
    if retriever is not None:
        st.markdown(
            '<span class="status-badge status-ready">● Retriever ready</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span style='color:#64748b;font-size:0.8rem;'>Index: {retriever.index.ntotal} chunks</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-error">✗ Retriever error</span>',
            unsafe_allow_html=True,
        )
        if st.session_state.init_error:
            st.error(st.session_state.init_error)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Messages", len(st.session_state.messages))

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown(
        "<div style='color:#475569;font-size:0.75rem;text-align:center;padding-top:1rem;'>"
        "EduVerse-1 · Sanjivani University<br>RAG Demo v1.0"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-header">
    <h1>🎓 EduVerse AI</h1>
    <p>Ask anything about Sanjivani University — admissions, exams, facilities, and more.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Suggestion chips (only when chat is empty)
# ──────────────────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What is the admission process?",
    "What are the exam rules?",
    "Tell me about hostel facilities.",
    "What is the fee structure?",
    "What courses are available?",
    "What is the attendance requirement?",
]

if not st.session_state.messages:
    st.markdown(
        "<p style='text-align:center;color:#475569;font-size:0.85rem;margin-bottom:0.6rem;'>"
        "Try asking…</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            with st.container():
                st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_query = suggestion
                st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
<div class="chat-bubble">
    <div class="chat-avatar avatar-user">👤</div>
    <div class="chat-content user-content">{msg["content"]}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        answer_html = msg["content"].replace("\n", "<br>")
        st.markdown(
            f"""
<div class="chat-bubble">
    <div class="chat-avatar avatar-bot">🤖</div>
    <div class="chat-content bot-content">{answer_html}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if msg.get("chunks"):
            with st.expander(
                f"📚 Retrieved sources ({len(msg['chunks'])} chunks) · {msg.get('latency', 0):.2f}s",
                expanded=False,
            ):
                for c in msg["chunks"]:
                    source_label = getattr(c, "source_file", "unknown")
                    score_val = getattr(c, "score", 0.0)
                    text_val = getattr(c, "text", "")
                    st.markdown(
                        f"""
<div class="chunk-card">
    <div class="chunk-score">
        <span class="source-pill">{source_label}</span>
        &nbsp; score: {score_val:.3f}
    </div>
    <div class="chunk-text">{text_val}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

# ──────────────────────────────────────────────────────────────────────────────
# Input form
# ──────────────────────────────────────────────────────────────────────────────
pending = st.session_state.get("pending_query", "")

with st.form(key="query_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            label="query",
            placeholder="Ask a question about Sanjivani University…",
            label_visibility="collapsed",
            value=pending if pending else "",
        )
    with col_btn:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

# Clear pending after form renders
if pending:
    st.session_state.pending_query = None

# ──────────────────────────────────────────────────────────────────────────────
# Process query
# ──────────────────────────────────────────────────────────────────────────────
if submitted and user_input.strip():
    query = user_input.strip()

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.total_queries += 1

    with st.spinner("🔍 Searching knowledge base…"):
        if extractive_mode:
            result = run_extractive(query, k=k_val, min_score=min_score_val)
        else:
            result = run_neural(
                query,
                checkpoint=checkpoint_path,
                tokenizer_path=tokenizer_path,
                k=k_val,
                max_new_tokens=max_new_tokens,
                min_score=min_score_val,
            )

    if "error" in result:
        bot_msg = {
            "role": "assistant",
            "content": f"⚠️ Error: {result['error']}",
            "chunks": [],
            "latency": 0.0,
        }
    else:
        bot_msg = {
            "role": "assistant",
            "content": result["answer"],
            "chunks": result.get("chunks", []),
            "latency": result.get("latency", 0.0),
        }

    st.session_state.messages.append(bot_msg)
    st.rerun()
