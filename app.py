# app.py
# =============================================================================
# COMP64702 RAG Culinary Assistant — Streamlit Demo UI
# =============================================================================
# Run with:
#   streamlit run app.py
#
# This launches a local web interface at http://localhost:8501
# =============================================================================

import os
import sys
import json
import time
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from embedder  import Embedder
from retriever import Retriever
from generator import Generator, build_context

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="East Asian Culinary Assistant",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.answer-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 1.2rem 1.5rem;
    border-radius: 0 8px 8px 0;
    font-size: 1.05rem;
    line-height: 1.7;
    margin: 1rem 0;
    color: #111;
}
.stat-box {
    background: #f0f4ff;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.stat-value { font-size: 1.3rem; font-weight: 700; color: #3f51b5; }
.stat-label { font-size: 0.78rem; color: #666; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load all models once — cached for the whole session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    load_dotenv()
    from huggingface_hub import login
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    emb  = Embedder("BAAI/bge-small-en-v1.5")
    ret  = Retriever("vector_store", embedder=emb)
    gen  = Generator("Qwen/Qwen2.5-0.5B-Instruct")
    return emb, ret, gen


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    retrieval_k = st.slider(
        "Chunks to retrieve", 1, 5, 5,
        help="Number of context chunks (max 5 per coursework spec)"
    )
    strategy = st.selectbox(
        "Prompting strategy",
        ["structured", "zero_shot", "few_shot", "chain_of_thought"],
        index=0,
    )
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_scores = st.toggle("Show relevance scores", value=True)

    st.divider()
    st.markdown("## 🏗️ System")
    st.markdown("""
| Component | Details |
|---|---|
| Chunking | Semantic |
| Embedder | BGE-small |
| Retrieval | Hybrid + RRF |
| Reranker | MS-MARCO |
| LLM | Qwen 2.5-0.5B |
""")

    st.divider()
    st.markdown("## 💡 Try these questions")
    SAMPLES = [
        "What is miso soup made from?",
        "How is kimchi made?",
        "What makes Sichuan food spicy?",
        "How does hot pot differ from shabu-shabu?",
        "What is the history of sushi?",
        "What are the ingredients in bibimbap?",
        "How is Peking duck prepared?",
        "What is wagyu beef?",
        "Difference between udon and soba?",
        "Where did dim sum originate?",
        "What is doenjang?",
        "How is sake produced?",
        "What is the Japanese tea ceremony?",
        "What is wok hei?",
        "How is tofu made?",
    ]
    for q in SAMPLES:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

    st.divider()
    st.caption("COMP64702 · East Asian Cuisine RAG")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🍜 East Asian Culinary Assistant")
st.markdown("*Retrieval-Augmented Generation — COMP64702 Coursework*")

# Initialise session state
if "history"  not in st.session_state: st.session_state.history  = []
if "prefill"  not in st.session_state: st.session_state.prefill  = ""

# Query input row
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "question",
        value=st.session_state.prefill,
        placeholder="Ask anything about East Asian cuisine...",
        label_visibility="collapsed",
    )
with col_btn:
    go = st.button("Ask 🔍", type="primary", use_container_width=True)

# Reset prefill
if st.session_state.prefill:
    st.session_state.prefill = ""

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

if go and query.strip():

    with st.spinner("Loading models — first run takes ~30 seconds..."):
        embedder, retriever, generator = load_models()

    with st.spinner("🔍 Searching knowledge base..."):
        t0 = time.time()
        retrieved = retriever.retrieve(query, initial_k=20, final_k=retrieval_k)
        t_ret = time.time() - t0

    with st.spinner("✍️ Generating answer..."):
        t1 = time.time()
        context = build_context(retrieved, max_words=600)
        answer  = generator.generate(
            query=query,
            context=context,
            strategy=strategy,
            max_new_tokens=200,
            temperature=0.3,
        )
        t_gen = time.time() - t1

    # Save history
    st.session_state.history.insert(0, {
        "query": query, "answer": answer,
        "retrieved": retrieved,
        "t_ret": t_ret, "t_gen": t_gen,
        "strategy": strategy,
        "ts": datetime.now().strftime("%H:%M:%S"),
    })

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("### 💬 Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{t_ret:.2f}s",   "Retrieval time"),
        (c2, f"{t_gen:.2f}s",   "Generation time"),
        (c3, str(len(retrieved)),"Chunks used"),
        (c4, strategy,           "Strategy"),
    ]:
        col.markdown(
            f'<div class="stat-box"><div class="stat-value">{val}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    if show_chunks:
        st.markdown("### 📚 Retrieved Chunks")
        st.caption(f"Top {len(retrieved)} chunks from 3,356 total")

        for i, res in enumerate(retrieved, 1):
            chunk = res["chunk"]
            score = res.get("reranker_score", res.get("score", 0))
            title = chunk.get("doc_title", "Unknown")
            src   = chunk.get("source_type", "").capitalize()
            url   = chunk.get("doc_url", "")
            text  = chunk.get("text", "")[:280] + "..."

            with st.expander(f"[{i}] {title}  ·  {src}", expanded=(i == 1)):
                if show_scores:
                    st.caption(f"Reranker score: {score:.4f}")
                if url:
                    st.caption(f"Source: {url}")
                st.write(text)

elif go and not query.strip():
    st.warning("Please enter a question.")


# ─────────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.history:
    st.divider()
    st.markdown("### 🕐 Session History")
    for i, item in enumerate(st.session_state.history[:5]):
        with st.expander(f"[{item['ts']}]  {item['query'][:80]}", expanded=False):
            st.markdown(f"**Answer:** {item['answer']}")
            st.caption(f"Retrieval {item['t_ret']:.2f}s · Generation {item['t_gen']:.2f}s · {item['strategy']}")
    if st.button("Clear history", use_container_width=False):
        st.session_state.history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Batch inference — demo day JSON upload
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("### 📥 Batch Inference  *(Demo Day)*")
st.caption("Upload the markers' input JSON → run inference → download your output JSON")

uploaded = st.file_uploader("Upload input_payload.json", type=["json"])

if uploaded:
    try:
        data    = json.load(uploaded)
        queries = data.get("queries", data) if isinstance(data, dict) else data
        st.success(f"✅ {len(queries)} queries loaded")

        if st.button("▶️ Run batch inference", type="primary"):
            with st.spinner("Loading models..."):
                embedder, retriever, generator = load_models()

            outputs  = []
            progress = st.progress(0)
            status   = st.empty()

            for i, item in enumerate(queries):
                q   = item.get("question", item.get("query", ""))
                qid = item.get("id", f"Q{i+1}")
                status.text(f"[{i+1}/{len(queries)}] {q[:70]}...")

                retrieved = retriever.retrieve(q, initial_k=20, final_k=5)
                context   = build_context(retrieved, max_words=600)
                answer    = generator.generate(
                    query=q, context=context,
                    strategy="structured",
                    max_new_tokens=200, temperature=0.3,
                )
                outputs.append({"id": qid, "answer": answer})
                progress.progress((i + 1) / len(queries))

            status.success(f"Done — {len(outputs)} answers generated!")

            payload = {
                "generated_at": datetime.now().isoformat(),
                "model":        "Qwen/Qwen2.5-0.5B-Instruct",
                "retrieval":    "hybrid_bm25_dense_rrf_crossencoder_rerank",
                "outputs":      outputs,
            }

            st.download_button(
                "⬇️ Download output_payload.json",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="output_payload.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error: {e}")
