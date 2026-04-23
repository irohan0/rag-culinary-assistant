#!/usr/bin/env python
# app.py
# =============================================================================
# COMP64702 RAG Culinary Assistant — Final Demo UI
# =============================================================================

import os
import sys
import json
import time
import torch
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import login

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BASE_DIR)

from embedder  import Embedder
from retriever import Retriever
from generator import Generator, build_context

# ── Auth ──────────────────────────────────────────────────────────────────────
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ── Load models ───────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  RAG Culinary Assistant — Loading Models")
print("="*55)

VECTOR_STORE = os.path.join(BASE_DIR, "vector_store")
embedder_    = Embedder("BAAI/bge-small-en-v1.5")
retriever_   = Retriever(VECTOR_STORE, embedder=embedder_)
generator_   = Generator("Qwen/Qwen2.5-0.5B-Instruct")

print("\nAll models ready — launching UI\n")

# ── Sample questions ──────────────────────────────────────────────────────────
SAMPLES = [
    "What is the main ingredient in miso soup?",
    "How is kimchi made?",
    "What makes Sichuan cuisine spicy?",
    "How does hot pot differ from shabu-shabu?",
    "What is the history of sushi?",
    "What are the key ingredients in bibimbap?",
    "How is Peking duck prepared?",
    "What is wagyu beef?",
    "What is the cultural significance of kimchi in Korea?",
    "How is tofu made?",
    "What is the difference between udon and soba noodles?",
    "What is dim sum and where did it originate?",
]

# ── Input parser ──────────────────────────────────────────────────────────────
def parse_input(data):
    queries = []
    counter = 1
    if isinstance(data, dict) and "sources" in data:
        for block in data["sources"]:
            src = block.get("source", "unknown")
            for q in block.get("questions", []):
                question = q if isinstance(q, str) else q.get("question", "")
                answer   = None if isinstance(q, str) else q.get("answer", None)
                if not question.strip():
                    continue
                queries.append({
                    "id":          q.get("id", f"Q{counter:03d}") if isinstance(q, dict) else f"Q{counter:03d}",
                    "question":    question.strip(),
                    "gold_answer": answer,
                    "source":      src,
                })
                counter += 1
        return queries
    if isinstance(data, dict) and "queries" in data:
        for q in data["queries"]:
            if isinstance(q, str):
                queries.append({"id": f"Q{counter:03d}", "question": q, "gold_answer": None})
            else:
                queries.append({
                    "id":          q.get("id", f"Q{counter:03d}"),
                    "question":    q.get("question", q.get("query", "")),
                    "gold_answer": q.get("answer", None),
                })
            counter += 1
        return queries
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                queries.append({"id": f"Q{counter:03d}", "question": item, "gold_answer": None})
            elif isinstance(item, dict):
                queries.append({
                    "id":          item.get("id", f"Q{counter:03d}"),
                    "question":    item.get("question", item.get("query", "")),
                    "gold_answer": item.get("answer", None),
                })
            counter += 1
        return queries
    raise ValueError(f"Unrecognised JSON format: {type(data)}")

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background: #0f1117 !important;
}
.header-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2d3561; border-radius: 16px;
    padding: 32px 40px; margin-bottom: 8px; text-align: center;
}
.header-title { font-size: 2.2rem; font-weight: 700; color: #e2e8f0;
    margin: 0 0 6px 0; letter-spacing: -0.5px; }
.header-sub { font-size: 0.95rem; color: #94a3b8; margin: 0 0 16px 0; }
.badge-row  { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.badge      { padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-blue   { background: #1e3a5f; color: #7dd3fc; border: 1px solid #2563eb44; }
.badge-green  { background: #14532d; color: #86efac; border: 1px solid #16a34a44; }
.badge-purple { background: #3b1f6e; color: #c4b5fd; border: 1px solid #7c3aed44; }
.badge-orange { background: #7c2d12; color: #fdba74; border: 1px solid #ea580c44; }
.badge-teal   { background: #134e4a; color: #5eead4; border: 1px solid #0d948844; }
.panel        { background: #1a1f2e; border: 1px solid #2d3561; border-radius: 12px; padding: 20px; }
.panel-title  { font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #64748b; margin-bottom: 12px; }
.answer-card  { background: #0d1117; border: 1px solid #2d3561;
    border-left: 4px solid #3b82f6; border-radius: 12px;
    padding: 20px 24px; min-height: 120px; color: #e2e8f0;
    font-size: 1rem; line-height: 1.7; white-space: pre-wrap; }
.answer-empty { color: #475569; font-style: italic; }
.chunk-card   { background: #0d1117; border: 1px solid #1e293b;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
.chunk-rank   { display:inline-block; background:#1e3a5f; color:#7dd3fc;
    font-size:0.7rem; font-weight:700; padding:2px 8px;
    border-radius:10px; margin-right:8px; }
.chunk-title  { color:#e2e8f0; font-weight:600; font-size:0.9rem; }
.chunk-score  { float:right; color:#64748b; font-size:0.75rem; font-family:monospace; }
.chunk-text   { color:#94a3b8; font-size:0.85rem; line-height:1.6;
    margin-top:8px; border-top:1px solid #1e293b; padding-top:8px; }
.stats-bar    { display:flex; gap:16px; background:#0d1117; border:1px solid #1e293b;
    border-radius:10px; padding:12px 20px; margin-bottom:16px; flex-wrap:wrap; }
.stat-item    { text-align:center; flex:1; min-width:80px; }
.stat-val     { font-size:1.3rem; font-weight:700; color:#7dd3fc; }
.stat-lbl     { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; }
.result-box   { background:#0d1117; border:1px solid #1e293b; border-radius:10px;
    padding:16px; min-height:160px; color:#e2e8f0; font-size:0.9rem; line-height:1.6; }
.result-label { font-size:0.65rem; font-weight:700; letter-spacing:1px;
    text-transform:uppercase; padding:4px 10px; border-radius:6px;
    margin-bottom:8px; display:inline-block; }
.label-blue   { background:#1e3a5f; color:#7dd3fc; }
.label-green  { background:#14532d; color:#86efac; }
.label-purple { background:#3b1f6e; color:#c4b5fd; }
.label-orange { background:#7c2d12; color:#fdba74; }
label.svelte-1b6s6s  { color:#94a3b8 !important; font-size:0.8rem !important; }
.svelte-1gfkn6j      { background:#1a1f2e !important; border-color:#2d3561 !important; }
textarea, input      { background:#0d1117 !important; color:#e2e8f0 !important;
    border-color:#2d3561 !important; }
.tab-nav button      { color:#94a3b8 !important; }
.tab-nav button.selected { color:#7dd3fc !important; border-bottom:2px solid #3b82f6 !important; }
footer { display:none !important; }
"""

# ── Core inference ────────────────────────────────────────────────────────────
def rag_query(question, strategy, num_chunks):
    question = (question or "").strip()
    if not question:
        return (
            '<div class="answer-card answer-empty">Enter a question above and click Ask.</div>',
            '<div style="color:#475569;padding:20px;text-align:center;">Chunks will appear here.</div>',
            "",
        )
    start     = time.time()
    retrieved = retriever_.retrieve(question, initial_k=20, final_k=int(num_chunks))
    context   = build_context(retrieved, max_words=600)
    answer    = generator_.generate(
        query=question, context=context, strategy=strategy,
        max_new_tokens=250, temperature=0.3,
    )
    elapsed = time.time() - start
    answer_html = f'<div class="answer-card">{answer}</div>'
    chunks_html = ""
    for i, r in enumerate(retrieved, 1):
        title  = r["chunk"].get("doc_title", "Unknown")
        text   = r["chunk"].get("text", "")[:280]
        source = r["chunk"].get("source_type", "").capitalize()
        score  = r.get("reranker_score", 0)
        chunks_html += f"""
        <div class="chunk-card">
            <span class="chunk-rank">#{i}</span>
            <span class="chunk-title">{title}</span>
            <span class="chunk-score">score: {score:.4f} · {source}</span>
            <div class="chunk-text">{text}...</div>
        </div>"""
    strat_label = strategy.replace("_", " ").title()
    stats_html  = f"""
    <div class="stats-bar">
        <div class="stat-item"><div class="stat-val">{len(retrieved)}</div>
            <div class="stat-lbl">Chunks used</div></div>
        <div class="stat-item"><div class="stat-val">{elapsed:.1f}s</div>
            <div class="stat-lbl">Latency</div></div>
        <div class="stat-item"><div class="stat-val">{len(answer.split())}</div>
            <div class="stat-lbl">Answer words</div></div>
        <div class="stat-item"><div class="stat-val">{strat_label}</div>
            <div class="stat-lbl">Strategy</div></div>
        <div class="stat-item"><div class="stat-val">Hybrid+RRF</div>
            <div class="stat-lbl">Retrieval</div></div>
    </div>"""
    return answer_html, chunks_html, stats_html


def compare_all(question):
    question = (question or "").strip()
    if not question:
        empty = '<div class="result-box answer-empty">Enter a question above.</div>'
        return empty, empty, empty, empty
    retrieved = retriever_.retrieve(question, initial_k=20, final_k=5)
    context   = build_context(retrieved, max_words=600)
    results   = {}
    for strat in ["zero_shot", "few_shot", "chain_of_thought", "structured"]:
        results[strat] = generator_.generate(
            query=question, context=context, strategy=strat,
            max_new_tokens=200, temperature=0.3,
        )
    def box(t): return f'<div class="result-box">{t}</div>'
    return box(results["zero_shot"]), box(results["few_shot"]), \
           box(results["chain_of_thought"]), box(results["structured"])


def batch_inference(file_obj, strategy):
    if file_obj is None:
        return "Please upload an input JSON file.", None
    try:
        with open(file_obj.name, encoding="utf-8") as f:
            data = json.load(f)

        queries = parse_input(data)
        if not queries:
            return "No questions found in the file. Check the format.", None

        strat_label = strategy.replace("_", " ").title()
        log         = [f"Detected {len(queries)} questions  ·  Strategy: {strat_label}\n"]
        outputs     = []

        for i, item in enumerate(queries, 1):
            question = item.get("question", "").strip()
            qid      = item.get("id", f"Q{i:03d}")

            if not question:
                log.append(f"[{i:02d}/{len(queries)}] SKIP — empty question")
                continue

            retrieved = retriever_.retrieve(question, initial_k=20, final_k=5)
            context   = build_context(retrieved, max_words=600)
            answer    = generator_.generate(
                query=question, context=context,
                strategy=strategy,
                max_new_tokens=200, temperature=0.3,
            )
            outputs.append({"id": qid, "answer": answer})
            log.append(f"[{i:02d}/{len(queries)}] [{strat_label}] {question[:55]}")

        payload = {
            "generated_at": datetime.now().isoformat(),
            "model":        "Qwen/Qwen2.5-0.5B-Instruct",
            "retrieval":    "hybrid_bm25_dense_rrf_crossencoder_rerank",
            "strategy":     strategy,
            "outputs":      outputs,
        }

        os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
        out_path = os.path.join(BASE_DIR, "outputs", "output_payload.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        log.append(f"\nDone — {len(outputs)} answers saved  ·  Strategy: {strat_label}")
        return "\n".join(log), out_path

    except Exception as e:
        import traceback
        return f"Error: {type(e).__name__}: {e}\n\n{traceback.format_exc()}", None

# ── Load metadata ─────────────────────────────────────────────────────────────
try:
    with open(os.path.join(VECTOR_STORE, "metadata.json")) as f:
        meta = json.load(f)
    chunks_n = meta.get("total_chunks", 3356)
    emb_dim  = meta.get("embedding_dim", 384)
    created  = meta.get("created_at", "")[:10]
except Exception:
    chunks_n, emb_dim, created = 3356, 384, "N/A"

# ── Reusable table row helper ─────────────────────────────────────────────────
def trow(label, baseline, rag, sig="", highlight=False):
    tc = "#86efac" if highlight else "#e2e8f0"
    return (
        f'<tr style="border-top:1px solid #1e293b">'
        f'<td style="padding:5px 0;color:{tc};font-weight:600;">{label}</td>'
        f'<td style="text-align:right;">{baseline}</td>'
        f'<td style="text-align:right;color:#86efac;font-weight:600;">{rag}</td>'
        f'</tr>'
    )

def trow_abl(label, rl, ndcg, tc="#e2e8f0"):
    return (
        f'<tr style="border-top:1px solid #1e293b">'
        f'<td style="padding:5px 0;color:{tc};font-weight:600;">{label}</td>'
        f'<td style="text-align:right;">{rl}</td>'
        f'<td style="text-align:right;">{ndcg}</td>'
        f'</tr>'
    )

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="RAG Culinary Assistant") as demo:

    gr.HTML("""
    <div class="header-box">
        <div class="header-title">🍜 East Asian Cuisine RAG Assistant</div>
        <div class="header-sub">
            COMP64702 · University of Manchester · Retrieval-Augmented Generation
        </div>
        <div class="badge-row">
            <span class="badge badge-blue">BGE-small Embeddings</span>
            <span class="badge badge-green">Hybrid BM25 + Dense + RRF</span>
            <span class="badge badge-purple">Cross-Encoder Reranking</span>
            <span class="badge badge-orange">Qwen 2.5-0.5B</span>
            <span class="badge badge-teal">Semantic Chunking</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Ask ────────────────────────────────────────────────────────
        with gr.Tab("🔍  Ask a Question"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    question_box = gr.Textbox(
                        placeholder="e.g. How is kimchi made? What is the history of sushi?",
                        label="Your question", lines=3,
                    )
                    with gr.Row():
                        strategy_dd = gr.Dropdown(
                            choices=[
                                ("Structured", "structured"),
                                ("Zero-shot",               "zero_shot"),
                                ("Few-shot",                "few_shot"),
                                ("Chain-of-thought",        "chain_of_thought"),
                            ],
                            value="structured", label="Prompting strategy",
                        )
                        chunks_sl = gr.Slider(
                            minimum=1, maximum=5, value=5, step=1,
                            label="Chunks to retrieve (max 5)",
                        )
                    ask_btn = gr.Button("⚡  Ask", variant="primary", size="lg")
                    gr.Examples(
                        examples=[[q] for q in SAMPLES],
                        inputs=[question_box],
                        label="Sample questions",
                        examples_per_page=6,
                    )
                with gr.Column(scale=6):
                    stats_out  = gr.HTML()
                    answer_out = gr.HTML(
                        value='<div class="answer-card answer-empty">Your answer will appear here.</div>'
                    )
                    gr.HTML('<div class="panel-title" style="margin-top:16px;">Retrieved chunks used as context</div>')
                    chunks_out = gr.HTML(
                        value='<div style="color:#475569;padding:20px;text-align:center;font-style:italic;">Chunks will appear after you ask a question.</div>'
                    )
            ask_btn.click(
                fn=rag_query,
                inputs=[question_box, strategy_dd, chunks_sl],
                outputs=[answer_out, chunks_out, stats_out],
            )
            question_box.submit(
                fn=rag_query,
                inputs=[question_box, strategy_dd, chunks_sl],
                outputs=[answer_out, chunks_out, stats_out],
            )

        # ── Tab 2: Compare ────────────────────────────────────────────────────
        with gr.Tab("⚖️  Compare Strategies"):
            gr.HTML('<div style="color:#94a3b8;margin-bottom:16px;">Run the same question through all 4 prompting strategies simultaneously.</div>')
            cmp_question = gr.Textbox(
                placeholder="Enter a question to compare...", label="Question", lines=2,
            )
            cmp_btn = gr.Button("⚡  Compare all strategies", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.HTML('<span class="result-label label-blue">Zero-shot</span>')
                    zs_out = gr.HTML('<div class="result-box answer-empty">Results will appear here.</div>')
                with gr.Column():
                    gr.HTML('<span class="result-label label-green">Few-shot</span>')
                    fs_out = gr.HTML('<div class="result-box answer-empty">Results will appear here.</div>')
            with gr.Row():
                with gr.Column():
                    gr.HTML('<span class="result-label label-purple">Chain-of-thought</span>')
                    cot_out = gr.HTML('<div class="result-box answer-empty">Results will appear here.</div>')
                with gr.Column():
                    gr.HTML('<span class="result-label label-orange">Structured ✓ selected</span>')
                    str_out = gr.HTML('<div class="result-box answer-empty">Results will appear here.</div>')
            cmp_btn.click(fn=compare_all, inputs=[cmp_question],
                          outputs=[zs_out, fs_out, cot_out, str_out])
            cmp_question.submit(fn=compare_all, inputs=[cmp_question],
                                outputs=[zs_out, fs_out, cot_out, str_out])

        # ── Tab 3: Batch ──────────────────────────────────────────────────────
        # ── Tab 3: Batch ──────────────────────────────────────────────────────
        with gr.Tab("📂  Batch Inference"):
            gr.HTML("""
            <div style="color:#94a3b8;margin-bottom:16px;">
                Upload any input JSON file — handles the demo day
                <code style="background:#1e293b;padding:2px 6px;border-radius:4px;">sources</code>
                format and our own
                <code style="background:#1e293b;padding:2px 6px;border-radius:4px;">queries</code>
                format automatically.
            </div>""")
            with gr.Row():
                with gr.Column():
                    file_in        = gr.File(label="Upload input JSON", file_types=[".json"])
                    batch_strategy = gr.Dropdown(
                        choices=[
                            ("Structured", "structured"),
                            ("Zero-shot",               "zero_shot"),
                            ("Few-shot",                "few_shot"),
                            ("Chain-of-thought",        "chain_of_thought"),
                        ],
                        value="structured",
                        label="Prompting strategy",
                    )
                    batch_btn = gr.Button("⚡  Run batch inference", variant="primary")
                with gr.Column():
                    batch_log = gr.Textbox(label="Progress log", lines=14, interactive=False)
                    file_out  = gr.File(label="Download output_payload.json")
            batch_btn.click(fn=batch_inference, inputs=[file_in, batch_strategy],
                            outputs=[batch_log, file_out])
    
        # ── Tab 4: Results & Ablation ─────────────────────────────────────────
        with gr.Tab("📊  Results & Ablation"):
            gr.HTML(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">

              <div class="panel">
                <div class="panel-title">Generation — RAG vs No-Context Baseline (23 metrics)</div>
                <table style="width:100%;color:#94a3b8;font-size:0.82rem;border-collapse:collapse;">
                  <tr style="color:#64748b;font-size:0.75rem;">
                    <td style="padding:3px 0;">Metric</td>
                    <td style="text-align:right;">Baseline</td>
                    <td style="text-align:right;color:#86efac;">RAG System</td>
                  </tr>
                  {trow("BLEU-1",             "0.1325", "0.2847")}
                  {trow("BLEU-2",             "0.0629", "0.2092")}
                  {trow("BLEU-4",             "0.0245", "0.1393")}
                  {trow("ROUGE-1",            "0.2145", "0.3725")}
                  {trow("ROUGE-2",            "0.0556", "0.2095")}
                  {trow("ROUGE-L",            "0.1543", "0.3280")}
                  {trow("BERTScore F1",       "0.7815", "0.8334")}
                  {trow("METEOR",             "0.2184", "0.3206")}
                  {trow("Answer F1",          "0.1853", "0.3379")}
                  {trow("Answer Correctness", "0.8141", "0.8398")}
                  {trow("Faithfulness",       "0.1137",  "0.6547")}
                  {trow("Hallucination %",    "100%",   "10.9%")}
                </table>
                <div style="color:#64748b;font-size:0.7rem;margin-top:8px;">
                  *** p&lt;0.001 · Faithfulness: content-word overlap, full chunks, threshold=0.40
                </div>
              </div>

              <div class="panel">
                <div class="panel-title">Retrieval — Full System vs Ablations</div>
                <table style="width:100%;color:#94a3b8;font-size:0.82rem;border-collapse:collapse;">
                  <tr style="color:#64748b;font-size:0.75rem;">
                    <td style="padding:3px 0;">Configuration</td>
                    <td style="text-align:right;">ROUGE-L</td>
                    <td style="text-align:right;">NDCG@5</td>
                  </tr>
                  {trow_abl("Full system ✓",   "0.3533", "0.9276", "#86efac")}
                  {trow_abl("No reranker",      "0.2921", "0.8818")}
                  {trow_abl("Dense only",       "0.3171", "0.9295")}
                  {trow_abl("BM25 only",        "0.2805", "0.7358")}
                  {trow_abl("Fixed chunking",   "0.1611", "0.0333")}
                  {trow_abl("Zero-shot prompt", "0.2193", "0.9276")}
                  {trow_abl("No RAG baseline",  "0.1495", "0.000")}
                </table>
              </div>

              <div class="panel">
                <div class="panel-title">Retrieval metrics — RAG system</div>
                <table style="width:100%;color:#94a3b8;font-size:0.82rem;border-collapse:collapse;">
                  {"".join([
                    f'<tr style="border-top:1px solid #1e293b">'
                    f'<td style="padding:5px 0;color:#e2e8f0;font-weight:600;">{m}</td>'
                    f'<td style="text-align:right;color:#86efac;font-weight:600;">{v}</td></tr>'
                    for m, v in [
                        ("MRR@1",            "0.8478"),
                        ("MRR@3",            "0.9058"),
                        ("MRR@5",            "0.9080"),
                        ("Recall@5",         "0.9783"),
                        ("Recall@10",        "0.9783"),
                        ("NDCG@5",           "0.9130"),
                        ("Context Precision","0.6783"),
                        ("Mean Latency",     "~12s / query"),
                    ]
                  ])}
                </table>
              </div>

              <div class="panel">
                <div class="panel-title">Ablation — ROUGE-L cost of removing each component</div>
                <table style="width:100%;color:#94a3b8;font-size:0.82rem;border-collapse:collapse;">
                  <tr style="color:#64748b;font-size:0.75rem;">
                    <td style="padding:3px 0;">Component removed</td>
                    <td style="text-align:right;">ROUGE-L cost</td>
                    <td style="text-align:right;">p-value</td>
                  </tr>
                  {"".join([
                    f'<tr style="border-top:1px solid #1e293b">'
                    f'<td style="padding:5px 0;color:#e2e8f0;">{m}</td>'
                    f'<td style="text-align:right;color:#f87171;font-weight:600;">{c}</td>'
                    f'<td style="text-align:right;font-size:0.7rem;color:#7dd3fc;">{p}</td></tr>'
                    for m, c, p in [
                        ("Semantic chunking",    "-0.1922", "p=0.0007"),
                        ("Structured prompting", "-0.1340", "p=0.0016"),
                        ("RAG entirely",         "-0.2038", "p=0.0002"),
                        ("Cross-encoder reranker","-0.0611","p=0.2151"),
                        ("BM25 (dense only)",    "-0.0362", "p=0.3795"),
                    ]
                  ])}
                </table>
                <div style="color:#64748b;font-size:0.7rem;margin-top:8px;">
                  Paired t-test on 30 queries per configuration
                </div>
              </div>

            </div>
            """)

        # ── Tab 5: System Info ────────────────────────────────────────────────
        with gr.Tab("⚙️  System Info"):
            gr.HTML(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">

              <div class="panel">
                <div class="panel-title">Pipeline configuration</div>
                <table style="width:100%;color:#94a3b8;font-size:0.85rem;border-collapse:collapse;">
                  <tr><td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Chunking</td>
                      <td>Semantic · threshold=0.45 · 3,356 chunks</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Embedding</td>
                      <td>BAAI/bge-small-en-v1.5 · {emb_dim}d</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Retrieval</td>
                      <td>Hybrid BM25 + Dense FAISS + RRF</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Reranker</td>
                      <td>ms-marco-MiniLM-L-6-v2 (cross-encoder)</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">LLM</td>
                      <td>Qwen/Qwen2.5-0.5B-Instruct</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Prompting</td>
                      <td>Structured/constrained (4 strategies tested)</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Built</td>
                      <td>{created}</td></tr>
                </table>
              </div>

              <div class="panel">
                <div class="panel-title">Corpus statistics</div>
                <table style="width:100%;color:#94a3b8;font-size:0.85rem;border-collapse:collapse;">
                  <tr><td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Wikipedia</td>
                      <td>184 docs · 252,937 words · CC-BY-SA 4.0</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Wikibooks</td>
                      <td>21 docs · 7,901 words · CC-BY-SA 4.0</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Blog</td>
                      <td>32 docs · 57,977 words · Public</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Total</td>
                      <td>237 docs · ~318,000 words</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Chunks</td>
                      <td>{chunks_n:,} semantic chunks · {emb_dim}d vectors</td></tr>
                  <tr style="border-top:1px solid #1e293b">
                      <td style="padding:6px 0;color:#e2e8f0;font-weight:600;">Benchmark</td>
                      <td>116 QA pairs · 6 types · LLM-generated</td></tr>
                </table>
              </div>

              <div class="panel" style="grid-column:1/-1;">
                <div class="panel-title">Design decisions — evidence for each choice</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;
                            color:#94a3b8;font-size:0.85rem;line-height:1.7;">
                  <div>🔵 <strong style="color:#e2e8f0;">Semantic chunking</strong> over fixed-size
                       — most impactful component: removing it costs -0.1922 ROUGE-L (p=0.0007)</div>
                  <div>🟢 <strong style="color:#e2e8f0;">BGE over MiniLM</strong>
                       — trained specifically on retrieval tasks; higher NDCG on passage benchmarks</div>
                  <div>🟣 <strong style="color:#e2e8f0;">Hybrid over dense</strong>
                       — RRF captures both semantic and keyword signals; BM25-only NDCG@5 drops to 0.74</div>
                  <div>🟠 <strong style="color:#e2e8f0;">Cross-encoder reranker</strong>
                       — joint query-chunk scoring; +0.06 ROUGE-L improvement over no-reranker</div>
                  <div>🔴 <strong style="color:#e2e8f0;">Structured prompting</strong>
                       — explicit rules anchor 0.5B model to context; removing costs -0.134 ROUGE-L (p=0.0016)</div>
                  <div>⚪ <strong style="color:#e2e8f0;">Faithfulness metric</strong>
                       — content-word overlap with full retrieved chunks; 89.1% of answers grounded</div>
                </div>
              </div>

            </div>
            """)

    gr.HTML("""
    <div style="text-align:center;color:#334155;font-size:0.78rem;
                margin-top:24px;padding-top:16px;border-top:1px solid #1e293b;">
        COMP64702 Transforming Text Into Meaning &nbsp;·&nbsp;
        University of Manchester &nbsp;·&nbsp;
        East Asian Cuisine RAG System &nbsp;·&nbsp;
        Qwen2.5-0.5B-Instruct + BGE-small-en-v1.5
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
