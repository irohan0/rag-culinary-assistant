# 🍜 East Asian Cuisine RAG Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-UI-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple?style=for-the-badge)

**A production-grade Retrieval-Augmented Generation (RAG) system for East Asian cuisine question answering, built entirely from scratch without any RAG frameworks.**

[Results](#results) • [UI Screenshots](#ui-screenshots) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Methodology](#methodology) • [Ablation Study](#ablation-study) • [Project Structure](#project-structure)

</div>

---

## Overview

This project implements a complete **end-to-end RAG pipeline** for a culinary question-answering assistant specialising in East Asian cuisine (Chinese, Japanese, Korean). Built as part of the COMP64702 *Transforming Text Into Meaning* coursework at the University of Manchester.

Rather than using off-the-shelf RAG frameworks like LangChain or LlamaIndex, every component was **designed, implemented, and evaluated from scratch** — demonstrating deep understanding of how modern NLP systems work under the hood.

## Project Link - 
[![GitHub](https://img.shields.io/badge/GitHub-irohan0-black?style=flat&logo=github)](https://github.com/irohan0/rag-culinary-assistant)

### What makes this project stand out

- **No RAG framework used** — every component built from first principles
- **Hybrid retrieval** combining dense FAISS and sparse BM25 fused via Reciprocal Rank Fusion
- **Semantic chunking** using cosine similarity between sentence embeddings to detect topic boundaries
- **Cross-encoder reranking** for precision improvement over bi-encoder retrieval
- **23 evaluation metrics** across generation, retrieval, faithfulness, and efficiency dimensions
- **Ablation study** across 7 system configurations with statistical significance testing
- **Corrected faithfulness metric** — content-word overlap with full retrieved chunks, stopwords removed
- **Interactive Gradio UI** with live strategy comparison and one-click batch inference

---

## Results

### Generation Quality — RAG System vs No-Context Baseline

| Metric | No-Context Baseline | **Our RAG System** | Improvement | p-value |
|---|---|---|---|---|
| BLEU-1 | 0.1325 | **0.2847** | +114.9% | p < 0.001 |
| BLEU-2 | 0.0629 | **0.2092** | +232.6% | p < 0.001 |
| BLEU-4 | 0.0245 | **0.1393** | +468.6% | p < 0.001 |
| ROUGE-1 | 0.2145 | **0.3725** | +73.7% | p < 0.001 |
| ROUGE-2 | 0.0556 | **0.2095** | +276.8% | p < 0.001 |
| ROUGE-L | 0.1543 | **0.3280** | +112.6% | p < 0.001 |
| BERTScore F1 | 0.7815 | **0.8334** | +6.6% | p < 0.001 |
| METEOR | 0.2184 | **0.3206** | +46.8% | p < 0.001 |
| Answer F1 | 0.1853 | **0.3379** | +82.4% | p < 0.001 |
| Answer Correctness | 0.8141 | **0.8398** | +3.2% | p < 0.001 |
| Faithfulness† | 0.1137 | **0.6547** | — | p < 0.001 |
| Hallucination Rate† | 100% | **10.9%** | -89.1pp | p < 0.001 |

> † Faithfulness computed as content-word overlap (stopwords removed) with full retrieved chunk text via chunk_id lookup, threshold = 0.40. Hallucination rate = fraction of answers below threshold.

### Retrieval Quality

| Metric | Score |
|---|---|
| MRR@1 | 0.8478 |
| MRR@3 | 0.9058 |
| MRR@5 | **0.9080** |
| Recall@5 | **0.9783** |
| Recall@10 | 0.9783 |
| NDCG@5 | **0.9130** |
| Context Precision | 0.6783 |
| Mean Latency | ~12s / query |

### Results by Question Type

| Question Type | N | ROUGE-L | BERTScore | BLEU-1 | METEOR |
|---|---|---|---|---|---|
| Factual | 21 | **0.3454** | **0.8374** | **0.3062** | **0.3345** |
| Ingredient | 16 | 0.3332 | **0.8464** | 0.2817 | 0.3400 |
| Cultural | 27 | 0.3415 | 0.8267 | 0.2846 | 0.3119 |
| Procedural | 26 | 0.3047 | 0.8289 | 0.2741 | 0.3148 |
| Comparative | 2 | 0.2232 | 0.8384 | 0.2220 | 0.2104 |

> Note: Comparative questions have only n=2 samples — insufficient for reliable per-type statistics.

### Summary Scorecard

| | Value |
|---|---|
| Total metrics evaluated | 23 |
| Metrics where RAG wins | 20 / 23 |
| Significant improvements (p < 0.001) | 16 / 23 |
| Queries evaluated | 92 (train set) |

---

## UI Screenshots

<!-- ═══════════════════════════════════════════════════════════════════════
     HOW TO ADD SCREENSHOTS:
     1. Run the app:  python app.py
     2. Take a screenshot of each tab
     3. Save them in a folder called  docs/screenshots/  in the repo root
     4. Replace the placeholder lines below with your actual image paths

     Example after adding screenshots:
         ![Ask a Question tab](docs/screenshots/ask_tab.png)

     To take screenshots on Windows: Win + Shift + S
     ═══════════════════════════════════════════════════════════════════════ -->

### Ask a Question

> *Screenshot of the Ask a Question tab — shows the question input, retrieved chunks, and generated answer*

![Ask Tab](docs/screenshots/ask_tab.png)

---

### Compare Prompting Strategies

> *Screenshot of the Compare Strategies tab — shows all 4 strategies side by side for the same question*

![Compare Tab](docs/screenshots/compare_tab.png)

---

### Results & Ablation

> *Screenshot of the Results & Ablation tab — shows the full evaluation tables*

![Results Tab](docs/screenshots/results_tab.png)

---

### Batch Inference

> *Screenshot of the Batch Inference tab — shows the file upload and progress log*

![Batch Tab](docs/screenshots/batch_tab.png)

---

### System Info

> *Screenshot of the System Info tab — shows pipeline configuration and design decisions*

![System Info Tab](docs/screenshots/system_tab.png)

---

> **To add your own screenshots:** Create a `docs/screenshots/` folder in the repo, save your screenshots there with the filenames above, then commit and push. The images will automatically appear here on GitHub.

---

## Ablation Study

A systematic ablation study was conducted across **7 configurations**, removing one component at a time to quantify each component's contribution. All comparisons use paired t-tests against the full system on 30 queries per configuration.

| Configuration | ROUGE-L | BERTScore | NDCG@5 | MRR@5 | Cost vs Full |
|---|---|---|---|---|---|
| **Full system** ✅ | **0.3533** | **0.8381** | 0.9276 | **0.9500** | — |
| No reranker | 0.2921 | 0.8194 | 0.8818 | 0.8806 | -0.0611 |
| Dense only (no BM25) | 0.3171 | 0.8237 | **0.9295** | 0.9500 | -0.0362 |
| BM25 only (no dense) | 0.2805 | 0.8200 | 0.7358 | 0.7167 | -0.0727 |
| Fixed chunking (256w) | 0.1611 | 0.7874 | 0.0333 | 0.0333 | **-0.1922** ✅ p=0.0007 |
| Zero-shot prompting | 0.2193 | 0.7946 | 0.9276 | 0.9500 | **-0.1340** ✅ p=0.0016 |
| No RAG (baseline) | 0.1495 | 0.7683 | 0.0000 | 0.0000 | **-0.2038** ✅ p=0.0002 |

**Key findings:**
- Semantic chunking is the **most impactful single component** — removing it costs -0.1922 ROUGE-L (p=0.0007)
- Structured prompting costs -0.1340 ROUGE-L if replaced with zero-shot (p=0.0016)
- RAG context improves ROUGE-L by +0.2038 over no-context baseline (p=0.0002)
- Hybrid retrieval outperforms BM25-only by +0.0727 ROUGE-L and +0.19 NDCG@5

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION  (offline, once)                 │
│                                                              │
│  237 docs  →  Semantic  →  BGE      →  FAISS + BM25         │
│  318k words   chunker      embedder    vector store          │
│               3,356 chunks  384-dim    saved to disk         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE  (per query)                     │
│                                                              │
│  Question → BGE embed → FAISS dense  ─┐                     │
│                       → BM25 sparse  ─┼→ RRF fusion         │
│                                        │                     │
│                             fused list → Cross-encoder       │
│                                          reranker            │
│                                          ↓                   │
│                              Top-5 most relevant chunks      │
│                                          ↓                   │
│                           Structured prompt builder          │
│                                          ↓                   │
│                        Qwen2.5-0.5B-Instruct                 │
│                                          ↓                   │
│                                     Final answer             │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+, Conda recommended
- ~4GB disk space for models
- Free HuggingFace account for Qwen access

### Installation

```bash
git clone https://github.com/irohan0/rag-culinary-assistant.git
cd rag-culinary-assistant

conda create -n rag-project python=3.11
conda activate rag-project

pip install -r requirements.txt

echo "HF_TOKEN=your_token_here" > .env
# Get your free token at: https://huggingface.co/settings/tokens
```

### Run the UI

```bash
python app.py
# Opens automatically at http://localhost:7860
```

### Command-line inference

```bash
# Handles demo day sources format and our own queries format automatically
python run_inference.py --input "east_asia_sample_qa.json" --output outputs/output_payload.json
```

### Run notebooks in order

```
01_data_collection.ipynb      → scrape 237 documents (~5 min)
02_benchmark_creation.ipynb   → generate 116 QA pairs
03_ingestion.ipynb            → chunk + embed + index (~10 min)
04_inference.ipynb            → run RAG pipeline on train set
05_evaluation.ipynb           → compute all 23 metrics
06_ablation_evaluation.ipynb  → ablation study across 7 configs
```

> **Note:** `vector_store/` index files are not committed due to size. Run `03_ingestion.ipynb` to rebuild them in ~10 minutes.

---

## Methodology

### 1. Data Collection

| Source | Documents | Words | Licence |
|---|---|---|---|
| Wikipedia | 184 | 252,937 | CC-BY-SA 4.0 |
| Wikibooks | 21 | 7,901 | CC-BY-SA 4.0 |
| Around the World in 80 Cuisines (Blog) | 32 | 57,977 | Public |
| **Total** | **237** | **~318,000** | — |

All scraping performed ethically: 1.5s crawl delay, transparent User-Agent, academic use only.

### 2. Benchmark Dataset

116 QA pairs auto-generated using `Qwen2.5-0.5B-Instruct` as an LLM-assisted annotation tool, with manual comparative and multi-hop pairs added for cross-document reasoning. Split 80/20 into train (92) and test (24) sets.

### 3. Chunking — 4 Strategies Compared

| Strategy | Chunks | Avg Words | Selected |
|---|---|---|---|
| Fixed-size 256w | ~4,100 | 256 | — |
| Fixed-size 512w | ~2,200 | 512 | — |
| Sentence-based | ~3,800 | ~200 | — |
| **Semantic (threshold=0.45)** | **3,356** | **~180** | ✅ |

Ablation confirms this is the most impactful component: **-0.1922 ROUGE-L if removed (p=0.0007)**.

### 4. Embedding — 2 Models Compared

| Model | Dim | Selected |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | — |
| **BAAI/bge-small-en-v1.5** | **384** | ✅ Retrieval-optimised contrastive training |

### 5. Retrieval — 3 Strategies Compared

| Strategy | ROUGE-L | NDCG@5 | Selected |
|---|---|---|---|
| Dense only | 0.3171 | 0.9295 | — |
| BM25 only | 0.2805 | 0.7358 | — |
| **Hybrid + RRF + Reranking** | **0.3533** | **0.9276** | ✅ |

### 6. Prompting — 4 Strategies Compared

| Strategy | ROUGE-L | Selected |
|---|---|---|
| Zero-shot | 0.2193 | — |
| Few-shot | — | — |
| Chain-of-thought | — | — |
| **Structured/constrained** | **0.3533** | ✅ |

### 7. Evaluation — 23 Metrics

**Generation:** BLEU-1/2/4, ROUGE-1/2/L, BERTScore F1, METEOR, Answer F1, Exact Match, Answer Relevance, Answer Correctness

**RAG-specific:** Faithfulness (content-word overlap, stopwords removed, full chunks), Hallucination Rate, Context Precision

**Retrieval:** MRR@1/3/5, Recall@5/10, NDCG@5/10

**Efficiency:** Mean latency per query (~12s)

Statistical significance via paired t-tests (p < 0.001 for 16 out of 23 metrics).

---

## Project Structure

```
rag-culinary-assistant/
├── app.py                           # Gradio UI (5 tabs)
├── run_inference.py                 # CLI inference — handles all JSON formats
├── requirements.txt
├── east_asia_sample_qa.json         # Demo day sample input
├── .env                             # HF_TOKEN (not committed)
│
├── src/
│   ├── chunker.py                   # Fixed, sentence, semantic chunking
│   ├── embedder.py                  # BGE embedding wrapper
│   ├── retriever.py                 # Dense, BM25, hybrid + RRF + reranking + MMR
│   ├── generator.py                 # Qwen + 4 prompting strategies
│   └── evaluator.py                 # All 23 metrics + corrected faithfulness
│
├── notebooks/
│   ├── 01 data_collection.ipynb
│   ├── 02 benchmark_creation.ipynb
│   ├── 03 ingestion.ipynb
│   ├── 04 inference.ipynb
│   ├── 05 evaluation.ipynb
│   └── 06 ablation_evaluation.ipynb
│
├── data/
│   ├── raw/                         # Scraped documents (not committed)
│   └── benchmark/
│       ├── benchmark_full.json      # All 116 QA pairs
│       ├── train_set.json           # 92 training pairs
│       └── test_set_queries_only.json
│
├── vector_store/
│   ├── index.faiss                  # FAISS dense index (not committed)
│   ├── chunks.pkl                   # Chunk text + metadata (not committed)
│   ├── bm25.pkl                     # BM25 index (not committed)
│   └── metadata.json
│
├── outputs/
│   ├── train_outputs.json           # Inference outputs (committed for reproducibility)
│   ├── output_payload.json          # Demo day submission file
│   ├── evaluation/
│   │   └── evaluation_report_complete.json
│   └── ablation/
│       └── ablation_report.json
│
└── docs/
    └── screenshots/                 # UI screenshots (add your own here)
        ├── ask_tab.png
        ├── compare_tab.png
        ├── results_tab.png
        ├── batch_tab.png
        └── system_tab.png
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **LLM** | Qwen2.5-0.5B-Instruct |
| **Embeddings** | BAAI/bge-small-en-v1.5 (Sentence Transformers) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Vector Store** | FAISS IndexFlatIP |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **UI** | Gradio |
| **Evaluation** | ROUGE, BERTScore, METEOR, BLEU, SciPy |
| **Scraping** | BeautifulSoup4, Requests |
| **NLP** | NLTK, scikit-learn |

---

## Academic Context

**Course:** COMP64702 Transforming Text Into Meaning
**Institution:** University of Manchester
**Year:** 2024/25
**Reference:** Yu et al., 2025. *Evaluation of Retrieval-Augmented Generation: A Survey*. Springer.

---

## Authors

**1. Rohan Inamdar** — MSc Data Science, University of Manchester

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/rohan-inamdar-47aa4b251/)
[![GitHub](https://img.shields.io/badge/GitHub-irohan0-black?style=flat&logo=github)](https://github.com/irohan0)

**2. Kavin Sundarr** — MSc Data Science, University of Manchester

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/s-kavin-sundarr-333450238/)
[![GitHub](https://img.shields.io/badge/GitHub-irohan0-black?style=flat&logo=github)](https://github.com/KavinSundarr)

**3. Deepen Khandelwal** — MSc Data Science, University of Manchester

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/deepen-khandelwal-49396b251/)
[![GitHub](https://img.shields.io/badge/GitHub-irohan0-black?style=flat&logo=github)](https://github.com/Deepen-cyph)

---

<div align="center">
<i>Built without RAG frameworks — every component designed, implemented, and ablation-tested from scratch.</i>
<br><br>
<i>23 evaluation metrics · 16 significant at p&lt;0.001 · 7-configuration ablation study</i>
</div>
