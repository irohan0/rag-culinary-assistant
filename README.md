# 🍜 East Asian Cuisine RAG Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-UI-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple?style=for-the-badge)

**A production-grade Retrieval-Augmented Generation (RAG) system for East Asian cuisine question answering, built entirely from scratch without any RAG frameworks.**

[Results](#results) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Methodology](#methodology) • [Ablation Study](#ablation-study) • [Project Structure](#project-structure)

</div>

---

## Overview

This project implements a complete **end-to-end RAG pipeline** for a culinary question-answering assistant specialising in East Asian cuisine (Chinese, Japanese, Korean). Built as part of the COMP64702 *Transforming Text Into Meaning* coursework at the University of Manchester.

Rather than using off-the-shelf RAG frameworks like LangChain or LlamaIndex, every component was **designed, implemented, and evaluated from scratch** — from the web scraper through to the statistical significance tests — demonstrating deep understanding of how modern NLP systems work under the hood.

### What makes this project stand out

- **No RAG framework used** — every component built from first principles
- **Hybrid retrieval** combining dense FAISS and sparse BM25 fused via Reciprocal Rank Fusion
- **Semantic chunking** using cosine similarity between sentence embeddings to detect topic boundaries
- **Cross-encoder reranking** for precision improvement over bi-encoder retrieval
- **12 evaluation metrics** across generation, retrieval, and RAG-specific dimensions
- **Ablation study** across 7 system configurations with statistical significance testing
- **Entropy-based confidence scoring** for every generated answer
- **Interactive Gradio UI** with live strategy comparison and one-click batch inference

---

## Results

### Generation Quality — RAG System vs No-Context Baseline

| Metric | No-Context Baseline | **Our RAG System** | Improvement | Significant |
|---|---|---|---|---|
| ROUGE-1 | 0.2145 | **0.4103** | +91.3% | — |
| ROUGE-2 | 0.0556 | **0.2522** | +353.6% | — |
| ROUGE-L | 0.1543 | **0.3668** | +137.7% | ✅ p < 0.05 |
| BERTScore F1 | 0.7815 | **0.8425** | +7.8% | ✅ p < 0.05 |
| METEOR | — | **0.3354** | — | — |
| Answer F1 | — | **0.3561** | — | — |
| Faithfulness | 0.0% | **95.7%** | — | — |
| Answer Relevance | — | **0.8530** | — | — |
| Confidence | — | **0.9969** | — | — |

### Retrieval Quality

| Metric | Dense Only | **Full System** | Improvement |
|---|---|---|---|
| MRR@5 | 0.9500 | **0.9500** | — |
| NDCG@5 | 0.9295 | **0.9276** | — |
| Context Precision | 0.6800 | **0.6933** | +0.013 |
| BM25 only MRR@5 | 0.7167 | **0.9500** | +0.233 |

### Results by Question Type

| Question Type | ROUGE-L | BERTScore | METEOR |
|---|---|---|---|
| Factual | **0.4072** | **0.8551** | **0.4250** |
| Ingredient | 0.4030 | 0.8513 | 0.3302 |
| Cultural | 0.3766 | 0.8387 | 0.3449 |
| Comparative | 0.2353 | 0.8426 | 0.1363 |
| Procedural | 0.2543 | 0.8131 | 0.2465 |

---

## Ablation Study

A systematic ablation study was conducted across **7 configurations**, removing one component at a time to quantify each component's contribution. All comparisons use paired t-tests against the full system.

| Configuration | ROUGE-L | BERTScore | NDCG@5 | MRR@5 | Cost vs Full |
|---|---|---|---|---|---|
| **Full system** ✅ | **0.3533** | **0.8381** | 0.9276 | **0.9500** | — |
| No reranker | 0.2921 | 0.8194 | 0.8818 | 0.8806 | -0.0611 |
| Dense only (no BM25) | 0.3171 | 0.8237 | **0.9295** | 0.9500 | -0.0362 |
| BM25 only (no dense) | 0.2805 | 0.8200 | 0.7358 | 0.7167 | -0.0727 |
| Fixed chunking (256w) | 0.1611 | 0.7874 | 0.0333 | 0.0333 | **-0.1922** ✅ sig |
| Zero-shot prompting | 0.2193 | 0.7946 | 0.9276 | 0.9500 | **-0.1340** ✅ sig |
| No RAG (baseline) | 0.1495 | 0.7683 | 0.0000 | 0.0000 | **-0.2038** ✅ sig |

**Key findings:**
- Semantic chunking is the **most impactful component** — removing it costs -0.19 ROUGE-L (p=0.0007)
- Structured prompting costs -0.13 ROUGE-L if replaced with zero-shot (p=0.0016)
- RAG context improves ROUGE-L by +0.20 over no-context baseline (p=0.0002)
- Hybrid retrieval outperforms BM25-only by +0.07 ROUGE-L on NDCG@5 (+0.89)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION  (offline)                       │
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
│                           Answer + confidence score          │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+, Conda recommended
- ~4GB disk for models
- Free HuggingFace account

### Installation

```bash
git clone https://github.com/irohan0/rag-culinary-assistant.git
cd rag-culinary-assistant

conda create -n rag-project python=3.11
conda activate rag-project

pip install -r requirements.txt

echo "HF_TOKEN=your_token_here" > .env
```

### Run the UI

```bash
python app.py
# Opens at http://localhost:7860
```

### Command-line inference

```bash
python run_inference.py --input input_payload.json --output outputs/output_payload.json
```

### Rebuild from scratch (run notebooks in order)

```
01_data_collection.ipynb      → scrape 237 documents
02_benchmark_creation.ipynb   → generate 116 QA pairs
03_ingestion.ipynb            → chunk + embed + index
04_inference.ipynb            → run RAG pipeline
05_evaluation.ipynb           → compute all metrics
06_ablation_evaluation.ipynb  → ablation study + confidence
```

> **Note:** `vector_store/` index files are not committed (too large). Run `03_ingestion.ipynb` to rebuild them in ~10 minutes.

---

## Methodology

### 1. Data Collection

| Source | Documents | Words | License |
|---|---|---|---|
| Wikipedia | 184 | 252,937 | CC-BY-SA 4.0 |
| Wikibooks | 21 | 7,901 | CC-BY-SA 4.0 |
| Around the World in 80 Cuisines | 32 | 57,977 | Public |
| **Total** | **237** | **~318,000** | — |

All scraping performed ethically: 1.5s crawl delay, transparent User-Agent, academic use only.

### 2. Benchmark Dataset

116 QA pairs auto-generated using `Qwen2.5-0.5B-Instruct` as an LLM-assisted annotation tool. Manual comparative and multi-hop pairs added for cross-document reasoning coverage.

| Type | Count | Example |
|---|---|---|
| Factual | 26 | "What is wagyu beef?" |
| Procedural | 31 | "How is miso paste produced?" |
| Ingredient | 25 | "What goes into mapo tofu?" |
| Cultural | 29 | "Why is kimchi culturally significant?" |
| Comparative | 3 | "How does hot pot differ from shabu-shabu?" |
| Multi-hop | 2 | "How is fermentation used across East Asian cuisines?" |

### 3. Chunking — 4 Strategies Compared

| Strategy | Chunks | Avg Words | Selected |
|---|---|---|---|
| Fixed-size 256w | ~4,100 | 256 | — |
| Fixed-size 512w | ~2,200 | 512 | — |
| Sentence-based | ~3,800 | ~200 | — |
| **Semantic** | **3,356** | **~180** | ✅ |

**Selected: Semantic chunking** — splits on topic shifts detected by cosine similarity drops between adjacent sentence embeddings (threshold=0.45). Ablation confirms this is the most impactful component (-0.19 ROUGE-L if removed, p=0.0007).

### 4. Embedding — 2 Models Compared

| Model | Dim | Selected |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | — |
| **BAAI/bge-small-en-v1.5** | **384** | ✅ |

### 5. Retrieval — 3 Strategies Compared

| Strategy | ROUGE-L | NDCG@5 | Selected |
|---|---|---|---|
| Dense only | 0.3171 | 0.9295 | — |
| BM25 only | 0.2805 | 0.7358 | — |
| **Hybrid + Reranking** | **0.3533** | **0.9276** | ✅ |

### 6. Prompting — 4 Strategies Compared

| Strategy | ROUGE-L | Selected |
|---|---|---|
| Zero-shot | 0.2193 | — |
| Few-shot | — | — |
| Chain-of-thought | — | — |
| **Structured** | **0.3533** | ✅ |

### 7. Evaluation — 12 Metrics

**Generation:** ROUGE-1/2/L, BERTScore F1, METEOR, Answer F1, Exact Match

**RAG-specific:** Faithfulness (95.7%), Answer Relevance (0.853), Context Precision (0.693)

**Retrieval:** MRR@K, Recall@K, Precision@K, NDCG@K (K ∈ {1,3,5})

**Confidence:** Entropy-based token probability scoring (mean: 0.9969)

Statistical significance via paired t-tests (p < 0.05 threshold).

---

## Project Structure

```
rag-culinary-assistant/
├── app.py                           # Gradio UI (5 tabs + live results)
├── run_inference.py                 # CLI inference script
├── requirements.txt
├── .env                             # HF_TOKEN (not committed)
│
├── src/
│   ├── chunker.py                   # Fixed, sentence, semantic chunking
│   ├── embedder.py                  # BGE embedding wrapper
│   ├── retriever.py                 # Dense, BM25, hybrid + reranking + MMR
│   ├── generator.py                 # Qwen + 4 prompting strategies
│   └── evaluator.py                 # All 12 metrics + confidence scoring
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_benchmark_creation.ipynb
│   ├── 03_ingestion.ipynb
│   ├── 04_inference.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_ablation_evaluation.ipynb
│
├── data/
│   ├── raw/                         # Scraped documents (not committed)
│   └── benchmark/
│       ├── benchmark_full.json      # 116 QA pairs
│       ├── train_set.json           # 92 training pairs
│       └── test_set_queries_only.json
│
├── vector_store/
│   ├── index.faiss                  # FAISS dense index (not committed)
│   ├── chunks.pkl                   # Chunk text + metadata (not committed)
│   ├── bm25.pkl                     # BM25 index (not committed)
│   └── metadata.json
│
└── outputs/
    ├── output_payload.json          # Demo day submission file
    ├── evaluation/
    │   └── evaluation_report.json
    └── ablation/
        └── ablation_report.json     # Full ablation results
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
| **Evaluation** | ROUGE, BERTScore, METEOR, SciPy |
| **Scraping** | BeautifulSoup4, Requests |

---

## Academic Context

**Course:** COMP64702 Transforming Text Into Meaning
**Institution:** University of Manchester
**Year:** 2024/25
**Reference:** Yu et al., 2025. *Evaluation of Retrieval-Augmented Generation: A Survey*. Springer.

---

## Author

**Rohan Inamdar** — MSc Data Science, University of Manchester

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-irohan0-black?style=flat&logo=github)](https://github.com/irohan0)

---

<div align="center">
<i>Built without RAG frameworks — every component designed, implemented, and ablation-tested from scratch.</i>
</div>