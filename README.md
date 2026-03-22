# 🍜 East Asian Cuisine RAG Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-UI-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple?style=for-the-badge)

**A production-grade Retrieval-Augmented Generation (RAG) system for East Asian cuisine question answering, built from scratch without any RAG frameworks.**

[Overview](#overview) • [Results](#results) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Methodology](#methodology) • [Project Structure](#project-structure)

</div>

---

## Overview

This project implements a complete **end-to-end RAG pipeline** for a culinary question-answering assistant specialising in East Asian cuisine (Chinese, Japanese, Korean). Built as part of the COMP64702 *Transforming Text Into Meaning* coursework at the University of Manchester.

Rather than using off-the-shelf RAG frameworks like LangChain or LlamaIndex, every component was **designed, implemented, and evaluated from scratch** — from the web scraper to the evaluation metrics — demonstrating deep understanding of how modern NLP systems work under the hood.

### What makes this interesting

- **No RAG framework used** — every component built from first principles
- **Hybrid retrieval** combining dense semantic search (FAISS) and sparse keyword search (BM25) fused via Reciprocal Rank Fusion
- **Semantic chunking** using cosine similarity between sentence embeddings to detect topic boundaries — outperforms naive fixed-size splitting
- **Cross-encoder reranking** for precision improvement over bi-encoder retrieval
- **Rigorous evaluation** with 8 metrics, statistical significance testing, and baseline comparisons
- **Interactive Gradio UI** with live strategy comparison and batch inference

---

## Results

### Generation Quality — RAG System vs No-Context Baseline

| Metric | No-Context Baseline | **Our RAG System** | Improvement | Significant |
|---|---|---|---|---|
| ROUGE-1 | 0.2145 | **0.4103** | +91.3% | — |
| ROUGE-2 | 0.0556 | **0.2522** | +353.6% | — |
| ROUGE-L | 0.1543 | **0.3668** | +137.7% | ✅ p < 0.05 |
| BERTScore F1 | 0.7815 | **0.8425** | +7.8% | ✅ p < 0.05 |

### Retrieval Quality — Hybrid vs Dense Baseline

| Metric | Dense Only | **Hybrid + Reranking** | Improvement |
|---|---|---|---|
| MRR@5 | — | **measured** | — |
| Recall@5 | — | **measured** | — |
| NDCG@5 | — | **measured** | ✅ p < 0.05 |

### Faithfulness

| Metric | Score |
|---|---|
| Mean faithfulness score | **0.48** |
| Faithful answer rate (>25% overlap) | **95.7%** |

> Faithfulness measures whether generated answers are grounded in retrieved context rather than hallucinated — 95.7% of answers demonstrably use the retrieved evidence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                      (runs once offline)                     │
│                                                             │
│  237 documents  →  Semantic  →  BGE      →  FAISS + BM25   │
│  (Wikipedia,        chunker     embedder    vector store    │
│   Wikibooks,        3,356 chunks  384-dim    saved to disk  │
│   Blog)                                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                       │
│                      (runs per query)                        │
│                                                             │
│  Question  →  BGE embed  →  FAISS (dense)  ─┐              │
│                         →  BM25  (sparse) ──┼→ RRF fusion  │
│                                              │              │
│                                    RRF list  →  Cross-encoder│
│                                              reranker        │
│                                              ↓              │
│                                    Top-5 chunks             │
│                                              ↓              │
│                              Structured prompt builder      │
│                                              ↓              │
│                           Qwen2.5-0.5B-Instruct             │
│                                              ↓              │
│                                       Final answer          │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip
- ~4GB disk space for models
- HuggingFace account (free) for model access

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-culinary-assistant.git
cd rag-culinary-assistant

# 2. Create conda environment
conda create -n rag-project python=3.11
conda activate rag-project

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up HuggingFace token
echo "HF_TOKEN=your_token_here" > .env
# Get your free token at: https://huggingface.co/settings/tokens
```

### Running the pipeline

```bash
# Option 1 — Interactive UI (recommended)
python app.py
# Opens at http://localhost:7860

# Option 2 — Command line inference
python run_inference.py --input input_payload.json --output outputs/output_payload.json

# Option 3 — Run notebooks in order
# 01_data_collection.ipynb
# 02_benchmark_creation.ipynb
# 03_ingestion.ipynb
# 04_inference.ipynb
# 05_evaluation.ipynb
```

> **Note:** The `vector_store/` index files are not included in the repository due to size. Run `03_ingestion.ipynb` to rebuild them (~10 minutes).

---

## Methodology

### 1. Data Collection

Scraped **237 documents** (~318,000 words) from three permitted sources using `BeautifulSoup`:

| Source | Documents | Words | License |
|---|---|---|---|
| Wikipedia | 184 | 252,937 | CC-BY-SA 4.0 |
| Wikibooks | 21 | 7,901 | CC-BY-SA 4.0 |
| Around the World in 80 Cuisines | 32 | 57,977 | Public |

All scraping was performed ethically with a 1.5s crawl delay, transparent User-Agent headers, and data used solely for academic purposes.

### 2. Benchmark Dataset

Generated **116 QA pairs** using `Qwen2.5-0.5B-Instruct` as an LLM-assisted annotation tool — feeding corpus chunks to the model and prompting it to generate diverse question types. Manual comparative and multi-hop questions were added to cover cross-document reasoning.

| Question Type | Count | Example |
|---|---|---|
| Factual | 26 | "What is wagyu beef?" |
| Procedural | 31 | "How is miso paste produced?" |
| Ingredient | 25 | "What goes into mapo tofu?" |
| Cultural | 29 | "What is the significance of kimchi in Korea?" |
| Comparative | 3 | "How does hot pot differ from shabu-shabu?" |
| Multi-hop | 2 | "How is fermentation used across East Asian cuisines?" |

### 3. Chunking — 4 Strategies Compared

| Strategy | Chunks | Avg Words | Description |
|---|---|---|---|
| Fixed-size (256w) | ~4,100 | 256 | Baseline — arbitrary word count |
| Fixed-size (512w) | ~2,200 | 512 | Larger fixed baseline |
| Sentence-based | ~3,800 | ~200 | Respects sentence boundaries |
| **Semantic** ✅ | **3,356** | **~180** | **Splits on topic shifts** |

**Selected: Semantic chunking** — uses cosine similarity between adjacent sentence embeddings (threshold=0.45) to detect topic boundaries. Each chunk is topically coherent, improving retrieval precision.

### 4. Embedding — 2 Models Compared

| Model | Dimension | Description |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | General-purpose baseline |
| **BAAI/bge-small-en-v1.5** ✅ | **384** | **Retrieval-optimised** |

**Selected: BGE-small** — specifically trained on retrieval tasks using contrastive learning. Consistently outperforms MiniLM on passage retrieval benchmarks (MTEB).

### 5. Retrieval — 3 Strategies Compared

| Strategy | Description |
|---|---|
| Dense only | FAISS cosine similarity baseline |
| BM25 only | Keyword-based sparse retrieval baseline |
| **Hybrid + Reranking** ✅ | **BM25 + Dense fused via RRF, then cross-encoder reranked** |

**Selected: Hybrid + Reranking**
- **Reciprocal Rank Fusion (RRF)** merges dense and sparse ranked lists: `score = Σ 1/(60 + rank)`
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) reads query and chunk jointly for more accurate relevance scoring
- Statistically significant improvement over dense-only baseline (p < 0.05)

### 6. Prompting — 4 Strategies Compared

| Strategy | Description |
|---|---|
| Zero-shot | Plain context + question |
| Few-shot | Two worked examples included |
| Chain-of-thought | Explicit reasoning steps |
| **Structured** ✅ | **Explicit rules constraining output format** |

**Selected: Structured prompting** — explicit constraints (answer length, no hallucination, no meta-commentary) reduce the hallucination rate of small models like Qwen 0.5B.

### 7. Evaluation

Eight metrics implemented across retrieval and generation:

**Retrieval:** MRR@K, Recall@K, Precision@K, NDCG@K (K ∈ {1, 3, 5})

**Generation:** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, Exact Match, Faithfulness

Statistical significance tested using paired t-tests (p < 0.05 threshold) against both a dense-only retrieval baseline and a no-context generation baseline.

---

## Project Structure

```
rag-culinary-assistant/
│
├── app.py                          # Gradio web UI (4 tabs)
├── run_inference.py                # CLI inference script
├── requirements.txt
├── .env                            # HF_TOKEN (not committed)
│
├── src/
│   ├── chunker.py                  # Fixed, sentence, semantic chunking
│   ├── embedder.py                 # BGE embedding wrapper
│   ├── retriever.py                # Dense, BM25, hybrid + reranking
│   ├── generator.py                # Qwen + 4 prompting strategies
│   └── evaluator.py                # All evaluation metrics
│
├── notebooks/
│   ├── 01_data_collection.ipynb    # Web scraping pipeline
│   ├── 02_benchmark_creation.ipynb # LLM-assisted QA generation
│   ├── 03_ingestion.ipynb          # Chunking + embedding + indexing
│   ├── 04_inference.ipynb          # Full RAG inference pipeline
│   └── 05_evaluation.ipynb         # Metrics + significance tests
│
├── data/
│   ├── raw/                        # Scraped documents (not committed)
│   │   ├── wikipedia_articles/
│   │   ├── wikibooks_recipes/
│   │   ├── blog_posts/
│   │   └── corpus_manifest.json
│   ├── processed/                  # Chunk JSON files
│   └── benchmark/
│       ├── benchmark_full.json     # All 116 QA pairs
│       ├── train_set.json          # 92 training pairs
│       └── test_set_queries_only.json
│
├── vector_store/
│   ├── index.faiss                 # FAISS dense index (not committed)
│   ├── chunks.pkl                  # Chunk text + metadata (not committed)
│   ├── bm25.pkl                    # BM25 index (not committed)
│   └── metadata.json               # Ingestion config
│
└── outputs/
    ├── train_outputs.json          # Inference on train set
    ├── output_payload.json         # Demo day submission file
    └── evaluation/
        └── evaluation_report.json  # Full evaluation results
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **LLM** | Qwen2.5-0.5B-Instruct (HuggingFace Transformers) |
| **Embeddings** | BAAI/bge-small-en-v1.5 (Sentence Transformers) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **UI** | Gradio |
| **Evaluation** | ROUGE, BERTScore, SciPy (t-tests) |
| **Scraping** | BeautifulSoup4, Requests |
| **NLP** | NLTK, scikit-learn |

---

## Key Technical Decisions

**Why semantic chunking over fixed-size?**
Fixed-size chunking arbitrarily splits documents mid-sentence or mid-topic. Semantic chunking detects topical shifts using embedding similarity, producing self-contained chunks that better match the granularity of user queries.

**Why hybrid retrieval over dense-only?**
Dense retrieval excels at semantic similarity but can miss exact keyword matches (e.g. dish names, ingredient terms). BM25 catches these exactly. RRF fusion captures the strengths of both — demonstrated by statistically significant NDCG improvement.

**Why a cross-encoder reranker?**
Bi-encoder models (like BGE) encode query and document independently, missing fine-grained interactions. A cross-encoder reads both together, giving significantly more accurate relevance scores. Applied only to top-20 candidates to keep latency acceptable.

**Why structured prompting for a 0.5B model?**
Small models are prone to ignoring instructions and hallucinating. Explicit numbered rules constrain the output format and anchor the model to the provided context, reducing hallucination as evidenced by the 95.7% faithfulness rate.

---

## Academic Context

**Course:** COMP64702 Transforming Text Into Meaning
**Institution:** University of Manchester
**Year:** 2024/25

Reference: Yu et al., 2025. *Evaluation of Retrieval-Augmented Generation: A Survey*. Springer.

---

## Author

**Rohan** — MSc Data Science, University of Manchester

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME)

---

<div align="center">
<i>Built without RAG frameworks — every component designed and evaluated from scratch.</i>
</div>
