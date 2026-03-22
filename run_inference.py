#!/usr/bin/env python
# run_inference.py
# =============================================================================
# COMP64702 RAG Culinary Assistant — Demo Day Inference Script
# =============================================================================
# Usage:
#   python run_inference.py --input input_payload.json --output output_payload.json
#
# This script runs the full RAG inference pipeline on a JSON file of queries
# and writes answers to an output JSON file in the required format.
#
# Requirements:
#   - vector_store/ folder must exist (built by 03_ingestion.ipynb)
#   - All packages in requirements.txt must be installed
#   - HF_TOKEN must be set in .env file
# =============================================================================

import argparse
import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import login

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from embedder  import Embedder
from retriever import Retriever
from generator import Generator, build_context


def load_input(input_path):
    """Loads queries from input JSON file."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats:
    # {"queries": [{"id": ..., "question": ...}]}   <- demo day format
    # [{"id": ..., "question": ...}]                <- flat list
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unrecognised input format in {input_path}")


def run_pipeline(queries, embedder, retriever, generator,
                 retrieval_k=5, strategy="structured"):
    """
    Runs the full RAG pipeline on a list of queries.

    Args:
        queries    : list of {"id": ..., "question": ...} dicts
        embedder   : Embedder instance
        retriever  : Retriever instance
        generator  : Generator instance
        retrieval_k: number of chunks to retrieve (max 5 per spec)
        strategy   : prompting strategy
    Returns:
        list of {"id": ..., "answer": ...} dicts
    """
    outputs = []

    for i, item in enumerate(queries, 1):
        query    = item.get("question", item.get("query", ""))
        query_id = item.get("id", f"Q{i}")

        print(f"[{i:03d}/{len(queries)}] {query[:70]}")

        start = time.time()

        # Step 1: Retrieve relevant chunks
        retrieved = retriever.retrieve(query, initial_k=20, final_k=retrieval_k)

        # Step 2: Build context from retrieved chunks
        context = build_context(retrieved, max_words=600)

        # Step 3: Generate answer
        answer = generator.generate(
            query=query,
            context=context,
            strategy=strategy,
            max_new_tokens=200,
            temperature=0.3,
        )

        elapsed = time.time() - start
        print(f"         → {elapsed:.1f}s | {answer[:80]}...")

        outputs.append({
            "id":     query_id,
            "answer": answer,
        })

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="COMP64702 RAG Culinary Assistant — Inference"
    )
    parser.add_argument(
        "--input",  "-i",
        default="data/benchmark/test_set_queries_only.json",
        help="Path to input JSON file with queries",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/output_payload.json",
        help="Path to write output JSON file",
    )
    parser.add_argument(
        "--vector_store",
        default="vector_store",
        help="Path to vector store folder",
    )
    parser.add_argument(
        "--strategy",
        default="structured",
        choices=["zero_shot", "few_shot", "chain_of_thought", "structured"],
        help="Prompting strategy to use",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (max 5 per spec)",
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  COMP64702 RAG Culinary Assistant — Inference")
    print("="*60)

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...")
    embedder  = Embedder("BAAI/bge-small-en-v1.5")
    retriever = Retriever(args.vector_store, embedder=embedder)
    generator = Generator("Qwen/Qwen2.5-0.5B-Instruct")

    # ── Load queries ──────────────────────────────────────────────────────────
    print(f"\nLoading queries from: {args.input}")
    queries = load_input(args.input)
    print(f"Queries to process: {len(queries)}")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print(f"\nRunning inference (strategy={args.strategy}, k={args.k})...\n")
    start_total = time.time()

    outputs = run_pipeline(
        queries, embedder, retriever, generator,
        retrieval_k=args.k,
        strategy=args.strategy,
    )

    total_time = time.time() - start_total

    # ── Save output ───────────────────────────────────────────────────────────
    output_payload = {
        "generated_at": datetime.now().isoformat(),
        "model":        "Qwen/Qwen2.5-0.5B-Instruct",
        "retrieval":    "hybrid_bm25_dense_rrf_crossencoder_rerank",
        "strategy":     args.strategy,
        "outputs":      outputs,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done — {len(outputs)} answers generated")
    print(f"  Total time    : {total_time:.1f}s")
    print(f"  Avg per query : {total_time/len(outputs):.1f}s")
    print(f"  Output saved  : {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
