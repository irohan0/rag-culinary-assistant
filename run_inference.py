#!/usr/bin/env python
# run_inference.py
# =============================================================================
# COMP64702 RAG Culinary Assistant — Demo Day Inference Script
# =============================================================================
# Usage:
#   python run_inference.py --input "D:\text mining\rag project\east_asia_sample_qa.json"
#   python run_inference.py --input input_payload.json --output outputs/output_payload.json
#
# Handles ALL of these input formats automatically:
#
#   Format A — flat queries list (our own test format):
#   {"queries": [{"id": "Q001", "question": "..."}, ...]}
#
#   Format B — sources nested format (demo day format):
#   {"sources": [{"source": "Wikipedia", "questions": [{"question": "...", "answer": "..."}]}]}
#
#   Format C — flat list:
#   [{"id": "Q001", "question": "..."}, ...]
#
#   Format D — sources with top-level questions only:
#   {"sources": [{"source": "...", "questions": [{"question": "..."}]}]}
# =============================================================================

import argparse
import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import login

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BASE_DIR)

from embedder  import Embedder
from retriever import Retriever
from generator import Generator, build_context


# ─────────────────────────────────────────────────────────────────────────────
# Input parsing — handles every format
# ─────────────────────────────────────────────────────────────────────────────

def parse_input(data):
    """
    Normalises any input JSON format into a flat list of:
        {"id": str, "question": str, "gold_answer": str or None}

    Handles:
      - {"queries": [...]}
      - {"sources": [{"source": "...", "questions": [...]}]}
      - [{"question": "...", ...}]
      - {"questions": [...]}
    """
    queries = []
    counter = 1

    # ── Format: {"sources": [...]} ────────────────────────────────────────────
    if isinstance(data, dict) and "sources" in data:
        for source_block in data["sources"]:
            source_name = source_block.get("source", "unknown")
            for q in source_block.get("questions", []):
                if isinstance(q, str):
                    # sometimes questions is a list of strings
                    question = q
                    answer   = None
                else:
                    question = q.get("question", "")
                    answer   = q.get("answer", None)

                if not question.strip():
                    continue

                queries.append({
                    "id":          q.get("id", f"Q{counter:03d}"),
                    "question":    question.strip(),
                    "gold_answer": answer,
                    "source":      source_name,
                })
                counter += 1
        return queries

    # ── Format: {"queries": [...]} ────────────────────────────────────────────
    if isinstance(data, dict) and "queries" in data:
        for q in data["queries"]:
            if isinstance(q, str):
                queries.append({"id": f"Q{counter:03d}", "question": q, "gold_answer": None})
            else:
                queries.append({
                    "id":          q.get("id", f"Q{counter:03d}"),
                    "question":    q.get("question", q.get("query", "")),
                    "gold_answer": q.get("answer", q.get("gold_answer", None)),
                    "source":      q.get("source", "unknown"),
                })
            counter += 1
        return queries

    # ── Format: {"questions": [...]} ─────────────────────────────────────────
    if isinstance(data, dict) and "questions" in data:
        for q in data["questions"]:
            if isinstance(q, str):
                queries.append({"id": f"Q{counter:03d}", "question": q, "gold_answer": None})
            else:
                queries.append({
                    "id":          q.get("id", f"Q{counter:03d}"),
                    "question":    q.get("question", ""),
                    "gold_answer": q.get("answer", None),
                })
            counter += 1
        return queries

    # ── Format: flat list ─────────────────────────────────────────────────────
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                queries.append({"id": f"Q{counter:03d}", "question": item, "gold_answer": None})
            elif isinstance(item, dict):
                queries.append({
                    "id":          item.get("id", f"Q{counter:03d}"),
                    "question":    item.get("question", item.get("query", "")),
                    "gold_answer": item.get("answer", item.get("gold_answer", None)),
                })
            counter += 1
        return queries

    raise ValueError(
        f"Unrecognised input format. Expected dict with 'queries' or 'sources' key, "
        f"or a flat list. Got: {type(data)}"
    )


def load_input(path):
    """Load and parse a JSON input file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Make sure to wrap paths with spaces in quotes:\n"
            f'  python run_inference.py --input "D:\\text mining\\rag project\\file.json"'
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_input(data)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(queries, embedder, retriever, generator,
                 retrieval_k=5, strategy="structured"):
    """
    Runs the full RAG pipeline on a list of parsed queries.
    Returns list of {"id": ..., "answer": ...} dicts.
    """
    outputs = []

    for i, item in enumerate(queries, 1):
        question = item.get("question", "").strip()
        qid      = item.get("id", f"Q{i:03d}")

        if not question:
            print(f"  [{i:03d}/{len(queries)}] SKIP — empty question (id={qid})")
            outputs.append({"id": qid, "answer": ""})
            continue

        print(f"  [{i:03d}/{len(queries)}] {question[:70]}")

        start = time.time()

        # Retrieve relevant chunks
        retrieved = retriever.retrieve(question, initial_k=20, final_k=retrieval_k)

        # Build context string
        context = build_context(retrieved, max_words=600)

        # Generate answer
        answer = generator.generate(
            query=question,
            context=context,
            strategy=strategy,
            max_new_tokens=200,
            temperature=0.3,
        )

        elapsed = time.time() - start
        print(f"           {elapsed:.1f}s  →  {answer[:80]}...")

        outputs.append({
            "id":          qid,
            "question":    question,
            "answer":      answer,
            "source":      item.get("source", ""),
            "gold_answer": item.get("gold_answer"),
        })

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="COMP64702 RAG Culinary Assistant — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --input "D:\\text mining\\rag project\\east_asia_sample_qa.json"
  python run_inference.py --input input_payload.json --output outputs/output_payload.json
  python run_inference.py --input input.json --strategy zero_shot --k 3
        """
    )
    parser.add_argument("--input",  "-i",
        default="data/benchmark/test_set_queries_only.json",
        help="Path to input JSON (wrap in quotes if path has spaces)")
    parser.add_argument("--output", "-o",
        default="outputs/output_payload.json",
        help="Path to write output JSON")
    parser.add_argument("--vector_store",
        default="vector_store",
        help="Path to vector store folder")
    parser.add_argument("--strategy",
        default="structured",
        choices=["zero_shot", "few_shot", "chain_of_thought", "structured"],
        help="Prompting strategy")
    parser.add_argument("--k", type=int, default=5,
        help="Number of chunks to retrieve (max 5 per spec)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  COMP64702 RAG Culinary Assistant — Inference")
    print("="*60)
    print(f"\n  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  Strategy : {args.strategy}")
    print(f"  Chunks k : {args.k}")

    # Auth
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("  HuggingFace: logged in")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load models
    print("\nLoading models...")
    embedder_  = Embedder("BAAI/bge-small-en-v1.5")
    retriever_ = Retriever(args.vector_store, embedder=embedder_)
    generator_ = Generator("Qwen/Qwen2.5-0.5B-Instruct")

    # Load and parse queries
    print(f"\nLoading queries from: {args.input}")
    queries = load_input(args.input)
    print(f"Queries to process  : {len(queries)}")

    # Show detected format info
    sources = list({q.get("source","") for q in queries if q.get("source","")})
    if sources:
        print(f"Sources detected    : {', '.join(sources)}")

    # Run pipeline
    print(f"\nRunning inference...\n")
    start_total = time.time()
    outputs     = run_pipeline(
        queries, embedder_, retriever_, generator_,
        retrieval_k=args.k, strategy=args.strategy,
    )
    total_time  = time.time() - start_total

    # Build output payload — demo day format
    # Only id and answer in the outputs list (as required by spec)
    demo_outputs = [{"id": o["id"], "answer": o["answer"]} for o in outputs]

    output_payload = {
        "generated_at": datetime.now().isoformat(),
        "model":        "Qwen/Qwen2.5-0.5B-Instruct",
        "retrieval":    "hybrid_bm25_dense_rrf_crossencoder_rerank",
        "strategy":     args.strategy,
        "total_queries":len(outputs),
        "outputs":      demo_outputs,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done")
    print(f"  Queries processed : {len(outputs)}")
    print(f"  Total time        : {total_time:.1f}s")
    print(f"  Avg per query     : {total_time/max(len(outputs),1):.1f}s")
    print(f"  Output saved      : {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()