# src/retriever.py
# =============================================================================
# Retrieval and ranking strategies.
# Three approaches implemented and compared:
#   1. Dense only   — FAISS cosine similarity (baseline)
#   2. BM25 only    — sparse keyword retrieval (baseline)
#   3. Hybrid + Reranking — RRF fusion + cross-encoder (selected)
# =============================================================================

import numpy as np
import faiss
import pickle
from sentence_transformers import CrossEncoder


class Retriever:
    """
    Handles all retrieval strategies for the RAG inference pipeline.
    Loads FAISS index, BM25 index, and chunks from disk.
    """

    def __init__(self, vector_store_path="vector_store", embedder=None):
        """
        Args:
            vector_store_path : path to folder containing index files
            embedder          : Embedder instance for query encoding
        """
        self.embedder = embedder

        # Load FAISS dense index
        self.index = faiss.read_index(f"{vector_store_path}/index.faiss")
        print(f"FAISS index loaded — {self.index.ntotal:,} vectors")

        # Load chunk metadata
        with open(f"{vector_store_path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Chunks loaded     — {len(self.chunks):,} chunks")

        # Load BM25 sparse index
        with open(f"{vector_store_path}/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"BM25 index loaded")

        # Load cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"Reranker loaded")

    # ── Strategy 1: Dense retrieval ──────────────────────────────────────────

    def dense_retrieve(self, query, k=10):
        """
        Pure dense retrieval using FAISS cosine similarity.

        Args:
            query : question string
            k     : number of results to return
        Returns:
            list of dicts with chunk, score, method, index
        """
        q_emb           = self.embedder.encode_query(query)
        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk":  self.chunks[idx],
                "score":  float(score),
                "method": "dense",
                "index":  int(idx),
            })
        return results

    # ── Strategy 2: BM25 sparse retrieval ────────────────────────────────────

    def bm25_retrieve(self, query, k=10):
        """
        BM25 sparse retrieval — keyword-based.
        Complements dense retrieval by catching exact keyword matches.

        Args:
            query : question string
            k     : number of results to return
        Returns:
            list of dicts with chunk, score, method, index
        """
        tokenised = query.lower().split()
        scores    = self.bm25.get_scores(tokenised)
        top_k     = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k:
            if scores[idx] > 0:
                results.append({
                    "chunk":  self.chunks[idx],
                    "score":  float(scores[idx]),
                    "method": "bm25",
                    "index":  int(idx),
                })
        return results

    # ── Strategy 3: Hybrid + Reranking ───────────────────────────────────────

    def reciprocal_rank_fusion(self, dense_results, bm25_results, k=60):
        """
        Combines dense and BM25 ranked lists using Reciprocal Rank Fusion.
        RRF score = sum of 1/(k + rank) across all lists.
        k=60 is the standard smoothing constant.

        Args:
            dense_results : list from dense_retrieve
            bm25_results  : list from bm25_retrieve
            k             : RRF smoothing constant
        Returns:
            merged and re-ranked list of results
        """
        scores = {}

        for rank, result in enumerate(dense_results, 1):
            idx = result["index"]
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        for rank, result in enumerate(bm25_results, 1):
            idx = result["index"]
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {"chunk": self.chunks[idx], "score": rrf_score,
             "method": "hybrid_rrf", "index": idx}
            for idx, rrf_score in sorted_indices
        ]

    def rerank(self, query, candidates, top_n=5):
        """
        Cross-encoder reranking: scores (query, chunk) pairs jointly.
        More accurate than bi-encoder similarity but slower — applied
        only to top candidates from RRF.

        Args:
            query      : question string
            candidates : list of candidate results from RRF
            top_n      : number of final results to return (max 5 per spec)
        Returns:
            top_n results sorted by reranker score
        """
        if not candidates:
            return []

        pairs  = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["reranker_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
        return reranked[:top_n]

    def retrieve(self, query, initial_k=20, final_k=5):
        """
        Full hybrid retrieval pipeline (selected strategy):
          1. Dense retrieval (top initial_k)
          2. BM25 retrieval  (top initial_k)
          3. RRF fusion
          4. Cross-encoder reranking (top final_k)

        Args:
            query     : question string
            initial_k : candidates per retrieval method
            final_k   : final chunks returned (max 5 per spec)
        Returns:
            list of up to final_k ranked chunk results
        """
        dense_results = self.dense_retrieve(query, k=initial_k)
        bm25_results  = self.bm25_retrieve(query,  k=initial_k)
        fused         = self.reciprocal_rank_fusion(dense_results, bm25_results)
        reranked      = self.rerank(query, fused[:initial_k], top_n=final_k)
        return reranked
