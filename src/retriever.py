# =============================================================================
# Retrieval and ranking module — four strategies:
#   1. Dense only          — FAISS cosine similarity (baseline)
#   2. BM25 only           — sparse keyword retrieval (baseline)
#   3. Hybrid + Reranking  — RRF + cross-encoder (selected)
#   4. Query expansion     — multi-query RRF + reranking (improved)
#
# Also includes MMR (Maximal Marginal Relevance) as an alternative reranker
# =============================================================================
 
import numpy as np
import faiss
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sentence_transformers import CrossEncoder
 
 
class Retriever:
    """
    Full retrieval pipeline with dense, sparse, hybrid,
    query expansion, and MMR support.
    """
 
    def __init__(self, vector_store_path="vector_store", embedder=None):
        self.embedder = embedder
 
        # FAISS dense index
        self.index = faiss.read_index(f"{vector_store_path}/index.faiss")
        print(f"FAISS index loaded — {self.index.ntotal:,} vectors")
 
        # Chunk metadata
        with open(f"{vector_store_path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Chunks loaded     — {len(self.chunks):,} chunks")
 
        # BM25 sparse index
        with open(f"{vector_store_path}/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"BM25 index loaded")
 
        # Cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"Reranker loaded")
 
    # ── Strategy 1: Dense retrieval ──────────────────────────────────────────
 
    def dense_retrieve(self, query, k=10):
        """Pure dense retrieval using FAISS cosine similarity."""
        q_emb           = self.embedder.encode_query(query)
        scores, indices = self.index.search(q_emb, k)
        return [
            {"chunk": self.chunks[idx], "score": float(s),
             "method": "dense", "index": int(idx)}
            for s, idx in zip(scores[0], indices[0]) if idx != -1
        ]
 
    # ── Strategy 2: BM25 retrieval ───────────────────────────────────────────
 
    def bm25_retrieve(self, query, k=10):
        """BM25 sparse keyword retrieval."""
        scores  = self.bm25.get_scores(query.lower().split())
        top_k   = np.argsort(scores)[::-1][:k]
        return [
            {"chunk": self.chunks[idx], "score": float(scores[idx]),
             "method": "bm25", "index": int(idx)}
            for idx in top_k if scores[idx] > 0
        ]
 
    # ── RRF fusion ───────────────────────────────────────────────────────────
 
    def reciprocal_rank_fusion(self, *ranked_lists, k=60):
        """
        Combines any number of ranked lists using Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) across all lists.
        k=60 is the standard smoothing constant.
        """
        scores = {}
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, 1):
                idx = result["index"]
                scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
 
        sorted_idx = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"chunk": self.chunks[idx], "score": rrf_score,
             "method": "hybrid_rrf", "index": idx}
            for idx, rrf_score in sorted_idx
        ]
 
    # ── Cross-encoder reranker ────────────────────────────────────────────────
 
    def rerank(self, query, candidates, top_n=5):
        """
        Cross-encoder reranking: scores (query, chunk) pairs jointly.
        Applied only to top candidates from RRF to keep latency acceptable.
        """
        if not candidates:
            return []
        pairs  = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["reranker_score"] = float(s)
        return sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)[:top_n]
 
    # ── MMR reranker ──────────────────────────────────────────────────────────
 
    def mmr_rerank(self, query, candidates, lambda_=0.5, k=5):
        """
        Maximal Marginal Relevance reranking.
        Balances relevance to query with diversity among selected chunks.
        Prevents returning multiple chunks that say the same thing.
 
        Args:
            query      : question string
            candidates : list of candidate results
            lambda_    : trade-off (1=pure relevance, 0=pure diversity)
            k          : number of chunks to return
 
        Returns:
            k chunks selected by MMR
        """
        if not candidates:
            return []
 
        texts      = [c["chunk"]["text"] for c in candidates]
        chunk_embs = self.embedder.encode_documents(texts, show_progress=False)
        q_emb      = self.embedder.encode_query(query)
 
        selected  = []
        remaining = list(range(len(candidates)))
 
        while len(selected) < k and remaining:
            rem_embs = chunk_embs[remaining]
 
            # Relevance to query
            rel_sims = sklearn_cosine(q_emb, rem_embs)[0]
 
            if not selected:
                best_local = np.argmax(rel_sims)
            else:
                # Redundancy with already selected
                sel_embs  = chunk_embs[selected]
                red_sims  = sklearn_cosine(rem_embs, sel_embs).max(axis=1)
                mmr_scores = lambda_ * rel_sims - (1 - lambda_) * red_sims
                best_local = np.argmax(mmr_scores)
 
            best_global = remaining[best_local]
            selected.append(best_global)
            remaining.remove(best_global)
 
        return [candidates[i] for i in selected]
 
    # ── Strategy 3: Hybrid + Reranking (selected) ────────────────────────────
 
    def retrieve(self, query, initial_k=20, final_k=5, use_mmr=False):
        """
        Full hybrid pipeline (selected strategy):
          1. Dense retrieval  (top initial_k)
          2. BM25 retrieval   (top initial_k)
          3. RRF fusion
          4. Cross-encoder or MMR reranking (top final_k)
 
        Args:
            query     : question string
            initial_k : candidates per retrieval method
            final_k   : final chunks returned (max 5 per spec)
            use_mmr   : use MMR instead of cross-encoder reranking
        """
        dense_r = self.dense_retrieve(query, k=initial_k)
        bm25_r  = self.bm25_retrieve(query,  k=initial_k)
        fused   = self.reciprocal_rank_fusion(dense_r, bm25_r)
 
        if use_mmr:
            return self.mmr_rerank(query, fused[:initial_k],
                                   lambda_=0.5, k=final_k)
        else:
            return self.rerank(query, fused[:initial_k], top_n=final_k)
 
    # ── Strategy 4: Query expansion ──────────────────────────────────────────
 
    def retrieve_with_expansion(self, query, generator,
                                initial_k=15, final_k=5):
        """
        Query expansion: generates alternative phrasings of the query,
        retrieves for each, fuses all lists via RRF, then reranks.
 
        Improves recall for queries where the user's phrasing doesn't
        directly match the vocabulary used in the corpus.
 
        Args:
            query     : original question string
            generator : Generator instance (used to expand query)
            initial_k : candidates per query variant
            final_k   : final chunks returned
        """
        # Generate query variants
        from src.generator import STRATEGIES
        strat      = STRATEGIES["zero_shot"]
        sys_prompt = strat["system"]
        user_msg   = (
            f"Write 2 different ways to ask this question about East Asian cuisine. "
            f"Output ONLY the questions, one per line, no numbering or explanation.\n\n"
            f"Original question: {query}"
        )
 
        import torch
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg},
        ]
        text   = generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(generator.model.parameters()).device
        inputs = generator.tokenizer([text], return_tensors="pt").to(device)
 
        with torch.no_grad():
            out = generator.model.generate(
                **inputs, max_new_tokens=80, temperature=0.7,
                do_sample=True, pad_token_id=generator.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        expanded  = generator.tokenizer.decode(generated, skip_special_tokens=True)
 
        # Parse variants — take lines that look like questions
        variants = [query]
        for line in expanded.split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if len(line) > 10 and len(line) < 200:
                variants.append(line)
        variants = variants[:3]  # original + at most 2 variants
 
        # Retrieve for each variant
        all_ranked_lists = []
        for variant in variants:
            dense_r = self.dense_retrieve(variant, k=initial_k)
            bm25_r  = self.bm25_retrieve(variant,  k=initial_k)
            all_ranked_lists.extend([dense_r, bm25_r])
 
        # Fuse all lists and rerank
        fused    = self.reciprocal_rank_fusion(*all_ranked_lists)
        reranked = self.rerank(query, fused[:initial_k*2], top_n=final_k)
 
        return reranked, variants
 