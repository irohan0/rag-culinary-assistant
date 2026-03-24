# src/evaluator.py
# =============================================================================
# Extended evaluation module — 12 metrics + confidence scoring
#
# Generation metrics : ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1,
#                      METEOR, Answer F1, Exact Match
# RAG-specific       : Faithfulness, Answer Relevance, Context Precision
# Retrieval metrics  : MRR@K, NDCG@K
# Confidence         : entropy-based token probability confidence
# =============================================================================

import re
import numpy as np
import torch
from collections import defaultdict
from scipy import stats

from rouge_score import rouge_scorer
from bert_score  import score as bert_score_fn
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.translate.meteor_score import meteor_score


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text)


def _normalise_set(text):
    """Return set of normalised tokens."""
    return set(_normalise(text).split())


def _is_relevant(chunk, gold_title):
    """Checks relevance by partial title match."""
    ct = chunk.get("doc_title", "").lower()
    gt = gold_title.lower()
    return gt in ct or ct in gt


# ─────────────────────────────────────────────────────────────────────────────
# Generation metrics
# ─────────────────────────────────────────────────────────────────────────────

def rouge_metrics(predictions, references):
    """ROUGE-1, ROUGE-2, ROUGE-L F1."""
    sc     = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    scores = {"rouge1":[], "rouge2":[], "rougeL":[]}
    for pred, ref in zip(predictions, references):
        r = sc.score(ref, pred)
        scores["rouge1"].append(r["rouge1"].fmeasure)
        scores["rouge2"].append(r["rouge2"].fmeasure)
        scores["rougeL"].append(r["rougeL"].fmeasure)
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    return scores, means


def bertscore_metric(predictions, references,
                     model_type="distilbert-base-uncased"):
    """
    BERTScore F1 — semantic similarity using contextual embeddings.
    More robust than ROUGE for paraphrased answers.
    """
    _, _, F1 = bert_score_fn(
        predictions, references,
        model_type=model_type, lang="en", verbose=False,
    )
    scores = F1.numpy().tolist()
    return scores, float(np.mean(scores))


def meteor_metric(predictions, references):
    """
    METEOR — synonym and stemming-aware overlap.
    Fairer than ROUGE for answers that paraphrase rather than copy.
    """
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            s = meteor_score([ref.lower().split()], pred.lower().split())
        except Exception:
            s = 0.0
        scores.append(float(s))
    return scores, float(np.mean(scores))


def answer_f1_metric(predictions, references):
    """
    Token-level F1 between prediction and reference.
    Standard metric in SQuAD-style QA evaluation.
    """
    def _f1(pred, ref):
        pt = set(pred.lower().split())
        rt = set(ref.lower().split())
        common = pt & rt
        if not common:
            return 0.0
        p = len(common) / len(pt)
        r = len(common) / len(rt)
        return 2 * p * r / (p + r)

    scores = [_f1(p, r) for p, r in zip(predictions, references)]
    return scores, float(np.mean(scores))


def exact_match_metric(predictions, references):
    """Strict exact match after normalisation."""
    scores = [
        1.0 if _normalise(p) == _normalise(r) else 0.0
        for p, r in zip(predictions, references)
    ]
    return scores, float(np.mean(scores))


def answer_relevance_metric(questions, predictions, embedder):
    """
    Cosine similarity between question embedding and answer embedding.
    Measures whether the answer actually addresses the question asked.
    """
    q_embs = embedder.encode_documents(questions, show_progress=False)
    a_embs = embedder.encode_documents(predictions, show_progress=False)
    scores = [
        float(sklearn_cosine(q.reshape(1,-1), a.reshape(1,-1))[0][0])
        for q, a in zip(q_embs, a_embs)
    ]
    return scores, float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# RAG-specific metrics
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness_metric(outputs, threshold=0.25):
    """
    Token-level recall of answer words that appear in retrieved context.
    High faithfulness = answer is grounded in retrieved evidence.
    Low faithfulness  = model is hallucinating.
    """
    scores = []
    for output in outputs:
        pred_words    = _normalise_set(output.get("pred_answer", ""))
        context_words = set()
        for chunk in output.get("retrieved", []):
            context_words.update(_normalise_set(chunk.get("text", "")))
        if not pred_words or not context_words:
            scores.append(0.0)
            continue
        scores.append(len(pred_words & context_words) / len(pred_words))

    faith_rate = sum(1 for s in scores if s >= threshold) / max(len(scores), 1)
    return scores, float(np.mean(scores)), float(faith_rate)


def context_precision_metric(outputs, source_lookup):
    """
    Fraction of retrieved chunks that are relevant to the query.
    High precision = retrieval is focused, not noisy.
    """
    scores = []
    for output in outputs:
        gold      = source_lookup.get(output["question"], "").lower()
        retrieved = output.get("retrieved", [])
        if not gold or not retrieved:
            scores.append(0.0)
            continue
        relevant = sum(1 for c in retrieved if _is_relevant(c, gold))
        scores.append(relevant / len(retrieved))
    return scores, float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_metrics(outputs, source_lookup, k_values=(1, 3, 5)):
    """
    Computes MRR@K, Recall@K, Precision@K, and NDCG@K.

    Args:
        outputs       : list of inference output dicts
        source_lookup : dict mapping question -> gold source title
        k_values      : tuple of K values to evaluate at
    Returns:
        dict of {k: {metric: mean_score}}
    """
    raw = {k: defaultdict(list) for k in k_values}

    for output in outputs:
        gold      = source_lookup.get(output["question"], "")
        retrieved = output.get("retrieved", [])
        if not gold or not retrieved:
            continue

        relevance = [1 if _is_relevant(r, gold) else 0 for r in retrieved]

        for k in k_values:
            top_k = relevance[:k]

            # MRR@K
            mrr = next((1/r for r, rel in enumerate(top_k, 1) if rel), 0.0)
            raw[k]["mrr"].append(mrr)

            # Recall@K
            raw[k]["recall"].append(1.0 if any(top_k) else 0.0)

            # Precision@K
            raw[k]["precision"].append(sum(top_k) / k)

            # NDCG@K
            dcg  = sum(r/np.log2(i+1) for i, r in enumerate(top_k, 1))
            idcg = sum(r/np.log2(i+1)
                       for i, r in enumerate(sorted(top_k, reverse=True), 1))
            raw[k]["ndcg"].append(dcg/idcg if idcg > 0 else 0.0)

    return {
        k: {m: float(np.mean(v)) if v else 0.0 for m, v in raw[k].items()}
        for k in k_values
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence_score(query, context, system_prompt,
                              user_prompt_fn, tokenizer, model,
                              device, max_new_tokens=150):
    """
    Computes a confidence score for a generated answer.

    Method:
        Runs generation with output_scores=True to collect per-token
        probabilities. Computes mean token probability and entropy.
        Confidence = 1 - normalised_entropy, mapped to [0, 1].

    Args:
        query          : question string
        context        : retrieved context string
        system_prompt  : system message for the LLM
        user_prompt_fn : function(query, context) -> user message
        tokenizer      : model tokenizer
        model          : language model
        device         : "cpu" or "cuda"
        max_new_tokens : generation length

    Returns:
        dict with keys: confidence, mean_token_prob, entropy, token_count
        confidence is in [0, 1] — higher is more confident
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt_fn(query, context)},
    ]
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty=1.1,
        )

    token_probs = []
    for step_scores in output.scores:
        probs    = torch.softmax(step_scores[0], dim=-1)
        top_prob = probs.max().item()
        token_probs.append(top_prob)

    if not token_probs:
        return {"confidence": 0.0, "mean_token_prob": 0.0,
                "entropy": 1.0, "token_count": 0}

    token_probs   = np.array(token_probs)
    mean_prob     = float(np.mean(token_probs))
    clipped       = np.clip(token_probs, 1e-10, 1.0)
    entropy       = float(-np.mean(clipped * np.log(clipped)))
    max_entropy   = np.log(60000)
    confidence    = float(max(0.0, min(1.0, 1.0 - entropy / max_entropy)))

    return {
        "confidence":      round(confidence, 4),
        "mean_token_prob": round(mean_prob,  4),
        "entropy":         round(entropy,    4),
        "token_count":     len(token_probs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite evaluation — run all metrics at once
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(outputs, source_lookup, embedder, verbose=True):
    """
    Runs all 12 evaluation metrics on a list of inference outputs.

    Args:
        outputs       : list of output dicts (with pred_answer, gold_answer,
                        question, retrieved fields)
        source_lookup : dict mapping question -> gold source title
        embedder      : Embedder instance for answer relevance
        verbose       : print progress

    Returns:
        means    : dict of metric_name -> mean score
        per_query: dict of metric_name -> list of per-query scores
    """
    predictions = [o["pred_answer"]  for o in outputs]
    references  = [o["gold_answer"]  for o in outputs]
    questions   = [o["question"]     for o in outputs]

    per_query = {}
    means     = {}

    if verbose: print("  ROUGE...")
    r_scores, r_means = rouge_metrics(predictions, references)
    per_query.update(r_scores)
    means.update(r_means)

    if verbose: print("  BERTScore...")
    bs_scores, bs_mean = bertscore_metric(predictions, references)
    per_query["bertscore"] = bs_scores
    means["bertscore"]     = bs_mean

    if verbose: print("  METEOR...")
    mt_scores, mt_mean = meteor_metric(predictions, references)
    per_query["meteor"] = mt_scores
    means["meteor"]     = mt_mean

    if verbose: print("  Answer F1...")
    af_scores, af_mean = answer_f1_metric(predictions, references)
    per_query["answer_f1"] = af_scores
    means["answer_f1"]     = af_mean

    if verbose: print("  Exact Match...")
    em_scores, em_mean = exact_match_metric(predictions, references)
    per_query["exact_match"] = em_scores
    means["exact_match"]     = em_mean

    if verbose: print("  Answer Relevance...")
    ar_scores, ar_mean = answer_relevance_metric(questions, predictions, embedder)
    per_query["answer_relevance"] = ar_scores
    means["answer_relevance"]     = ar_mean

    if verbose: print("  Faithfulness...")
    fa_scores, fa_mean, fa_rate = faithfulness_metric(outputs)
    per_query["faithfulness"]      = fa_scores
    means["faithfulness"]          = fa_mean
    means["faithfulness_rate"]     = fa_rate

    if verbose: print("  Context Precision...")
    cp_scores, cp_mean = context_precision_metric(outputs, source_lookup)
    per_query["context_precision"] = cp_scores
    means["context_precision"]     = cp_mean

    if verbose: print("  Retrieval metrics (MRR, NDCG)...")
    ret = retrieval_metrics(outputs, source_lookup, k_values=(1, 3, 5))
    for k, metric_dict in ret.items():
        for metric, val in metric_dict.items():
            key = f"{metric}@{k}"
            means[key] = val

    return means, per_query


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance
# ─────────────────────────────────────────────────────────────────────────────

def significance_test(system_scores, baseline_scores, metric_name="metric"):
    """Paired t-test: is system significantly better than baseline?"""
    n = min(len(system_scores), len(baseline_scores))
    t, p = stats.ttest_rel(system_scores[:n], baseline_scores[:n])
    return {
        "metric":      metric_name,
        "system_mean": float(np.mean(system_scores[:n])),
        "base_mean":   float(np.mean(baseline_scores[:n])),
        "difference":  float(np.mean(system_scores[:n]) - np.mean(baseline_scores[:n])),
        "t_stat":      float(t),
        "p_value":     float(p),
        "significant": bool(p < 0.05),
    }