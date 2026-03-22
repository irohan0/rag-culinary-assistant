# src/evaluator.py
# =============================================================================
# Evaluation metrics for both retrieval and generation quality.
#
# Retrieval metrics : MRR@K, Recall@K, Precision@K, NDCG@K
# Generation metrics: ROUGE-1/2/L, BERTScore, Exact Match, Faithfulness
# =============================================================================

import re
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from scipy import stats


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def is_relevant(retrieved_chunk, gold_source_title):
    """Checks if a retrieved chunk comes from the gold source document."""
    chunk_title = retrieved_chunk.get("doc_title", "").lower()
    gold_title  = gold_source_title.lower()
    return gold_title in chunk_title or chunk_title in gold_title


def retrieval_metrics(outputs, source_lookup, k_values=[1, 3, 5]):
    """
    Computes MRR, Recall, Precision, and NDCG at multiple K values.

    Args:
        outputs       : list of inference output dicts (with 'retrieved' field)
        source_lookup : dict mapping question -> gold source title
        k_values      : list of K values to evaluate at
    Returns:
        dict of {k: {metric: mean_score}}
    """
    results = {k: {"mrr": [], "recall": [], "precision": [], "ndcg": []}
               for k in k_values}

    for output in outputs:
        question    = output["question"]
        gold_source = source_lookup.get(question, "")
        retrieved   = output.get("retrieved", [])

        if not gold_source or not retrieved:
            continue

        relevance = [1 if is_relevant(r, gold_source) else 0 for r in retrieved]

        for k in k_values:
            top_k = relevance[:k]

            # MRR
            mrr = 0.0
            for rank, rel in enumerate(top_k, 1):
                if rel == 1:
                    mrr = 1.0 / rank
                    break
            results[k]["mrr"].append(mrr)

            # Recall@K
            results[k]["recall"].append(1.0 if any(top_k) else 0.0)

            # Precision@K
            results[k]["precision"].append(sum(top_k) / k)

            # NDCG@K
            dcg  = sum(r / np.log2(i + 1) for i, r in enumerate(top_k, 1))
            idcg = sum(r / np.log2(i + 1) for i, r in enumerate(sorted(top_k, reverse=True), 1))
            results[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)

    return {
        k: {metric: float(np.mean(vals)) if vals else 0.0
            for metric, vals in results[k].items()}
        for k in k_values
    }


# ── Generation metrics ────────────────────────────────────────────────────────

def rouge_metrics(predictions, references):
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Args:
        predictions : list of predicted answer strings
        references  : list of gold answer strings
    Returns:
        dict of {metric: list_of_scores} and {metric: mean_score}
    """
    scorer_ = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores  = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer_.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    means = {k: float(np.mean(v)) for k, v in scores.items()}
    return scores, means


def bertscore_metric(predictions, references, model_type="distilbert-base-uncased"):
    """
    Computes BERTScore F1 — semantic similarity between predictions and references.
    More robust than ROUGE as it captures paraphrases.

    Args:
        predictions : list of predicted answer strings
        references  : list of gold answer strings
        model_type  : BERT variant to use for scoring
    Returns:
        list of per-example F1 scores and mean F1
    """
    _, _, F1 = bert_score_fn(
        predictions, references,
        model_type=model_type,
        lang="en",
        verbose=False,
    )
    f1_list = F1.numpy().tolist()
    return f1_list, float(np.mean(f1_list))


def exact_match_metric(predictions, references):
    """
    Exact match after normalisation (lowercase, strip punctuation).

    Args:
        predictions : list of predicted answer strings
        references  : list of gold answer strings
    Returns:
        list of 0/1 scores and mean score
    """
    def normalise(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    scores = [1 if normalise(p) == normalise(r) else 0
              for p, r in zip(predictions, references)]
    return scores, float(np.mean(scores))


def faithfulness_metric(outputs, threshold=0.25):
    """
    Measures whether predicted answers are grounded in retrieved context.
    Computed as token-level recall: fraction of answer words in the context.

    Args:
        outputs   : list of inference output dicts (with 'retrieved' field)
        threshold : minimum overlap to count as faithful
    Returns:
        list of per-example scores, mean score, faithful rate
    """
    def normalise(text):
        return set(re.sub(r'[^\w\s]', '', text.lower()).split())

    scores = []
    for output in outputs:
        pred_words    = normalise(output.get("pred_answer", ""))
        context_words = set()
        for chunk in output.get("retrieved", []):
            context_words.update(normalise(chunk.get("text", "")))

        if not pred_words or not context_words:
            scores.append(0.0)
            continue

        overlap = pred_words & context_words
        scores.append(len(overlap) / len(pred_words))

    mean_score   = float(np.mean(scores)) if scores else 0.0
    faithful_rate = sum(1 for s in scores if s >= threshold) / len(scores) if scores else 0.0
    return scores, mean_score, faithful_rate


# ── Statistical significance ──────────────────────────────────────────────────

def significance_test(system_scores, baseline_scores, metric_name="metric"):
    """
    Paired t-test to check if system significantly outperforms baseline.

    Args:
        system_scores   : list of per-query scores for our system
        baseline_scores : list of per-query scores for baseline
        metric_name     : label for printing
    Returns:
        dict with t_stat, p_value, significant flag
    """
    n        = min(len(system_scores), len(baseline_scores))
    t_stat, p_value = stats.ttest_rel(system_scores[:n], baseline_scores[:n])
    return {
        "metric":      metric_name,
        "system_mean": float(np.mean(system_scores[:n])),
        "base_mean":   float(np.mean(baseline_scores[:n])),
        "t_stat":      float(t_stat),
        "p_value":     float(p_value),
        "significant": bool(p_value < 0.05),
    }
