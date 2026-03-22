# src/chunker.py
# =============================================================================
# Chunking strategies for the RAG ingestion pipeline.
# Three approaches implemented and compared:
#   1. Fixed-size  — baseline, splits by word count
#   2. Sentence    — respects sentence boundaries
#   3. Semantic    — splits on topic shifts (selected for final system)
# =============================================================================

import re
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    """Remove citation markers, collapse whitespace."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[note\s?\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text)
    text = re.sub(r'\[edit\]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fixed_size_chunker(text, chunk_size=256, overlap=32):
    """
    Splits text into fixed-size chunks by word count with sliding overlap.

    Args:
        text       : document text
        chunk_size : target words per chunk
        overlap    : words to overlap between chunks
    Returns:
        list of chunk strings
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def sentence_chunker(text, target_words=200, overlap_sentences=1):
    """
    Groups complete sentences into chunks targeting a word count.
    Preserves sentence boundaries unlike fixed-size chunking.

    Args:
        text              : document text
        target_words      : target words per chunk
        overlap_sentences : sentences carried over to next chunk
    Returns:
        list of chunk strings
    """
    sentences = sent_tokenize(text)
    chunks    = []
    current   = []
    count     = 0

    for sent in sentences:
        words  = sent.split()
        count += len(words)
        current.append(sent)
        if count >= target_words:
            chunk_text = " ".join(current)
            if len(chunk_text.strip()) > 20:
                chunks.append(chunk_text)
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            count   = sum(len(s.split()) for s in current)

    if current:
        chunk_text = " ".join(current)
        if len(chunk_text.strip()) > 20:
            chunks.append(chunk_text)

    return chunks


def semantic_chunker(text, embed_model, threshold=0.45,
                     min_chunk_words=50, max_chunk_words=400):
    """
    Splits text at points where cosine similarity between adjacent
    sentence embeddings drops below a threshold (topic shift).

    This is the state-of-the-art approach — each chunk is semantically
    self-contained, improving retrieval precision.

    Args:
        text            : document text
        embed_model     : SentenceTransformer model for sentence embeddings
        threshold       : similarity below which a split is made
        min_chunk_words : minimum words before a split is allowed
        max_chunk_words : force split if chunk exceeds this size
    Returns:
        list of chunk strings
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [text] if text.strip() else []

    embeddings   = embed_model.encode(sentences, batch_size=32, show_progress_bar=False)
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i+1].reshape(1, -1)
        )[0][0]
        similarities.append(float(sim))

    chunks        = []
    current_sents = [sentences[0]]
    current_words = len(sentences[0].split())

    for i, sim in enumerate(similarities):
        next_sent   = sentences[i + 1]
        next_words  = len(next_sent.split())
        should_split = (
            sim < threshold and current_words >= min_chunk_words
        ) or (
            current_words + next_words > max_chunk_words
        )
        if should_split:
            chunks.append(" ".join(current_sents))
            current_sents = [next_sent]
            current_words = next_words
        else:
            current_sents.append(next_sent)
            current_words += next_words

    if current_sents:
        chunks.append(" ".join(current_sents))

    return [c for c in chunks if len(c.strip()) > 20]


def chunk_documents(corpus, strategy="semantic", embed_model=None, **kwargs):
    """
    Chunks all documents in the corpus using the specified strategy.

    Args:
        corpus      : list of document dicts (from load_corpus)
        strategy    : "fixed", "sentence", or "semantic"
        embed_model : required if strategy="semantic"
        **kwargs    : passed to the chunker function
    Returns:
        list of chunk dicts with text and metadata
    """
    chunks = []

    for doc in corpus:
        text   = clean_text(doc.get("text", ""))
        title  = doc.get("title", "Unknown")
        url    = doc.get("url", "")
        source = doc.get("source_type", "unknown")

        if len(text.split()) < 50:
            continue

        if strategy == "fixed":
            doc_chunks = fixed_size_chunker(text, **kwargs)
        elif strategy == "sentence":
            doc_chunks = sentence_chunker(text, **kwargs)
        elif strategy == "semantic":
            if embed_model is None:
                raise ValueError("embed_model required for semantic chunking")
            doc_chunks = semantic_chunker(text, embed_model, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "chunk_id":     f"{source}_{title[:30]}_{i}",
                "text":         chunk_text,
                "doc_title":    title,
                "doc_url":      url,
                "source_type":  source,
                "chunk_index":  i,
                "total_chunks": len(doc_chunks),
                "strategy":     strategy,
                "word_count":   len(chunk_text.split()),
            })

    return chunks
