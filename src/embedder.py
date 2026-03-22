# src/embedder.py
# =============================================================================
# Embedding model wrapper.
# Handles loading, encoding, and normalisation for both ingestion and inference.
# Two models compared: MiniLM (baseline) and BGE-small (selected).
# =============================================================================

import numpy as np
from sentence_transformers import SentenceTransformer


# Models compared during development
AVAILABLE_MODELS = {
    "minilm": "all-MiniLM-L6-v2",           # baseline — fast, lightweight
    "bge":    "BAAI/bge-small-en-v1.5",      # selected — retrieval-optimised
}

# BGE models work best with this instruction prefix
BGE_PREFIX = "Represent this sentence: "


class Embedder:
    """
    Wrapper around SentenceTransformer for consistent encoding
    across ingestion and inference.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        """
        Args:
            model_name : HuggingFace model name or local path
        """
        self.model_name = model_name
        self.is_bge     = "bge" in model_name.lower()
        self.model      = SentenceTransformer(model_name)
        self.dim        = self.model.get_sentence_embedding_dimension()
        print(f"Embedder loaded: {model_name} (dim={self.dim})")

    def encode_documents(self, texts, batch_size=64, show_progress=True):
        """
        Encodes a list of document/chunk texts for ingestion.
        BGE prefix is NOT used for documents, only for queries.

        Args:
            texts         : list of strings
            batch_size    : encoding batch size
            show_progress : show tqdm progress bar
        Returns:
            numpy array of shape (len(texts), dim), float32, L2-normalised
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query):
        """
        Encodes a single query string for retrieval.
        Applies BGE instruction prefix if using a BGE model.

        Args:
            query : question string
        Returns:
            numpy array of shape (1, dim), float32, L2-normalised
        """
        text = f"{BGE_PREFIX}{query}" if self.is_bge else query
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)
