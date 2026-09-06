# openoutnews/ml/embeddings.py
"""Fastembed text embedding utilities — copied from OpenOutFind's
``core/ml/embeddings.py`` with the Django settings dependency dropped."""
from __future__ import annotations

import numpy as np

from openoutnews import conf

_model = None


def _get_model():
    """Lazy-load fastembed model singleton."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        cache_dir = conf.fastembed_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        _model = TextEmbedding(model_name=conf.EMBEDDING_MODEL, cache_dir=str(cache_dir))
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string → 384-dim numpy array."""
    model = _get_model()
    embeddings = list(model.embed([text]))
    return np.array(embeddings[0], dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed multiple texts → (N, 384) numpy array."""
    model = _get_model()
    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)
