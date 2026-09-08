# openoutnews/ml/embeddings.py
"""Fastembed text embedding utilities — thin wrapper over ``openoutlearn.embeddings``,
supplying this install's model name and cache dir."""
from __future__ import annotations

import numpy as np

from openoutlearn import embeddings as _shared

from openoutnews import conf


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string → 384-dim numpy array."""
    return _shared.embed_text(text, model_name=conf.EMBEDDING_MODEL, cache_dir=conf.fastembed_cache_dir())


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed multiple texts → (N, 384) numpy array."""
    return _shared.embed_texts(texts, model_name=conf.EMBEDDING_MODEL, cache_dir=conf.fastembed_cache_dir())
