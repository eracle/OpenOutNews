# openoutnews/conf.py
"""Env-var configuration — no wizard, no config row, read fresh on every run.

Mirrors OpenOutFind's ``OPENOUTFIND_*`` / state-dir pattern
(``core/conf.py``, ``settings.py``), scoped to ``OPENOUTNEWS_*``.
"""
from __future__ import annotations

import os
from pathlib import Path


def state_dir() -> Path:
    """``~/.openoutnews`` unless overridden — where the store and the embedding
    model cache live."""
    override = os.environ.get("OPENOUTNEWS_HOME")
    return Path(override) if override else Path.home() / ".openoutnews"


def db_path() -> Path:
    """SQLite store path. ``OPENOUTNEWS_DB`` overrides it outright, the way
    ``OPENOUTFIND_DB`` does for the finder."""
    override = os.environ.get("OPENOUTNEWS_DB")
    if override:
        return Path(override)
    return state_dir() / "data" / "db.sqlite3"


def fastembed_cache_dir() -> Path:
    return state_dir() / "fastembed_cache"


def topics() -> list[str]:
    """Search topics fed to the news source — a comma-separated
    ``OPENOUTNEWS_TOPICS``, e.g. ``"open source AI,B2B sales tools"``.

    There is no reader-profile model yet: topics are the only steer on *what
    gets fetched*, and the GP qualifier decides *what gets sent* from the
    engagement labels alone.
    """
    raw = os.environ.get("OPENOUTNEWS_TOPICS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Cosine-similarity threshold above which two candidates are treated as
# syndicated coverage of the same story and collapsed into one representative
# before the qualifier scores anything (see ``openoutnews.dedup``). Sits in
# the ~0.9-0.95 range production news-dedup systems use; tune against real
# `find` output, not in the abstract.
SYNDICATION_SIMILARITY_THRESHOLD = 0.92
