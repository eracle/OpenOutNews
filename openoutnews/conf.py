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

    Topics steer *what gets fetched*; the brief (``brief_text``) steers *what
    gets sent*.
    """
    raw = os.environ.get("OPENOUTNEWS_TOPICS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def brief_text() -> str:
    """Who reads the newsletter and what it covers, in the operator's words —
    ``OPENOUTNEWS_BRIEF``. The stored brief is written from it once
    (``openoutnews.brief``), and every verdict reads both."""
    return os.environ.get("OPENOUTNEWS_BRIEF", "").strip()


def ai_model() -> str:
    """A pydantic-ai ``provider:model`` id — ``OPENOUTNEWS_AI_MODEL``, e.g.
    ``anthropic:claude-haiku-5-5``."""
    return os.environ.get("OPENOUTNEWS_AI_MODEL", "").strip()


def llm_api_key() -> str:
    return os.environ.get("OPENOUTNEWS_LLM_API_KEY", "").strip()


def verdicts_per_issue() -> int:
    """How many articles an issue may put to the model —
    ``OPENOUTNEWS_VERDICTS_PER_ISSUE``, default 150."""
    return int(os.environ.get("OPENOUTNEWS_VERDICTS_PER_ISSUE", "150"))


def issue_days() -> int:
    """How many days back the first issue's window reaches —
    ``OPENOUTNEWS_ISSUE_DAYS``, default 7. Every later issue starts where the
    previous one ended."""
    return int(os.environ.get("OPENOUTNEWS_ISSUE_DAYS", "7"))


def dedup_lookback_days() -> int:
    """How far back a candidate is compared against articles already sent —
    ``OPENOUTNEWS_DEDUP_LOOKBACK_DAYS``, default 28."""
    return int(os.environ.get("OPENOUTNEWS_DEDUP_LOOKBACK_DAYS", "28"))


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

PROMPTS_DIR = Path(__file__).parent / "prompts"

# How many invented ideal headlines anchor a cold start. Several rather than
# one so the fit region is outlined rather than pinned to one guess, few
# enough that a handful of real fits outweighs them — OpenOutFind's number.
ANCHOR_COUNT = 3

# Cosine-similarity threshold above which two candidates are treated as
# syndicated coverage of the same story and collapsed into one representative
# before the qualifier scores anything (see ``openoutnews.dedup``). Sits in
# the ~0.9-0.95 range production news-dedup systems use. Measured on
# embedded titles (263 live articles, two topics, three days): every pair at
# or above it was one story.
SYNDICATION_SIMILARITY_THRESHOLD = 0.90
