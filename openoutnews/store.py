# openoutnews/store.py
"""The local SQLite store — one table, one file, no Django.

Holds every article this install has ever fetched: its embedding once
computed, its engagement label once the operator has provided one, and
whether it has already been handed out by ``find`` (so a re-run doesn't
resend the same story).
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from openoutnews import conf

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TEXT,
    summary TEXT,
    embedding BLOB,
    label INTEGER,
    fetched_at TEXT NOT NULL,
    sent_at TEXT,
    absorbed_into TEXT
);
"""


def _migrate(conn):
    """Add columns introduced after a store already existed on disk."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
    if "absorbed_into" not in columns:
        conn.execute("ALTER TABLE articles ADD COLUMN absorbed_into TEXT")


def article_id(url: str) -> str:
    """Stable id for an article, keyed on its URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


@contextmanager
def connect():
    path = conf.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_candidate(conn, *, title: str, url: str, published_at: str | None, summary: str):
    """Insert a freshly-fetched candidate if its URL isn't already known.

    Never overwrites an existing row — a re-fetch must not clobber a label or
    a ``sent_at`` this article has already earned.
    """
    conn.execute(
        "INSERT OR IGNORE INTO articles (id, title, url, published_at, summary, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (article_id(url), title, url, published_at, summary,
         datetime.now(timezone.utc).isoformat()),
    )


def set_embedding(conn, aid: str, embedding: np.ndarray):
    conn.execute(
        "UPDATE articles SET embedding = ? WHERE id = ?",
        (np.asarray(embedding, dtype=np.float32).tobytes(), aid),
    )


def unembedded_ids(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT id FROM articles WHERE embedding IS NULL")]


def get_row(conn, aid: str) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM articles WHERE id = ?", (aid,)).fetchone()


def label_article(conn, aid: str, label: int):
    """Record an engagement label — 1 = read, 0 = skip."""
    conn.execute("UPDATE articles SET label = ? WHERE id = ?", (label, aid))


def mark_sent(conn, ids: list[str]):
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "UPDATE articles SET sent_at = ? WHERE id = ?",
        [(now, aid) for aid in ids],
    )


def labeled_arrays(conn) -> tuple[np.ndarray, np.ndarray]:
    """Every labelled article's (embedding, label) — what the qualifier fits on."""
    rows = conn.execute(
        "SELECT embedding, label FROM articles WHERE label IS NOT NULL AND embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return np.empty((0, conf.EMBEDDING_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.int64)
    return X, y


def unsent_candidates(conn) -> list[sqlite3.Row]:
    """Fetched, embedded, never sent, not absorbed into another candidate's
    cluster — the pool ``find`` picks from.

    A candidate stays eligible whether or not it has since been labelled: a
    label comes from a *sent* article the reader reacted to, so an unsent row
    is always unlabelled in practice, but the query doesn't assume it.
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM articles WHERE embedding IS NOT NULL AND sent_at IS NULL "
        "AND absorbed_into IS NULL ORDER BY published_at DESC"
    ).fetchall()


def absorb_candidates(conn, clusters: dict[str, list[str]]):
    """Mark every non-representative member of a syndication cluster.

    ``clusters`` is ``{representative_id: [absorbed_id, ...]}`` from
    ``openoutnews.dedup.cluster_representatives``. An absorbed row stays in
    the store (so a re-fetch of the same story isn't re-clustered for no
    benefit) but drops out of ``unsent_candidates`` for good — the
    representative is what the qualifier scores and ``label`` accepts.
    """
    pairs = [
        (representative, absorbed_id)
        for representative, absorbed_ids in clusters.items()
        for absorbed_id in absorbed_ids
    ]
    conn.executemany(
        "UPDATE articles SET absorbed_into = ? WHERE id = ?", pairs
    )
