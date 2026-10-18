# openoutnews/store.py
"""The local SQLite store — one file, no Django.

``articles`` holds every article this install has ever fetched: its embedding
once computed, the model's verdict once one was asked for (what the qualifier
trains on), the reader's label once the operator has given one (kept apart —
what the reader did is a different question from whether the article fits),
and the issue that sent it, if any (so a re-run doesn't resend the same
story). ``issues`` holds each issue's date window and mode — unfinished while
its verdicts are still being given — and ``issue_fetches`` every day query
that went into it, complete, saturated or failed. ``brief`` is the one stored
reading of the operator's words, and ``anchors`` the invented ideal headlines
that let the qualifier fit before the first real verdict.
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
    absorbed_into TEXT,
    issue_id INTEGER,
    issue_score REAL,
    verdict INTEGER,
    verdict_reason TEXT,
    verdict_issue_id INTEGER,
    repeat_of TEXT
);
CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    mode TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    pending_article_id TEXT
);
CREATE TABLE IF NOT EXISTS brief (
    rules TEXT NOT NULL,
    written_by TEXT NOT NULL,
    written_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS anchors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS issue_fetches (
    issue_id INTEGER NOT NULL REFERENCES issues(id),
    topic TEXT NOT NULL,
    day TEXT NOT NULL,
    status TEXT NOT NULL,
    returned INTEGER,
    error TEXT,
    PRIMARY KEY (issue_id, topic, day)
);
"""

_ADDED_COLUMNS = {
    "absorbed_into": "TEXT",
    "issue_id": "INTEGER",
    "issue_score": "REAL",
    "verdict": "INTEGER",
    "verdict_reason": "TEXT",
    "verdict_issue_id": "INTEGER",
    "repeat_of": "TEXT",
}


def _migrate(conn):
    """Add columns introduced after a store already existed on disk."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
    for name, sql_type in _ADDED_COLUMNS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {name} {sql_type}")


def article_id(url: str) -> str:
    """Stable id for an article, keyed on its URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


@contextmanager
def connect():
    path = conf.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
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
    """Record the reader's label — 1 = read, 0 = skip. Stored only: it trains
    nothing until engagement data says how it should combine with verdicts."""
    conn.execute("UPDATE articles SET label = ? WHERE id = ?", (label, aid))


def set_verdict(conn, aid: str, issue_id: int, verdict: int, reason: str,
                repeat_of: str | None = None):
    """Record the model's verdict — 1 = fits the newsletter, 0 = not — given in
    ``issue_id``; ``repeat_of`` names the accepted article of the same issue a
    fit reports the same story as."""
    conn.execute(
        "UPDATE articles SET verdict = ?, verdict_reason = ?, verdict_issue_id = ?, repeat_of = ? "
        "WHERE id = ?",
        (verdict, reason, issue_id, repeat_of, aid),
    )


def issue_stories(conn, issue_id: int) -> list[sqlite3.Row]:
    """The fits ``issue_id`` accepted that repeat no other — one per story, in
    the order they were judged."""
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM articles WHERE verdict_issue_id = ? AND verdict = 1 AND repeat_of IS NULL "
        "ORDER BY rowid",
        (issue_id,),
    ).fetchall()


def mark_sent(conn, issue_id: int, picks: list[tuple[str, float | None]]):
    """Mark each ``(article_id, score)`` pick as sent in ``issue_id``."""
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "UPDATE articles SET sent_at = ?, issue_id = ?, issue_score = ? WHERE id = ?",
        [(now, issue_id, score, aid) for aid, score in picks],
    )


def recently_sent(conn, since: str) -> list[sqlite3.Row]:
    """Embedded articles sent at or after the ISO instant ``since``."""
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM articles WHERE embedding IS NOT NULL AND sent_at >= ?", (since,)
    ).fetchall()


# ── Issues ──

def last_issue(conn) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM issues ORDER BY id DESC LIMIT 1").fetchone()


def create_issue(conn, *, window_start: str, window_end: str) -> int:
    """Record a new, unfinished issue over ``[window_start, window_end)``; returns its id."""
    cursor = conn.execute(
        "INSERT INTO issues (window_start, window_end, created_at) VALUES (?, ?, ?)",
        (window_start, window_end, datetime.now(timezone.utc).isoformat()),
    )
    return cursor.lastrowid


def finish_issue(conn, issue_id: int, *, mode: str, picks: list[tuple[str, float | None]]):
    """Mark the picks sent and the issue finished, ranked by ``mode``."""
    mark_sent(conn, issue_id, picks)
    conn.execute(
        "UPDATE issues SET mode = ?, finished_at = ?, pending_article_id = NULL WHERE id = ?",
        (mode, datetime.now(timezone.utc).isoformat(), issue_id),
    )


def set_pending(conn, issue_id: int, aid: str | None):
    """Name the one article handed to a calling agent for a verdict, or clear it."""
    conn.execute("UPDATE issues SET pending_article_id = ? WHERE id = ?", (aid, issue_id))


def issue_verdict_count(conn, issue_id: int) -> int:
    """How many verdicts ``issue_id`` has spent of its budget."""
    return conn.execute(
        "SELECT COUNT(*) FROM articles WHERE verdict_issue_id = ?", (issue_id,)
    ).fetchone()[0]


def record_fetch(conn, issue_id: int, *, topic: str, day: str, status: str,
                 returned: int | None, error: str | None):
    """Record one day query of an issue: ``complete``, ``saturated`` or ``failed``."""
    conn.execute(
        "INSERT INTO issue_fetches (issue_id, topic, day, status, returned, error) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (issue_id, topic, day, status, returned, error),
    )


def issue_fetches(conn, issue_id: int) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM issue_fetches WHERE issue_id = ? ORDER BY topic, day", (issue_id,)
    ).fetchall()


def issue_picks(conn, issue_id: int) -> list[sqlite3.Row]:
    """An issue's picks in the order ``find`` ranked them — by score, or
    newest first for an issue ranked before the qualifier could fit."""
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM articles WHERE issue_id = ? "
        "ORDER BY issue_score DESC, published_at DESC",
        (issue_id,),
    ).fetchall()


def verdict_arrays(conn) -> tuple[np.ndarray, np.ndarray]:
    """Every judged article's (embedding, label) — what the qualifier fits on.

    The label is the verdict, except a repeat counts 0: it fits, but another
    verdict near a story already accepted buys nothing, and a 1 there kept the
    GP spending verdicts on copies of one story (13 of 30 in a winery issue
    went to rewrites of the same harvest reports).
    """
    rows = conn.execute(
        "SELECT embedding, CASE WHEN repeat_of IS NULL THEN verdict ELSE 0 END FROM articles "
        "WHERE verdict IS NOT NULL AND embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return np.empty((0, conf.EMBEDDING_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.int64)
    return X, y


def unsent_candidates(conn) -> list[sqlite3.Row]:
    """Fetched, embedded, never sent, not absorbed into another article —
    the rows an issue's pool is drawn from, once bounded to its window.

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
    ``openoutnews.dedup.cluster_representatives`` or ``absorb_into_sent`` —
    the representative may be an article already sent. An absorbed row stays in
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


# ── Brief ──

def get_brief(conn) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM brief").fetchone()


def set_brief(conn, rules: str, written_by: str):
    """Store ``rules`` as this store's brief, replacing any brief already there."""
    conn.execute("DELETE FROM brief")
    conn.execute(
        "INSERT INTO brief (rules, written_by, written_at) VALUES (?, ?, ?)",
        (rules, written_by, datetime.now(timezone.utc).isoformat()),
    )


# ── Anchors ──

def anchor_rows(conn) -> list[sqlite3.Row]:
    """The invented ideal headlines, oldest first."""
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM anchors ORDER BY id").fetchall()


def add_anchor(conn, title: str, embedding: np.ndarray):
    conn.execute(
        "INSERT INTO anchors (title, embedding) VALUES (?, ?)",
        (title, np.asarray(embedding, dtype=np.float32).tobytes()),
    )
