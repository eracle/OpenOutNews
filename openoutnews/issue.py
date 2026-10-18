# openoutnews/issue.py
"""One issue of the newsletter: its window, its fetch, its pool.

An issue covers ``[window_start, window_end)`` in Google News' Pacific-time
days: from where the previous issue ended (or ``conf.issue_days()`` back, for
the first) up to today, which is excluded so every day fetched is complete.
Each topic is fetched one day at a time (see ``openoutnews.fetch``), and the
pool is the unsent articles published inside the window, deduplicated against
what was already sent and then against each other.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from http.client import HTTPException
from xml.etree import ElementTree

from openoutnews import conf, dedup, fetch, store
from openoutnews.ml import embeddings


@dataclass
class DayFetch:
    """The outcome of one topic's query for one day."""

    topic: str
    day: date
    status: str  # complete | saturated | failed
    returned: int | None = None
    error: str | None = None


def next_window(conn, today: date) -> tuple[date, date]:
    """The next issue's ``(start, end)``: from the last issue's end to ``today``."""
    last = store.last_issue(conn)
    if last is None:
        return today - timedelta(days=conf.issue_days()), today
    return date.fromisoformat(last["window_end"]), today


# ── Fetch ──

def fetch_window(conn, topics: list[str], start: date, end: date) -> list[DayFetch]:
    """Fetch every topic one day at a time over ``[start, end)`` into the store.

    A failed day is logged and returned as a gap; the rest still run.
    """
    days = [start + timedelta(days=i) for i in range((end - start).days)]
    results = [_fetch_day(conn, topic, day) for topic in topics for day in days]
    conn.commit()
    return results


def _fetch_day(conn, topic: str, day: date) -> DayFetch:
    try:
        candidates = fetch.fetch_day(topic, day)
    except (OSError, HTTPException, ElementTree.ParseError) as exc:
        print(f"fetch failed: {topic!r} on {day}: {exc}", file=sys.stderr)
        return DayFetch(topic, day, "failed", error=str(exc))

    for candidate in candidates:
        store.upsert_candidate(
            conn,
            title=candidate.title,
            url=candidate.url,
            published_at=candidate.published_at,
            summary=candidate.summary,
        )
    if len(candidates) >= fetch.RSS_CAP:
        print(f"saturated: {topic!r} on {day} returned {len(candidates)}, the feed's cap",
              file=sys.stderr)
        return DayFetch(topic, day, "saturated", returned=len(candidates))
    return DayFetch(topic, day, "complete", returned=len(candidates))


def embed_pending(conn) -> None:
    """Embed every stored article that has no vector yet, from its title.

    The title alone: a search item's summary repeats the title's headline and
    outlet inside a tracking link, whose random characters only pulled
    unrelated articles closer together (median pair similarity 0.62 with it,
    0.55 without, on 263 live articles).
    """
    pending = store.unembedded_ids(conn)
    if not pending:
        return
    rows = [store.get_row(conn, aid) for aid in pending]
    vectors = embeddings.embed_texts([r["title"] for r in rows])
    for aid, vector in zip(pending, vectors):
        store.set_embedding(conn, aid, vector)


# ── Pool ──

def pool(conn, start: date, end: date) -> list:
    """The unsent articles published in ``[start, end)`` that are left once
    repeats of recently sent articles and syndicated copies are absorbed.

    Newest first, as ``store.unsent_candidates`` orders them.
    """
    in_window = [row for row in store.unsent_candidates(conn) if _published_in(row, start, end)]

    since = datetime.now(timezone.utc) - timedelta(days=conf.dedup_lookback_days())
    repeats = dedup.absorb_into_sent(in_window, store.recently_sent(conn, since.isoformat()))
    store.absorb_candidates(conn, repeats)
    repeated = {aid for ids in repeats.values() for aid in ids}
    fresh = [row for row in in_window if row["id"] not in repeated]

    clusters = dedup.cluster_representatives(fresh)
    store.absorb_candidates(conn, clusters)
    syndicated = {aid for ids in clusters.values() for aid in ids}
    return [row for row in fresh if row["id"] not in syndicated]


def _published_in(row, start: date, end: date) -> bool:
    day = fetch.google_day(row["published_at"])
    return day is not None and start <= day < end
