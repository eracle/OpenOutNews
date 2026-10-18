# openoutnews/fetch.py
"""Candidate news fetch — Google News RSS, no API key required.

One query returns at most ``RSS_CAP`` (100) items and the feed has no paging:
``&start=``, ``&num=`` and ``&page=`` all return the same 100. What does reach
further is a date window inside the query, ``after:YYYY-MM-DD
before:YYYY-MM-DD``, which is its own 100 — so ``fetch_day`` asks for one day
at a time.

Boundary semantics, measured against live results (2026-10-18, topics ``open
source AI`` and ``local LLMs``, days 9–12 Oct):

- ``after:D before:D+1`` returns the **Pacific-time** day D: every item's
  ``pubDate`` lay between D 07:00 UTC and D+1 07:00 UTC (midnight to midnight
  PDT). Both ends are inclusive — an item stamped exactly midnight PT can come
  back for both neighbouring days.
- Consecutive day windows overlap slightly and leave no gap: 0–4 of ~60 links
  were shared between neighbouring days, the shared ones stamped on the
  boundary or, once, an item whose Google index date differs from its
  ``pubDate``.

So consecutive days are fired back to back, the overlap is absorbed by the
store's URL key, and which issue an article belongs to is decided by its own
``pubDate`` read as a Pacific date (``google_day``), not by which query
returned it. The Italian edition reads the dates in Pacific time too (all 177
items for ``vino``/``cantine``/``vendemmia`` on 14 Oct stamped between 07:00
UTC that day and the next), so the edition changes the feed, not the days.

The edition (``conf.edition()``) is three URL parameters, and ``hl`` is not
derivable from the other two in one way: an English edition takes a regional
``hl`` (``en-US``, ``en-GB``) and the others the bare language (``it``,
``fr``) — each returns an empty body the other way round.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from openoutnews import conf

_RSS_URL = ("https://news.google.com/rss/search?q={query}"
            "&hl={hl}&gl={country}&ceid={country}:{language}")
_TIMEOUT_S = 15

# Items one Google News RSS query returns at most. A day that returns this
# many is saturated: more was published than the feed would show.
RSS_CAP = 100

# The timezone Google News' ``after:``/``before:`` dates are read in.
GOOGLE_TZ = ZoneInfo("America/Los_Angeles")


@dataclass
class Candidate:
    title: str
    url: str
    published_at: str | None
    summary: str


def feed_url(query: str) -> str:
    """The RSS search URL for ``query`` in the configured edition."""
    country, language = conf.edition()
    hl = f"{language}-{country}" if language == "en" else language
    return _RSS_URL.format(query=urllib.parse.quote(query), hl=hl,
                           country=country, language=language)


def fetch_candidates(query: str) -> list[Candidate]:
    """Every article Google News' RSS returns for one search query."""
    request = urllib.request.Request(
        feed_url(query),
        headers={"User-Agent": "OpenOutNews/0.1 (+https://openoutreach.app)"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        body = response.read()

    root = ElementTree.fromstring(body)
    candidates = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        candidates.append(Candidate(
            title=title,
            url=link,
            published_at=_to_iso(item.findtext("pubDate")),
            summary=(item.findtext("description") or "").strip(),
        ))
    return candidates


def day_query(topic: str, day: date) -> str:
    """The query for one topic's articles published on one Pacific-time day."""
    return f"{topic} after:{day.isoformat()} before:{(day + timedelta(days=1)).isoformat()}"


def fetch_day(topic: str, day: date) -> list[Candidate]:
    """Every article Google News returns for ``topic`` on ``day``."""
    return fetch_candidates(day_query(topic, day))


def google_today() -> date:
    """Today as Google News' date operators read it — every earlier day is complete."""
    return datetime.now(GOOGLE_TZ).date()


def google_day(published_at: str | None) -> date | None:
    """The Pacific-time date of an ISO ``published_at``, or ``None`` if unknown."""
    if not published_at:
        return None
    moment = datetime.fromisoformat(published_at)
    if moment.tzinfo is None:  # a ``-0000`` pubDate parses naive; it means UTC
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(GOOGLE_TZ).date()


def _to_iso(pub_date: str | None) -> str | None:
    """RFC-2822 ``pubDate`` -> ISO-8601, so ``ORDER BY published_at`` is chronological.

    The raw string sorts by weekday name first (``"Wed, 26 Aug..."``), which is
    not a date order at all — this is what the store persists and sorts on.
    """
    if not pub_date:
        return None
    try:
        return parsedate_to_datetime(pub_date.strip()).isoformat()
    except (TypeError, ValueError):
        return None
