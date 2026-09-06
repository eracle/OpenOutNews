# openoutnews/fetch.py
"""Candidate news fetch — Google News RSS, no API key required.

Deliberately the simplest thing that fetches real articles for a first
version. Swapping in a different provider (NewsAPI, GNews, ...) is a matter
of implementing ``fetch_candidates(topic) -> list[Candidate]`` differently;
nothing downstream cares which source a row came from.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_TIMEOUT_S = 15


@dataclass
class Candidate:
    title: str
    url: str
    published_at: str | None
    summary: str


def fetch_candidates(topic: str) -> list[Candidate]:
    """Every article Google News' RSS returns for one search topic."""
    query = urllib.parse.quote(topic)
    request = urllib.request.Request(
        _RSS_URL.format(query=query),
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
