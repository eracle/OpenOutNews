# tests/test_fetch.py
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from openoutnews import fetch

_RSS = """<?xml version="1.0"?>
<rss><channel>
<item>
  <title>First story</title>
  <link>https://example.com/first</link>
  <pubDate>Wed, 26 Aug 2026 10:00:00 GMT</pubDate>
  <description>Summary one</description>
</item>
<item>
  <title>No link, skipped</title>
  <link></link>
  <pubDate>Wed, 26 Aug 2026 11:00:00 GMT</pubDate>
  <description>Summary two</description>
</item>
</channel></rss>
"""


def _urlopen_returning(body: bytes):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = body
    return response


def test_fetch_candidates_parses_titled_linked_items():
    with patch("openoutnews.fetch.urllib.request.urlopen", return_value=_urlopen_returning(_RSS.encode())):
        candidates = fetch.fetch_candidates("open source AI", ("US", "en"))

    assert len(candidates) == 1
    assert candidates[0].title == "First story"
    assert candidates[0].url == "https://example.com/first"
    assert candidates[0].published_at == "2026-08-26T10:00:00+00:00"
    assert candidates[0].summary == "Summary one"


def test_fetch_candidates_skips_items_missing_a_link_or_title():
    body = b"<rss><channel><item><title></title><link>https://example.com/x</link></item></channel></rss>"
    with patch("openoutnews.fetch.urllib.request.urlopen", return_value=_urlopen_returning(body)):
        candidates = fetch.fetch_candidates("topic", ("US", "en"))

    assert candidates == []


def test_day_query_bounds_one_day_with_after_and_before():
    assert fetch.day_query("open source AI", date(2026, 10, 31)) == \
        "open source AI after:2026-10-31 before:2026-11-01"


def test_fetch_day_sends_the_dated_query():
    with patch("openoutnews.fetch.urllib.request.urlopen",
               return_value=_urlopen_returning(_RSS.encode())) as urlopen:
        fetch.fetch_day("llama.cpp", date(2026, 10, 10))

    url = urlopen.call_args.args[0].full_url
    assert "q=llama.cpp%20after%3A2026-10-10%20before%3A2026-10-11" in url


def test_feed_url_takes_a_regional_hl_for_english():
    assert fetch.feed_url("wine", ("US", "en")).endswith("?q=wine&hl=en-US&gl=US&ceid=US:en")


def test_feed_url_takes_the_bare_language_outside_english():
    assert fetch.feed_url("vino", ("IT", "it")).endswith("?q=vino&hl=it&gl=IT&ceid=IT:it")


def _fetched_url(topic: str) -> str:
    with patch("openoutnews.fetch.urllib.request.urlopen",
               return_value=_urlopen_returning(_RSS.encode())) as urlopen:
        fetch.fetch_day(topic, date(2026, 10, 10))
    return urlopen.call_args.args[0].full_url


def test_fetch_day_searches_the_default_edition(monkeypatch):
    monkeypatch.delenv("OPENOUTNEWS_EDITION", raising=False)
    assert _fetched_url("wine").endswith("&hl=en-US&gl=US&ceid=US:en")
    monkeypatch.setenv("OPENOUTNEWS_EDITION", "IT:it")
    assert _fetched_url("vino").endswith("&hl=it&gl=IT&ceid=IT:it")


def test_a_topic_searches_its_own_edition(monkeypatch):
    monkeypatch.setenv("OPENOUTNEWS_EDITION", "IT:it")

    url = _fetched_url("Italian wine@GB:en")

    assert "q=Italian%20wine%20after%3A2026-10-10" in url
    assert url.endswith("&hl=en-GB&gl=GB&ceid=GB:en")


def test_a_malformed_edition_is_refused(monkeypatch):
    monkeypatch.setenv("OPENOUTNEWS_EDITION", "italy")
    with pytest.raises(ValueError, match="COUNTRY:language"):
        fetch.fetch_day("vino", date(2026, 10, 10))
    with pytest.raises(ValueError, match="topic 'vino@italy'"):
        fetch.fetch_day("vino@italy", date(2026, 10, 10))


def test_google_day_reads_published_at_as_a_pacific_date():
    assert fetch.google_day("2026-10-11T07:00:00+00:00") == date(2026, 10, 11)
    assert fetch.google_day("2026-10-11T06:59:59+00:00") == date(2026, 10, 10)
    assert fetch.google_day("2026-10-11T06:59:59") == date(2026, 10, 10)  # naive is UTC
    assert fetch.google_day(None) is None


def test_to_iso_returns_none_for_unparseable_dates():
    assert fetch._to_iso("not a date") is None
    assert fetch._to_iso(None) is None
