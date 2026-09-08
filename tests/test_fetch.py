# tests/test_fetch.py
from unittest.mock import MagicMock, patch

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
        candidates = fetch.fetch_candidates("open source AI")

    assert len(candidates) == 1
    assert candidates[0].title == "First story"
    assert candidates[0].url == "https://example.com/first"
    assert candidates[0].published_at == "2026-08-26T10:00:00+00:00"
    assert candidates[0].summary == "Summary one"


def test_fetch_candidates_skips_items_missing_a_link_or_title():
    body = b"<rss><channel><item><title></title><link>https://example.com/x</link></item></channel></rss>"
    with patch("openoutnews.fetch.urllib.request.urlopen", return_value=_urlopen_returning(body)):
        candidates = fetch.fetch_candidates("topic")

    assert candidates == []


def test_to_iso_returns_none_for_unparseable_dates():
    assert fetch._to_iso("not a date") is None
    assert fetch._to_iso(None) is None
