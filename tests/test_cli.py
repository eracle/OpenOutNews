# tests/test_cli.py
import csv
import io
from unittest.mock import patch

import numpy as np
import pytest

from openoutnews import cli, store
from openoutnews.fetch import Candidate


def _fake_candidates(topic):
    return [Candidate(title=f"{topic} story", url=f"https://example.com/{topic}",
                       published_at="2026-09-01T00:00:00+00:00", summary="s")]


def _fake_embed(texts):
    return np.zeros((len(texts), store.conf.EMBEDDING_DIM), dtype=np.float32)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(cli.fetch, "fetch_candidates", _fake_candidates)
    monkeypatch.setattr(cli.embeddings, "embed_texts", _fake_embed)


def test_find_without_topics_prints_nothing_and_exits_clean(monkeypatch, capsys):
    monkeypatch.delenv("OPENOUTNEWS_TOPICS", raising=False)
    assert cli.cmd_find(10) == 0
    assert capsys.readouterr().out == ""


def test_find_cold_start_falls_back_to_newest_first(monkeypatch, capsys):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai,sales")
    assert cli.cmd_find(10) == 0

    out = capsys.readouterr().out
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 2
    assert all(r["mode"] == "cold-start (newest)" for r in rows)
    assert all(r["score"] == "" for r in rows)


def test_find_marks_picked_articles_as_sent(monkeypatch):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.cmd_find(10)

    with store.connect() as conn:
        assert store.unsent_candidates(conn) == []


def test_find_n_limits_how_many_are_picked(monkeypatch, capsys):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai,sales,crm")
    cli.cmd_find(1)

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 1


def test_find_collapses_syndicated_coverage_before_picking(monkeypatch, capsys):
    """Three outlets covering the same event embed near-identically; a
    fourth, unrelated topic doesn't — only the representative and the
    unrelated story should reach the CSV."""
    def fake_candidates(topic):
        return [Candidate(title=f"{topic} story", url=f"https://example.com/{topic}",
                           published_at="2026-09-01T00:00:00+00:00", summary="s")]

    def fake_embed(texts):
        vectors = []
        for text in texts:
            if "unrelated" in text:
                vectors.append(np.array([0.0, 1.0] + [0.0] * (store.conf.EMBEDDING_DIM - 2),
                                         dtype=np.float32))
            else:
                vectors.append(np.array([1.0, 0.0] + [0.0] * (store.conf.EMBEDDING_DIM - 2),
                                         dtype=np.float32))
        return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(cli.fetch, "fetch_candidates", fake_candidates)
    monkeypatch.setattr(cli.embeddings, "embed_texts", fake_embed)
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "outlet-a,outlet-b,outlet-c,unrelated")

    cli.cmd_find(10)

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    titles = {r["title"] for r in rows}
    assert len(rows) == 2
    assert "unrelated story" in titles
    assert len(titles & {"outlet-a story", "outlet-b story", "outlet-c story"}) == 1


def test_label_records_a_verdict_on_a_known_article():
    with store.connect() as conn:
        store.upsert_candidate(conn, title="T", url="https://example.com/x",
                                published_at=None, summary="")
    aid = store.article_id("https://example.com/x")

    assert cli.cmd_label(aid, "read") == 0

    with store.connect() as conn:
        assert store.get_row(conn, aid)["label"] == 1


def test_label_rejects_an_unknown_verdict(capsys):
    assert cli.cmd_label("whatever", "maybe") == 1
    assert "unknown label" in capsys.readouterr().err


def test_label_fails_for_an_unknown_article_id(capsys):
    assert cli.cmd_label("does-not-exist", "read") == 1
    assert "no article" in capsys.readouterr().err


def test_main_dispatches_find_and_label(monkeypatch, capsys):
    monkeypatch.delenv("OPENOUTNEWS_TOPICS", raising=False)
    assert cli.main(["find"]) == 0

    assert cli.main(["label"]) == 1
    assert "usage" in capsys.readouterr().err


def test_main_reports_unknown_commands(capsys):
    assert cli.main(["bogus"]) == 1
    assert "unknown command" in capsys.readouterr().err
