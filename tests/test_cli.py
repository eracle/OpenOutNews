# tests/test_cli.py
import csv
import io
import json
import urllib.error
import zlib
from datetime import date

import numpy as np
import pytest

from openoutnews import cli, store
from openoutnews.fetch import RSS_CAP, Candidate

TODAY = date(2026, 10, 12)  # a Monday; the first issue covers 5–11 Oct


def _published(day: date) -> str:
    return f"{day.isoformat()}T18:00:00+00:00"  # 11:00 PDT, well inside the day


def _one_story_per_day(topic, day):
    return [Candidate(title=f"{topic} story {day}", url=f"https://example.com/{topic}/{day}",
                      published_at=_published(day), summary="s")]


def _vector(index: int) -> np.ndarray:
    vector = np.zeros(store.conf.EMBEDDING_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _text_vector(text: str) -> np.ndarray:
    """A random vector seeded by the text: distinct texts sit near cosine 0."""
    rng = np.random.default_rng(zlib.crc32(text.encode()))
    return rng.standard_normal(store.conf.EMBEDDING_DIM).astype(np.float32)


def _distinct_embed(texts):
    """Every text its own vector — nothing deduplicates."""
    return np.array([_text_vector(t) for t in texts], dtype=np.float32)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, fake_model):
    monkeypatch.setattr(cli.issue.fetch, "fetch_day", _one_story_per_day)
    monkeypatch.setattr(cli.issue.embeddings, "embed_texts", _distinct_embed)
    monkeypatch.setattr(cli.fetch, "google_today", lambda: TODAY)


def _rows(capsys):
    return list(csv.DictReader(io.StringIO(capsys.readouterr().out)))


def test_find_without_topics_prints_nothing_and_exits_clean(monkeypatch, capsys):
    monkeypatch.delenv("OPENOUTNEWS_TOPICS", raising=False)
    assert cli.cmd_find(10) == 0
    assert capsys.readouterr().out == ""


def test_first_issue_fetches_each_topic_one_day_at_a_time(monkeypatch):
    queried = []
    monkeypatch.setattr(cli.issue.fetch, "fetch_day",
                        lambda topic, day: queried.append((topic, day)) or [])
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai,sales")

    cli.cmd_find(10)

    days = [date(2026, 10, d) for d in range(5, 12)]
    assert queried == [("ai", d) for d in days] + [("sales", d) for d in days]


def test_issue_window_runs_from_the_last_issue_to_today(monkeypatch):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.cmd_find(10)

    queried = []
    monkeypatch.setattr(cli.fetch, "google_today", lambda: date(2026, 10, 15))
    monkeypatch.setattr(cli.issue.fetch, "fetch_day",
                        lambda topic, day: queried.append(day) or _one_story_per_day(topic, day))
    cli.cmd_find(10)

    assert queried == [date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14)]
    with store.connect() as conn:
        last = store.last_issue(conn)
        assert (last["window_start"], last["window_end"]) == ("2026-10-12", "2026-10-15")


def test_fits_with_no_rejection_yet_are_picked_newest_first(monkeypatch, capsys):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    assert cli.cmd_find(3) == 0

    rows = _rows(capsys)
    assert [r["published_at"][:10] for r in rows] == ["2026-10-11", "2026-10-10", "2026-10-09"]
    assert all(r["issue"] == "1" for r in rows)
    assert all(r["mode"] == "newest" and r["score"] == "" for r in rows)


def test_an_old_unsent_article_is_not_picked(monkeypatch, capsys):
    with store.connect() as conn:
        store.upsert_candidate(conn, title="Old story", url="https://example.com/old",
                               published_at=_published(date(2026, 9, 20)), summary="s")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    cli.cmd_find(100)

    titles = {r["title"] for r in _rows(capsys)}
    assert len(titles) == 7
    assert "Old story" not in titles


def test_a_rewrite_of_a_sent_story_is_absorbed_into_it(monkeypatch, capsys):
    monkeypatch.setattr(cli.issue.embeddings, "embed_texts",
                        lambda texts: np.array([_vector(0) if "release" in t else _text_vector(t)
                                                for t in texts], dtype=np.float32))
    monkeypatch.setattr(cli.issue.fetch, "fetch_day", lambda topic, day: (
        [Candidate(title="Model release", url="https://example.com/release",
                   published_at=_published(day), summary="s")] if day == date(2026, 10, 11) else []
    ))
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.cmd_find(10)
    sent_id = store.article_id("https://example.com/release")
    capsys.readouterr()

    monkeypatch.setattr(cli.fetch, "google_today", lambda: date(2026, 10, 19))
    monkeypatch.setattr(cli.issue.fetch, "fetch_day", lambda topic, day: (
        [Candidate(title="Model release, rewritten", url="https://example.com/rewrite",
                   published_at=_published(day), summary="s")] if day == date(2026, 10, 14)
        else _one_story_per_day(topic, day)
    ))
    cli.cmd_find(10)

    titles = {r["title"] for r in _rows(capsys)}
    assert "Model release, rewritten" not in titles
    with store.connect() as conn:
        assert store.get_row(conn, store.article_id("https://example.com/rewrite"))["absorbed_into"] == sent_id


def test_a_saturated_day_is_recorded(monkeypatch):
    def fetch_day(topic, day):
        if day == date(2026, 10, 7):
            return [Candidate(title=f"t{i}", url=f"https://example.com/{i}",
                              published_at=_published(day), summary="s") for i in range(RSS_CAP)]
        return _one_story_per_day(topic, day)

    monkeypatch.setattr(cli.issue.fetch, "fetch_day", fetch_day)
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.cmd_find(10)

    with store.connect() as conn:
        statuses = {f["day"]: f["status"] for f in store.issue_fetches(conn, 1)}
    assert statuses["2026-10-07"] == "saturated"
    assert statuses["2026-10-08"] == "complete"


def test_a_failed_day_is_recorded_and_the_issue_is_still_built(monkeypatch, capsys):
    def fetch_day(topic, day):
        if day == date(2026, 10, 9):
            raise urllib.error.URLError("timed out")
        return _one_story_per_day(topic, day)

    monkeypatch.setattr(cli.issue.fetch, "fetch_day", fetch_day)
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10) == 0

    assert len(_rows(capsys)) == 6
    with store.connect() as conn:
        failed = [f for f in store.issue_fetches(conn, 1) if f["status"] == "failed"]
    assert [(f["day"], f["error"]) for f in failed] == [("2026-10-09", "<urlopen error timed out>")]


def test_a_second_find_on_the_same_day_prints_that_issue_again(monkeypatch, capsys):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.cmd_find(3)
    first = _rows(capsys)

    assert cli.cmd_find(3) == 0

    assert _rows(capsys) == first
    with store.connect() as conn:
        assert store.last_issue(conn)["id"] == 1


def test_an_empty_pool_builds_no_issue(monkeypatch, capsys):
    monkeypatch.setattr(cli.issue.fetch, "fetch_day", lambda topic, day: [])
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10) == 0

    assert "No candidates" in capsys.readouterr().err
    with store.connect() as conn:
        assert store.last_issue(conn) is None


def test_find_collapses_syndicated_coverage_before_picking(monkeypatch, capsys):
    """Three outlets covering the same event embed near-identically; an
    unrelated story doesn't — only one copy and the unrelated story reach the CSV."""
    monkeypatch.setattr(cli.issue.fetch, "fetch_day", lambda topic, day: (
        [Candidate(title=f"{topic} story", url=f"https://example.com/{topic}",
                   published_at=_published(day), summary="s")] if day == date(2026, 10, 11) else []
    ))
    monkeypatch.setattr(cli.issue.embeddings, "embed_texts",
                        lambda texts: np.array([_vector(1) if "unrelated" in t else _vector(0)
                                                for t in texts], dtype=np.float32))
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "outlet-a,outlet-b,outlet-c,unrelated")

    cli.cmd_find(10)

    titles = {r["title"] for r in _rows(capsys)}
    assert len(titles) == 2
    assert "unrelated story" in titles
    assert len(titles & {"outlet-a story", "outlet-b story", "outlet-c story"}) == 1


# ── Verdicts ──

def _is_release(headline: str) -> bool:
    return "release" in headline


def _releases_and_funding(topic, day):
    """One model release and one funding round each day."""
    return [Candidate(title=f"{kind} {day} - Outlet", url=f"https://example.com/{kind}/{day}",
                      published_at=_published(day), summary="s")
            for kind in ("release", "funding")]


def test_a_fit_and_a_rejection_carry_their_reasons(monkeypatch, capsys, fake_model):
    fake_model.fits = _is_release
    monkeypatch.setattr(cli.issue.fetch, "fetch_day", _releases_and_funding)
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10) == 0

    rows = _rows(capsys)
    assert len(rows) == 7
    assert all(r["title"].startswith("release") for r in rows)
    assert all(r["reason"] == f"fits: {r['title'].removesuffix(' - Outlet')}" for r in rows)
    assert all(r["mode"] == "P(fit)" and r["score"] for r in rows)
    with store.connect() as conn:
        funding = store.get_row(conn, store.article_id("https://example.com/funding/2026-10-08"))
        assert (funding["verdict"], funding["verdict_reason"]) == (0, "rejected: funding 2026-10-08")
        assert funding["sent_at"] is None


def test_the_budget_stops_verdicts(monkeypatch, capsys, fake_model):
    monkeypatch.setenv("OPENOUTNEWS_VERDICTS_PER_ISSUE", "3")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10) == 0

    assert len(fake_model.judged) == 3
    assert len(_rows(capsys)) == 3
    with store.connect() as conn:
        assert store.issue_verdict_count(conn, 1) == 3


def test_the_cold_start_judges_the_article_nearest_the_anchors_first(monkeypatch, fake_model):
    """The anchors embed on axis 0 and so does one mid-week story: it is judged
    first, though four stories are newer."""
    fake_model.anchors = ["Anchor one - Outlet", "Anchor two - Outlet", "Anchor three - Outlet"]
    monkeypatch.setattr(cli.issue.embeddings, "embed_texts", lambda texts: np.array(
        [_vector(0) + 0.01 * _text_vector(t) if "Anchor" in t or "2026-10-07" in t
         else _text_vector(t) for t in texts], dtype=np.float32))
    monkeypatch.setenv("OPENOUTNEWS_VERDICTS_PER_ISSUE", "1")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    cli.cmd_find(10)

    assert fake_model.judged == ["ai story 2026-10-07"]


def test_a_missing_brief_is_an_error_that_says_so(monkeypatch, capsys):
    queried = []
    monkeypatch.setattr(cli.issue.fetch, "fetch_day",
                        lambda topic, day: queried.append(day) or [])
    monkeypatch.delenv("OPENOUTNEWS_BRIEF")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10) == 1

    assert "OPENOUTNEWS_BRIEF is not set" in capsys.readouterr().err
    assert queried == []
    with store.connect() as conn:
        assert store.last_issue(conn) is None


def test_a_malformed_edition_stops_before_fetching(monkeypatch, capsys):
    queried = []
    monkeypatch.setattr(cli.issue.fetch, "fetch_day",
                        lambda topic, day: queried.append(day) or [])
    monkeypatch.setenv("OPENOUTNEWS_EDITION", "italy")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "vino")

    assert cli.cmd_find(10) == 1

    assert "OPENOUTNEWS_EDITION" in capsys.readouterr().err
    assert queried == []


def test_anchors_are_asked_for_in_the_editions_language(monkeypatch, fake_model):
    prompts = []
    answer = fake_model.__call__
    monkeypatch.setattr(cli.llm, "ask", lambda prompt, output_type, timeout=60:
                        prompts.append(prompt) or answer(prompt, output_type, timeout))
    monkeypatch.setenv("OPENOUTNEWS_EDITION", "IT:it")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "vino")

    cli.cmd_find(10)

    assert any("ISO code `it`" in p for p in prompts)


def test_missing_model_settings_are_named(monkeypatch, capsys):
    monkeypatch.delenv("OPENOUTNEWS_LLM_API_KEY")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10) == 1

    assert "OPENOUTNEWS_LLM_API_KEY" in capsys.readouterr().err


def test_the_brief_is_written_once_and_kept(monkeypatch, fake_model):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.cmd_find(10)
    monkeypatch.setattr(cli.fetch, "google_today", lambda: date(2026, 10, 19))
    cli.cmd_find(10)

    assert fake_model.briefs_written == 1
    with store.connect() as conn:
        assert store.get_brief(conn)["written_by"] == "anthropic:claude-haiku-5-5"


def test_brief_flag_replaces_the_stored_brief(monkeypatch, tmp_path, fake_model):
    rules = tmp_path / "rules.txt"
    rules.write_text("- A fit is: an open-weight release\n")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10, brief_path=str(rules)) == 0

    assert fake_model.briefs_written == 0
    with store.connect() as conn:
        brief = store.get_brief(conn)
        assert (brief["rules"], brief["written_by"]) == ("- A fit is: an open-weight release", "--brief")


def test_a_brief_flag_stands_in_for_the_operators_words(monkeypatch, tmp_path):
    rules = tmp_path / "rules.txt"
    rules.write_text("- A fit is: an open-weight release\n")
    monkeypatch.delenv("OPENOUTNEWS_BRIEF")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.cmd_find(10, brief_path=str(rules)) == 0


def test_a_run_that_dies_mid_verdicts_resumes_without_fetching_again(monkeypatch, capsys,
                                                                    fake_model):
    def fits(headline):
        if len(fake_model.judged) == 4:
            raise ConnectionError("model unreachable")
        return True

    fake_model.fits = fits
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    with pytest.raises(ConnectionError):
        cli.cmd_find(10)

    queried = []
    monkeypatch.setattr(cli.issue.fetch, "fetch_day",
                        lambda topic, day: queried.append(day) or [])
    fake_model.fits = lambda headline: True
    assert cli.cmd_find(10) == 0

    assert queried == []
    assert len(_rows(capsys)) == 7
    assert len(set(fake_model.judged)) == 7
    with store.connect() as conn:
        assert store.last_issue(conn)["id"] == 1


def test_reader_labels_are_stored_apart_from_verdicts(monkeypatch, capsys):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.cmd_find(10)
    aid = _rows(capsys)[0]["id"]

    assert cli.cmd_label(aid, "skip") == 0

    with store.connect() as conn:
        row = store.get_row(conn, aid)
        assert (row["label"], row["verdict"]) == (0, 1)


# ── --agent-qualify ──

def _pending_payload(err: str) -> dict:
    """The JSON line under the ``qualify_pending`` line."""
    lines = err.splitlines()
    marker = next(i for i, line in enumerate(lines) if "qualify_pending" in line)
    return json.loads(lines[marker + 1])


def test_agent_qualify_hands_the_article_to_the_agent_instead_of_the_model(monkeypatch, capsys,
                                                                          fake_model):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.main(["find", "10", "--agent-qualify"]) == 1

    err = capsys.readouterr().err
    assert "qualify_pending" in err
    payload = _pending_payload(err)
    assert payload["headline"] == "ai story 2026-10-11"
    assert payload["words"] == "Engineers who self-host open-weight models."
    assert fake_model.judged == []
    assert fake_model.briefs_written == 0


def test_agent_verdicts_are_recorded_and_finish_the_issue(monkeypatch, capsys, fake_model):
    monkeypatch.setenv("OPENOUTNEWS_VERDICTS_PER_ISSUE", "2")
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")
    cli.main(["find", "10", "--agent-qualify"])
    first = _pending_payload(capsys.readouterr().err)["article_id"]

    assert cli.main(["find", "10", "--agent-qualify", "--verdict", "fit",
                     "--reason", "An open-weight release"]) == 1
    second = _pending_payload(capsys.readouterr().err)["article_id"]
    assert second != first

    assert cli.main(["find", "10", "--agent-qualify", "--verdict", "no-fit",
                     "--reason", "A funding round"]) == 0

    rows = _rows(capsys)
    assert [(r["id"], r["reason"]) for r in rows] == [(first, "An open-weight release")]
    assert fake_model.judged == []


def test_a_verdict_with_nothing_pending_is_an_error(monkeypatch, capsys):
    monkeypatch.setenv("OPENOUTNEWS_TOPICS", "ai")

    assert cli.main(["find", "10", "--agent-qualify", "--verdict", "fit", "--reason", "r"]) == 1

    assert "no article is waiting" in capsys.readouterr().err


def test_a_verdict_needs_agent_qualify_and_a_reason(capsys):
    assert cli.main(["find", "10", "--verdict", "fit", "--reason", "r"]) == 1
    assert cli.main(["find", "10", "--agent-qualify", "--verdict", "fit"]) == 1
    assert "go together" in capsys.readouterr().err


# ── label ──

def test_label_records_the_readers_label_on_a_known_article():
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

    assert cli.main(["label"]) == 2
    assert "usage" in capsys.readouterr().err


def test_main_reports_unknown_commands(capsys):
    assert cli.main(["bogus"]) == 2
    assert "invalid choice" in capsys.readouterr().err
