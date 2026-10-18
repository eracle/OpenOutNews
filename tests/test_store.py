# tests/test_store.py
import numpy as np

from openoutnews import store


def test_upsert_candidate_inserts_a_new_row(conn):
    store.upsert_candidate(
        conn, title="Title", url="https://example.com/a",
        published_at="2026-09-01T00:00:00+00:00", summary="summary",
    )
    row = store.get_row(conn, store.article_id("https://example.com/a"))
    assert row["title"] == "Title"
    assert row["embedding"] is None
    assert row["label"] is None


def test_upsert_candidate_never_overwrites_an_existing_row(conn):
    url = "https://example.com/a"
    store.upsert_candidate(conn, title="First", url=url, published_at=None, summary="")
    aid = store.article_id(url)
    store.label_article(conn, aid, 1)

    store.upsert_candidate(conn, title="Second", url=url, published_at=None, summary="")

    row = store.get_row(conn, aid)
    assert row["title"] == "First"
    assert row["label"] == 1


def test_set_embedding_round_trips_through_the_blob(conn):
    store.upsert_candidate(conn, title="T", url="https://example.com/b", published_at=None, summary="")
    aid = store.article_id("https://example.com/b")
    vector = np.arange(4, dtype=np.float32)

    store.set_embedding(conn, aid, vector)

    row = store.get_row(conn, aid)
    np.testing.assert_array_equal(np.frombuffer(row["embedding"], dtype=np.float32), vector)


def test_unembedded_ids_lists_only_rows_with_no_vector(conn):
    store.upsert_candidate(conn, title="A", url="https://example.com/a", published_at=None, summary="")
    store.upsert_candidate(conn, title="B", url="https://example.com/b", published_at=None, summary="")
    aid_b = store.article_id("https://example.com/b")
    store.set_embedding(conn, aid_b, np.zeros(4, dtype=np.float32))

    pending = store.unembedded_ids(conn)

    assert pending == [store.article_id("https://example.com/a")]


def test_get_row_returns_none_for_an_unknown_id(conn):
    assert store.get_row(conn, "does-not-exist") is None


def test_mark_sent_excludes_rows_from_unsent_candidates(conn):
    store.upsert_candidate(conn, title="A", url="https://example.com/a", published_at="2026-09-01", summary="")
    aid = store.article_id("https://example.com/a")
    store.set_embedding(conn, aid, np.zeros(4, dtype=np.float32))

    assert len(store.unsent_candidates(conn)) == 1
    store.mark_sent(conn, 1, [(aid, None)])
    assert store.unsent_candidates(conn) == []


def test_issue_picks_come_back_in_score_order(conn):
    issue_id = store.create_issue(conn, window_start="2026-10-05", window_end="2026-10-12")
    for name in ("low", "high"):
        store.upsert_candidate(conn, title=name, url=f"https://example.com/{name}",
                               published_at=None, summary="")
    store.mark_sent(conn, issue_id, [(store.article_id("https://example.com/low"), 0.2),
                                     (store.article_id("https://example.com/high"), 0.9)])

    assert [r["title"] for r in store.issue_picks(conn, issue_id)] == ["high", "low"]
    assert store.last_issue(conn)["window_end"] == "2026-10-12"


def test_an_issue_is_unfinished_until_finish_issue(conn):
    issue_id = store.create_issue(conn, window_start="2026-10-05", window_end="2026-10-12")
    store.upsert_candidate(conn, title="A", url="https://example.com/a", published_at=None, summary="")
    aid = store.article_id("https://example.com/a")
    store.set_pending(conn, issue_id, aid)
    assert store.last_issue(conn)["finished_at"] is None

    store.finish_issue(conn, issue_id, mode="P(fit)", picks=[(aid, 0.8)])

    issue = store.last_issue(conn)
    assert issue["finished_at"] is not None
    assert (issue["mode"], issue["pending_article_id"]) == ("P(fit)", None)
    assert store.get_row(conn, aid)["issue_id"] == issue_id


def test_last_issue_is_none_before_the_first(conn):
    assert store.last_issue(conn) is None


def test_record_fetch_keeps_each_day_query_of_an_issue(conn):
    issue_id = store.create_issue(conn, window_start="2026-10-05", window_end="2026-10-12")
    store.record_fetch(conn, issue_id, topic="ai", day="2026-10-05", status="saturated",
                       returned=100, error=None)
    store.record_fetch(conn, issue_id, topic="ai", day="2026-10-06", status="failed",
                       returned=None, error="timed out")

    fetches = store.issue_fetches(conn, issue_id)

    assert [(f["day"], f["status"]) for f in fetches] == [("2026-10-05", "saturated"),
                                                          ("2026-10-06", "failed")]


def test_recently_sent_returns_only_rows_sent_since(conn):
    store.upsert_candidate(conn, title="A", url="https://example.com/a", published_at=None, summary="")
    aid = store.article_id("https://example.com/a")
    store.set_embedding(conn, aid, np.zeros(4, dtype=np.float32))
    store.mark_sent(conn, 1, [(aid, None)])

    assert [r["id"] for r in store.recently_sent(conn, "2000-01-01T00:00:00+00:00")] == [aid]
    assert store.recently_sent(conn, "2999-01-01T00:00:00+00:00") == []


def test_connect_adds_issue_columns_to_an_older_store(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE articles (id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL, "
                "published_at TEXT, summary TEXT, embedding BLOB, label INTEGER, "
                "fetched_at TEXT NOT NULL, sent_at TEXT)")
    old.commit()
    old.close()
    monkeypatch.setenv("OPENOUTNEWS_DB", str(path))

    with store.connect() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}

    assert {"absorbed_into", "issue_id", "issue_score",
            "verdict", "verdict_reason", "verdict_issue_id", "repeat_of"} <= columns


def test_unsent_candidates_excludes_unembedded_rows(conn):
    store.upsert_candidate(conn, title="A", url="https://example.com/a", published_at="2026-09-01", summary="")
    assert store.unsent_candidates(conn) == []


def test_verdict_arrays_is_empty_with_no_verdicts(conn):
    X, y = store.verdict_arrays(conn)
    assert X.shape == (0, store.conf.EMBEDDING_DIM)
    assert y.shape == (0,)


def test_absorb_candidates_excludes_absorbed_rows_from_unsent_candidates(conn):
    for i, url in enumerate(["https://example.com/rep", "https://example.com/dup"]):
        store.upsert_candidate(conn, title=str(i), url=url, published_at="2026-09-01", summary="")
        store.set_embedding(conn, store.article_id(url), np.zeros(4, dtype=np.float32))
    rep_id = store.article_id("https://example.com/rep")
    dup_id = store.article_id("https://example.com/dup")

    assert len(store.unsent_candidates(conn)) == 2
    store.absorb_candidates(conn, {rep_id: [dup_id]})

    remaining = store.unsent_candidates(conn)
    assert [r["id"] for r in remaining] == [rep_id]
    assert store.get_row(conn, dup_id)["absorbed_into"] == rep_id


def test_verdict_arrays_collects_every_judged_row_and_no_reader_label(conn):
    for i, verdict in enumerate([1, 0]):
        url = f"https://example.com/{i}"
        store.upsert_candidate(conn, title=str(i), url=url, published_at=None, summary="")
        aid = store.article_id(url)
        store.set_embedding(conn, aid, np.full(store.conf.EMBEDDING_DIM, i, dtype=np.float32))
        store.set_verdict(conn, aid, 1, verdict, "reason")
    store.upsert_candidate(conn, title="read", url="https://example.com/read", published_at=None, summary="")
    read_id = store.article_id("https://example.com/read")
    store.set_embedding(conn, read_id, np.zeros(store.conf.EMBEDDING_DIM, dtype=np.float32))
    store.label_article(conn, read_id, 1)

    X, y = store.verdict_arrays(conn)

    assert X.shape == (2, store.conf.EMBEDDING_DIM)
    assert sorted(y.tolist()) == [0, 1]
    assert store.issue_verdict_count(conn, 1) == 2


def test_set_brief_replaces_the_brief(conn):
    assert store.get_brief(conn) is None
    store.set_brief(conn, "first", "--brief")
    store.set_brief(conn, "second", "anthropic:claude-haiku-5-5")

    brief = store.get_brief(conn)
    assert (brief["rules"], brief["written_by"]) == ("second", "anthropic:claude-haiku-5-5")
    assert conn.execute("SELECT COUNT(*) FROM brief").fetchone()[0] == 1


def test_anchors_round_trip_oldest_first(conn):
    store.add_anchor(conn, "One - Outlet", np.ones(4, dtype=np.float32))
    store.add_anchor(conn, "Two - Outlet", np.zeros(4, dtype=np.float32))

    rows = store.anchor_rows(conn)

    assert [r["title"] for r in rows] == ["One - Outlet", "Two - Outlet"]
    np.testing.assert_array_equal(np.frombuffer(rows[0]["embedding"], dtype=np.float32), np.ones(4))
