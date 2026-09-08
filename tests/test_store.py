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
    store.mark_sent(conn, [aid])
    assert store.unsent_candidates(conn) == []


def test_unsent_candidates_excludes_unembedded_rows(conn):
    store.upsert_candidate(conn, title="A", url="https://example.com/a", published_at="2026-09-01", summary="")
    assert store.unsent_candidates(conn) == []


def test_labeled_arrays_is_empty_with_no_labels(conn):
    X, y = store.labeled_arrays(conn)
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


def test_labeled_arrays_collects_every_labelled_embedded_row(conn):
    for i, verdict in enumerate([1, 0]):
        url = f"https://example.com/{i}"
        store.upsert_candidate(conn, title=str(i), url=url, published_at=None, summary="")
        aid = store.article_id(url)
        store.set_embedding(conn, aid, np.full(store.conf.EMBEDDING_DIM, i, dtype=np.float32))
        store.label_article(conn, aid, verdict)

    X, y = store.labeled_arrays(conn)

    assert X.shape == (2, store.conf.EMBEDDING_DIM)
    assert sorted(y.tolist()) == [0, 1]
