# tests/test_dedup.py
import numpy as np

from openoutnews import dedup


def _row(id_, vector, published_at):
    return {"id": id_, "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
            "published_at": published_at}


def test_near_identical_embeddings_cluster_with_one_representative():
    rows = [
        _row("a", [1.0, 0.0], "2026-09-01T00:00:00+00:00"),
        _row("b", [1.0, 0.0001], "2026-09-01T00:05:00+00:00"),
    ]

    clusters = dedup.cluster_representatives(rows)

    assert clusters == {"a": ["b"]}


def test_dissimilar_embeddings_do_not_cluster():
    rows = [
        _row("a", [1.0, 0.0], "2026-09-01T00:00:00+00:00"),
        _row("b", [0.0, 1.0], "2026-09-01T00:05:00+00:00"),
    ]

    assert dedup.cluster_representatives(rows) == {}


def test_batch_with_no_duplicates_is_unchanged():
    rows = [
        _row("a", [1.0, 0.0, 0.0], "2026-09-01T00:00:00+00:00"),
        _row("b", [0.0, 1.0, 0.0], "2026-09-01T00:05:00+00:00"),
        _row("c", [0.0, 0.0, 1.0], "2026-09-01T00:10:00+00:00"),
    ]

    assert dedup.cluster_representatives(rows) == {}


def test_earliest_published_member_survives_as_representative():
    rows = [
        _row("later", [1.0, 0.0], "2026-09-01T12:00:00+00:00"),
        _row("earliest", [1.0, 0.0001], "2026-09-01T00:00:00+00:00"),
        _row("middle", [1.0, 0.0002], "2026-09-01T06:00:00+00:00"),
    ]

    clusters = dedup.cluster_representatives(rows)

    assert clusters == {"earliest": ["later", "middle"]}


def test_a_missing_published_at_never_wins_earliest_over_a_known_date():
    rows = [
        _row("unknown-date", [1.0, 0.0], None),
        _row("dated", [1.0, 0.0001], "2026-09-01T00:00:00+00:00"),
    ]

    clusters = dedup.cluster_representatives(rows)

    assert clusters == {"dated": ["unknown-date"]}


def test_a_single_row_never_clusters():
    assert dedup.cluster_representatives([_row("a", [1.0, 0.0], "2026-09-01")]) == {}
