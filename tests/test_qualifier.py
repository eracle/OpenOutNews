# tests/test_qualifier.py
import numpy as np

from openoutnews.ml.qualifier import FitQualifier


def _qualifier(labels, dim=4, seed=42):
    q = FitQualifier(seed=seed, embedding_dim=dim)
    X = np.array([np.full(dim, i, dtype=np.float64) for i in range(len(labels))])
    q.warm_start(X, np.array(labels))
    return q


def test_cold_start_returns_none_before_both_labels_exist():
    q = _qualifier([1, 1, 1])
    assert q.acquisition_mode() is None
    assert q.acquisition_scores(np.zeros((2, 4))) is None


def test_acquisition_mode_explores_while_classes_are_balanced():
    q = _qualifier([1, 0])
    assert q.acquisition_mode() == "explore (BALD)"


def test_acquisition_mode_exploits_once_negatives_outnumber_positives():
    q = _qualifier([1, 0, 0])
    assert q.acquisition_mode() == "exploit (p)"


def test_acquisition_scores_returns_one_score_per_candidate():
    q = _qualifier([1, 0, 0])
    result = q.acquisition_scores(np.random.RandomState(0).randn(5, 4))
    assert result is not None
    mode, scores = result
    assert mode == "exploit (p)"
    assert scores.shape == (5,)


def test_update_appends_an_observation_and_invalidates_the_fit():
    q = _qualifier([1, 0])
    q._fit_if_needed()
    assert q._fitted is True

    q.update(np.zeros(4), 0)

    assert q._fitted is False
    assert q.n_obs == 3
    assert q.class_counts == (2, 1)


# ── Anchors ──

def _anchors(dim=4):
    return np.array([np.eye(dim)[0], np.eye(dim)[0] + 0.1, np.eye(dim)[0] - 0.1])


def test_anchors_let_the_gp_fit_after_one_rejection():
    q = _qualifier([0])
    assert q.acquisition_mode() is None

    q.set_anchors(_anchors())

    assert q.acquisition_mode() is not None
    assert q.class_counts == (1, 3)


def test_anchors_alone_cannot_fit():
    q = FitQualifier(embedding_dim=4)
    q.set_anchors(_anchors())
    assert q.acquisition_scores(np.zeros((2, 4))) is None


def test_the_cold_phase_exploits_even_while_classes_are_even():
    """Three anchors and three rejections: the balance alone would explore."""
    q = _qualifier([0, 0, 0])
    q.set_anchors(_anchors())

    assert q.is_cold
    assert q.acquisition_mode() == "exploit (p)"


def test_the_cold_phase_ends_once_real_fits_match_the_anchors():
    q = _qualifier([1, 1, 1, 0])
    q.set_anchors(_anchors())

    assert not q.is_cold


def test_set_anchors_replaces_rather_than_adds():
    q = FitQualifier(embedding_dim=4)
    q.set_anchors(_anchors())
    q.set_anchors(_anchors())
    assert q.n_anchors == 3


def test_anchor_similarity_ranks_by_the_nearest_anchor():
    q = FitQualifier(embedding_dim=4)
    q.set_anchors(_anchors())

    scores = q.anchor_similarity(np.array([np.eye(4)[1], np.eye(4)[0] * 2, np.zeros(4)]))

    assert int(np.argmax(scores)) == 1
    assert scores[2] == 0.0


def test_anchor_similarity_is_none_unanchored():
    assert FitQualifier(embedding_dim=4).anchor_similarity(np.zeros((1, 4))) is None
