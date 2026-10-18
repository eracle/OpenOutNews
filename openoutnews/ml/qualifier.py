# openoutnews/ml/qualifier.py
"""GP Regression qualifier: BALD active learning via exact GP posterior.

``FitQualifier`` is ``openoutlearn.GPBaldQualifier`` trained on the model's
verdicts (1 = fits the newsletter, 0 = not), plus the cold start OpenOutFind's
``BayesianQualifier`` already solved: invented ideal headlines (the anchors)
held as permanent positives, so the GP can fit after the first rejection
instead of after the first fit *and* the first rejection, and exploit while
the positive class is still mostly invented. The anchor code is
``BayesianQualifier``'s, minus its calibration counts and its balancing; if
the two stay identical, it belongs in ``GPBaldQualifier``.
"""
from __future__ import annotations

import numpy as np

from openoutlearn.qualifier import GPBaldQualifier

from openoutnews import conf


class FitQualifier(GPBaldQualifier):
    """The shared GP+BALD engine, scoring articles by fit, anchored on a cold start."""

    def __init__(self, seed: int = 42, embedding_dim: int = 384, n_mc_samples: int = 100):
        super().__init__(seed=seed, embedding_dim=embedding_dim, n_mc_samples=n_mc_samples)
        # Invented ideal-headline embeddings, all label 1 — kept apart from the
        # real verdicts and never trimmed as real fits arrive.
        self._anchor_X: list[np.ndarray] = []

    @property
    def n_obs(self) -> int:
        return len(self._y) + len(self._anchor_X)

    @property
    def class_counts(self) -> tuple[int, int]:
        """Return (n_negatives, n_positives) — anchors counted as positives."""
        n_pos = sum(self._y) + len(self._anchor_X)
        return len(self._y) - sum(self._y), n_pos

    @property
    def n_anchors(self) -> int:
        return len(self._anchor_X)

    @property
    def is_cold(self) -> bool:
        """Whether the positive class is still mostly invented: fewer real fits
        than anchors."""
        return bool(self._anchor_X) and sum(self._y) < conf.ANCHOR_COUNT

    def set_anchors(self, embeddings: np.ndarray):
        """Replace the anchor set. The caller owns the whole set, so passing
        the stored one again is a no-op."""
        self._anchor_X = [np.asarray(e, dtype=np.float64).ravel() for e in embeddings]
        self._fitted = False

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Real verdicts plus any anchors, as ``(X, y)`` — what the GP fits on."""
        X = self._X + self._anchor_X
        y = self._y + [1] * len(self._anchor_X)
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

    def acquisition_mode(self) -> str | None:
        """Exploit while cold, then the engine's balance-driven choice.

        While the anchors outnumber the real fits, the one goal is a real fit,
        and the highest-P article is the one most like the ideal headlines.
        BALD would spend each verdict on the article the model is most confused
        about — the one least like them (OpenOutFind measured exactly that).
        """
        if self._fit_if_needed() and self.is_cold:
            return "exploit (p)"
        return super().acquisition_mode()

    def anchor_similarity(self, embeddings: np.ndarray) -> np.ndarray | None:
        """Each embedding's cosine similarity to its nearest anchor — the
        ranking before the GP can fit (no rejection yet). ``None`` unanchored."""
        if not self._anchor_X:
            return None
        anchors = np.array(self._anchor_X)
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
        vectors = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms @ anchors.T).max(axis=1)
