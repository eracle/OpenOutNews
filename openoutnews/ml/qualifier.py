# openoutnews/ml/qualifier.py
"""GP Regression qualifier: BALD active learning via exact GP posterior.

Copied forward from OpenOutFind's ``core/ml/qualifier.py`` (``BayesianQualifier``)
and adapted to a new domain: the label is reader *engagement* (1 = read, 0 =
skip) instead of an LLM's ICP-fit verdict, and the thing being ranked is a news
article's embedding instead of a lead's.

Dropped on the way over, deliberately:

- **Anchors / cold-start invented positives.** OpenOutFind needs them because a
  campaign with zero acceptances has no positive class at all and the GP is
  unfittable from the first run. Here the cold start is accepted instead: the
  qualifier returns ``None`` until both a "read" and a "skip" exist, and the
  caller falls back to newest-first. No invented reader taste to seed it with.
- **Django / Campaign coupling and the per-process fit cache.** That cache
  existed because a resident daemon queried the same campaign's qualifier many
  times per run; ``outnews find`` fits once and exits, so there is nothing to
  cache.
- **Model persistence.** The store keeps every labelled ``(embedding, label)``
  pair; refitting from them each run is cheap at newsletter-label volumes.

This is the piece a future shared active-learning library would be extracted
from, once this repo and OpenOutFind both have a working version to design the
interface against — see the OpenOutNews roadmap card in openoutreach-docs.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _binary_entropy(p):
    """H(p) = -p log p - (1-p) log(1-p), safe for edge values."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def _prob_above_half(mean, std):
    """P(f > 0.5) from GP posterior."""
    return norm.sf(0.5, loc=mean, scale=std)


def _gpr_predict(pipe, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform through all steps except GPR, then predict with return_std."""
    from sklearn.pipeline import Pipeline

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_transformed = Pipeline(pipe.steps[:-1]).transform(X)
    return pipe.named_steps["gpr"].predict(X_transformed, return_std=True)


# ---------------------------------------------------------------------------
# EngagementQualifier  (GP Regression backend)
# ---------------------------------------------------------------------------

class EngagementQualifier:
    """Gaussian Process Regressor for active-learning news selection.

    Same mechanism as OpenOutFind's ``BayesianQualifier``: an sklearn Pipeline
    (StandardScaler -> GPR) fitted on every labelled observation, P(f > 0.5)
    read off the exact GP posterior, and BALD scores via MC sampling for the
    explore side. Training data is accumulated externally (the caller loads it
    from the store) and handed in via ``warm_start``.
    """

    def __init__(self, seed: int = 42, embedding_dim: int = 384, n_mc_samples: int = 100):
        self.embedding_dim = embedding_dim
        self._seed = seed
        self._n_mc_samples = n_mc_samples
        self._pipeline = None
        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._fitted = False
        self._rng = np.random.RandomState(seed)

    @property
    def n_obs(self) -> int:
        return len(self._y)

    @property
    def class_counts(self) -> tuple[int, int]:
        """Return (n_negatives, n_positives)."""
        n_pos = sum(self._y)
        return len(self._y) - n_pos, n_pos

    # ------------------------------------------------------------------
    # Update  (append + invalidate)
    # ------------------------------------------------------------------

    def update(self, embedding: np.ndarray, label: int):
        """Record a new labelled observation. Model is lazily re-fitted."""
        self._X.append(np.asarray(embedding, dtype=np.float64).ravel())
        self._y.append(int(label))
        self._fitted = False

    def warm_start(self, X: np.ndarray, y: np.ndarray):
        """Bulk-load labels from the store. Replaces any prior observations."""
        self._X = [np.asarray(X[i], dtype=np.float64).ravel() for i in range(len(X))]
        self._y = [int(y[i]) for i in range(len(y))]
        self._fitted = False

    # ------------------------------------------------------------------
    # Lazy refit
    # ------------------------------------------------------------------

    def _fit_if_needed(self) -> bool:
        """Fit StandardScaler + GPR pipeline if dirty and feasible.

        Returns True once the model is usable — False during cold start, when
        fewer than two labels exist or only one class has been seen. There are
        no anchors here to paper over that gap; the caller falls back to
        newest-first until a "read" and a "skip" both exist.
        """
        if self._fitted:
            return True
        if len(self._y) < 2 or len(set(self._y)) < 2:
            return False

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array(self._X, dtype=np.float64)
        y = np.array(self._y, dtype=np.float64)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=np.sqrt(self.embedding_dim)),
                n_restarts_optimizer=3,
                random_state=self._seed,
                alpha=0.1,
            )),
        ])
        started = time.monotonic()
        self._pipeline.fit(X, y)
        self._fitted = True
        logger.debug("qualifier fitted on %d labelled article(s) in %.2fs",
                     len(y), time.monotonic() - started)
        return True

    # ------------------------------------------------------------------
    # Prediction  (needs posterior std)
    # ------------------------------------------------------------------

    def predict_probs(self, embeddings: np.ndarray) -> np.ndarray | None:
        """Predicted probability P(f > 0.5) for each candidate.

        Returns None when the model cannot be fitted yet.
        """
        if not self._fit_if_needed():
            return None
        mean, std = _gpr_predict(self._pipeline, embeddings)
        return _prob_above_half(mean, std)

    # ------------------------------------------------------------------
    # BALD acquisition via GP posterior
    # ------------------------------------------------------------------

    def compute_bald(self, embeddings: np.ndarray) -> np.ndarray | None:
        """BALD scores for (N, embedding_dim) candidates.

        BALD = H(E[p]) - E[H(p)], MC-sampled from the exact GP posterior with a
        probit link. Higher BALD = the model disagrees with itself most = most
        informative to ask the reader about.
        """
        if not self._fit_if_needed():
            return None

        f_mean, f_std = _gpr_predict(self._pipeline, embeddings)
        f_samples = (
            f_mean[np.newaxis, :]
            + f_std[np.newaxis, :] * self._rng.randn(self._n_mc_samples, len(f_mean))
        )
        p_samples = norm.cdf(f_samples - 0.5)
        p_pred = p_samples.mean(axis=0)
        H_pred = _binary_entropy(p_pred)
        H_individual = _binary_entropy(p_samples).mean(axis=0)
        return H_pred - H_individual

    # ------------------------------------------------------------------
    # Acquisition mode  (balance-driven, no cold-phase override)
    # ------------------------------------------------------------------

    def acquisition_mode(self) -> str | None:
        """The live acquisition axis: ``"exploit (p)"``, ``"explore (BALD)"``, or None.

        Exploit once real skips outnumber real reads — the model has enough
        rejections to trust its favourites; explore while the classes are
        still even, spending picks on the articles it's most unsure about.
        None on cold start (fewer than one of each label).
        """
        if not self._fit_if_needed():
            return None
        n_neg, n_pos = self.class_counts
        return "exploit (p)" if n_neg > n_pos else "explore (BALD)"

    def acquisition_scores(self, embeddings: np.ndarray) -> tuple[str, np.ndarray] | None:
        """Score candidates using the balance-driven acquisition strategy.

        Returns ``(strategy_name, scores)`` or ``None`` on cold start.
        """
        strategy = self.acquisition_mode()
        if strategy is None:
            return None
        scores = (
            self.predict_probs(embeddings) if strategy == "exploit (p)"
            else self.compute_bald(embeddings)
        )
        return strategy, scores
