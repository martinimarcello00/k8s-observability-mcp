"""PCA baseline on B0: per service, the subspace of healthy behaviour, and how far each step falls outside it.

Fit on healthy steps only. Per service, each feature is z-scored with the training mean and deviation, and the SVD of
the centred matrix gives the principal directions; the first k keep VAR of the variance. The score of a step is its
squared residual outside those k directions, the SPE (the Q statistic of Jackson & Mudholkar, 1979). The SPE is a sum
over features, so every alarm says which features make it. A score counts only when it holds for two steps (persist).

restarts is not a PCA input. It is 0 in every healthy run, so its deviation is 0 and its z-score undefined: inside the
PCA it would be dropped without a word. It is read as a rule instead (anomaly/models/evaluate.py): a service that
restarts is in alarm.
"""
from types import SimpleNamespace
import warnings

import numpy as np

from anomaly import inputs

FEATURES = tuple(f for f in inputs.B0 if f != "restarts")
VAR = 0.95  # share of the healthy variance the subspace keeps
MIN_FINITE = 0.95  # a column enters the model of a service when it is finite in this share of the training steps
MIN_SD = 1e-6  # a column flatter than this in training is left out: its z-score would explode on any change


def persist(e):
    """A score holds at t when it held at t-1 too: the smaller of the two, along axis 0 (0 at the first step)."""
    return np.minimum(e, np.concatenate([np.zeros_like(e[:1]), e[:-1]]))


class PCA:
    def fit(self, X):
        """X: list of [T, S, len(FEATURES)] arrays from inputs.prepare, healthy steps only."""
        A = np.concatenate(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: features a service does not have
            self.mu, self.sd = np.nanmean(A, 0), np.nanstd(A, 0)
        Z = self._z(A)
        self.services = []
        for i in range(A.shape[1]):
            cols = np.isfinite(Z[:, i]).mean(0) >= MIN_FINITE
            if not cols.any():
                self.services.append(None)
                continue
            B = Z[:, i][:, cols]
            B = B[np.isfinite(B).all(1)]
            mean = B.mean(0)
            _, sv, Vt = np.linalg.svd(B - mean, full_matrices=False)
            kept = np.cumsum(sv**2) / np.sum(sv**2)
            k = min(int(np.searchsorted(kept, VAR)) + 1, len(sv))
            self.services.append(SimpleNamespace(cols=cols, mean=mean, V=Vt[:k].T, variance=float(kept[k - 1])))
        return self

    def _z(self, X):
        return (X - self.mu) / np.where(self.sd > MIN_SD, self.sd, np.nan)

    def score(self, X):
        """One run [T, S, F] -> (r2, expected), both [T, S, F].

        r2: each feature's squared residual outside the healthy subspace, 0 for the columns out of the model; its sum
        over F is the SPE. expected: the value the subspace reconstructs, in prepare units, NaN out of the model.
        A missing value counts as the training mean: the PCA does not see a pod that is gone.
        """
        Z = self._z(X)
        r2, expected = np.zeros(X.shape), np.full(X.shape, np.nan)
        for i, m in enumerate(self.services):
            if m is None:
                continue
            d = np.nan_to_num(Z[:, i][:, m.cols] - m.mean)
            proj = d @ m.V @ m.V.T
            r2[:, i, m.cols] = (d - proj) ** 2
            expected[:, i, m.cols] = (m.mean + proj) * self.sd[i, m.cols] + self.mu[i, m.cols]
        return r2, expected

    def spe(self, X):
        """[T, S] persistent SPE of one run."""
        return persist(self.score(X)[0].sum(2))

    def calibrate(self, runs):
        """Threshold per service: the largest persistent SPE over healthy runs, given as (X, steps) pairs."""
        thr = np.max([self.spe(X)[steps].max(0) for X, steps in runs], 0)
        self.thr = np.where(thr > 0, thr, np.inf)  # a service that never scored cannot alarm
        return self

    def card(self, services):
        """What the model is, per service: directions kept, variance they hold, threshold, columns left out."""
        return {s: None if m is None else {
            "k": int(m.V.shape[1]), "columns": int(m.cols.sum()), "variance_kept": round(m.variance, 4),
            "threshold": float(self.thr[i]), "left_out": [f for f, c in zip(FEATURES, m.cols) if not c]}
            for i, (s, m) in enumerate(zip(services, self.services))}


def explain(X, r2, expected, t, i, top=3):
    """The features that make the SPE of service i at step t, largest first: [(feature, share, observed, expected)],
    values in inputs.UNITS."""
    c = r2[t, i]
    order = [j for j in np.argsort(-c)[:top] if c[j] > 0]
    return [(FEATURES[j], float(c[j] / c.sum()), float(inputs.to_units(X[t, i, j], FEATURES[j])),
             float(inputs.to_units(expected[t, i, j], FEATURES[j]))) for j in order]
