"""PCA baseline on B0: per service, the subspace of healthy behaviour, and how far each step falls outside it.

Fit on healthy steps only. Per service, each feature is z-scored with the training mean and deviation, and the SVD of
the centred matrix gives the principal directions; the first k keep VAR of the variance. The score of a step is its
squared residual outside those k directions, the SPE (the Q statistic of Jackson & Mudholkar, 1979). The SPE is a sum
over features, so every alarm says which features make it.

This file is the model only. Persistence, thresholds, the restart rule and the metrics are the protocol, shared by every
model, in anomaly/models/evaluate.py. restarts is not a PCA input: it is 0 in every healthy run, so its deviation is 0
and its z-score undefined; the protocol reads it as a rule instead.
"""
from types import SimpleNamespace
import warnings

import numpy as np

from anomaly import inputs

VAR = 0.95  # share of the healthy variance the subspace keeps
MIN_FINITE = 0.95  # a column enters the model of a service when it is finite in this share of the training steps
MIN_SD = 1e-6  # a column flatter than this in training is left out: its z-score would explode on any change


class PCA:
    features = tuple(f for f in inputs.B0 if f != "restarts")
    params = {"VAR": VAR, "MIN_FINITE": MIN_FINITE, "MIN_SD": MIN_SD}

    def fit(self, X):
        """X: list of [T, S, len(features)] arrays from inputs.prepare, healthy steps only."""
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

    def card(self, services):
        """What the model learned, per service: directions kept, variance they hold, columns left out."""
        return {s: None if m is None else {
            "k": int(m.V.shape[1]), "columns": int(m.cols.sum()), "variance_kept": round(m.variance, 4),
            "left_out": [f for f, c in zip(self.features, m.cols) if not c]}
            for s, m in zip(services, self.services)}
