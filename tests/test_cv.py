# Run from the repo root: PYTHONPATH=. python tests/test_cv.py  (no cluster, no data needed)
from types import SimpleNamespace

import numpy as np
import optuna

from anomaly import inputs
from anomaly.models import cv, evaluate as ev

S = 2


def run(i, role, T, fault=False):
    """A run whose feature 0 is the number i everywhere: a stub model that scores it tells which runs it was given."""
    X = np.zeros((T, S, len(inputs.B0)))
    X[:, :, 0] = i
    t = np.arange(T) * 30.0
    return SimpleNamespace(id=f"{role}/r{i}", role=role, fold="dev" if fault else None, t=t, on=np.ones(T, bool),
                           services=["a", "b"], t_inject=-1.0 if fault else None, fault=np.full(T, fault),
                           healthy=np.zeros(T, bool), X=X, absent=np.zeros((T, S), bool), targets=[0] if fault else [],
                           scenario="x", source=role)


class Stub:
    """Scores a step with its run number: the threshold is the largest number among the runs it was fit on."""
    features = (inputs.B0[0],)
    params = {}
    seen = []

    def fit(self, X, val, services):
        Stub.seen.append(({int(x[0, 0, 0]) for x in X}, {int(x[0, 0, 0]) for x in val}))

    def score(self, X):
        return X.copy(), np.zeros_like(X)


ev.MODELS["stub"] = Stub
sessions = [run(i, "train", 180 if i <= 6 else 90) for i in range(1, 12)] + [run(i, "val", 180) for i in (12, 13)]
others = [run(20, "test", 90), run(21, "test-unseen", 90), run(22, "noop", 30)]
dev = [run(30 + i, "fault-dev", 30, fault=True) for i in range(3)]
runs = sessions + others + dev

# --- folds: a partition of the clean train and val sessions only, balanced by steps
fs = cv.folds(runs)
ids = [r.id for f in fs for r in f]
assert sorted(ids) == sorted(r.id for r in sessions) and len(set(ids)) == len(ids)
steps = [sum(int(r.on.sum()) for r in f) for f in fs]
assert max(steps) - min(steps) <= 180, steps

# --- each fold is fit on the other folds only, stopped on its own sessions, thresholded on the fit sessions only
res = cv.cross_validate("stub", runs)
for f, held, (fit, val) in zip(res.folds, fs, Stub.seen):
    mine = {int(r.X[0, 0, 0]) for r in held}
    assert val == mine and not fit & mine and fit | mine == {int(r.X[0, 0, 0]) for r in sessions}
    assert (f.thr == max(fit)).all()
    assert not {r.id for r in f.held} & set(f.train)
assert all(r.role in ("train", "val") for f in res.folds for r in f.held)

# --- pooled: every healthy step held out once (summed), the dev fault steps once (averaged over the folds); the runs
#     passed in are not touched
p = res.pooled.loc[cv.VARIANT]
assert p.FP + p.TN == sum(steps) and p.TP + p.FN == sum(int(r.fault.sum()) for r in dev)
assert not hasattr(sessions[0], "alarm")
# a held-out session numbered above all the fit ones is a false alarm: here the highest numbered ones
assert p.FP > 0 and res.table.groupby("fold").FP.first().gt(0).sum() >= 1
assert p["faults found %"] == 100  # the dev runs (30+) are above every threshold

# --- the search space: lists are choices, (lo, hi) ranges, (lo, hi, "log") log ranges, the rest fixed
space = {"VAR": [0.9, 0.95], "SD_FLOOR": (0.0, 0.3), "K": (2, 8), "LR": (1e-4, 1e-2, "log"), "MIN_SD": 1e-6}
t = optuna.trial.FixedTrial({"VAR": 0.95, "SD_FLOOR": 0.1, "K": 4, "LR": 1e-3})
assert cv.suggest(t, space) == {"VAR": 0.95, "SD_FLOOR": 0.1, "K": 4, "LR": 1e-3, "MIN_SD": 1e-6}
assert cv.in_space(0.95, [0.9, 0.95]) and cv.in_space(0.1, (0.0, 0.3)) and not cv.in_space(0.5, (0.0, 0.3))
print("ok")
