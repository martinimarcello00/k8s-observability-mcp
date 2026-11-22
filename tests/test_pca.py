# Run from the repo root: PYTHONPATH=. python tests/test_pca.py  (no cluster, no data needed)
import numpy as np

from anomaly import build, inputs
from anomaly.models import evaluate as ev, pca

rng = np.random.default_rng(0)
F = len(pca.PCA.features)

# --- prepare: values in UNITS through log1p, memory as its change per step (NaN at the first step); to_units inverts it
names = [f[0] for f in build.FEATURES]
x = np.zeros((3, 1, len(names)))
x[:, 0, names.index("cpu_user")] = 0.002  # cores
x[:, 0, names.index("mem_working_set")] = np.array([100, 101, 103]) * 2**20  # bytes
o = inputs.prepare(x, names)
cu, ws = inputs.B0.index("cpu_user"), inputs.B0.index("mem_working_set")
assert o.shape == (3, 1, len(inputs.B0)) and np.isnan(o[0, 0, ws])
assert np.allclose(inputs.to_units(o[1:, 0, ws], "mem_working_set"), [1, 2])
assert np.allclose(inputs.to_units(o[:, 0, cu], "cpu_user"), 2)  # mC

# --- one service whose features follow 2 hidden factors, plus a column flat in training
T = 2000
X = (rng.normal(size=(T, 2)) @ rng.normal(size=(2, F)) + 0.01 * rng.normal(size=(T, F)))[:, None, :]
X[:, :, F - 1] = 0.5
m = pca.PCA().fit([X])
s = m.services[0]
assert s.V.shape[1] == 2 and not s.cols[F - 1]  # two directions kept, the flat column left out
assert m.card(["svc"])["svc"]["left_out"] == [m.features[F - 1]]

# --- a healthy step scores about 0 and is reconstructed; a step off the subspace scores on the feature that moved
y = X[:6].copy()
y[3:5, 0, 4] += 10 * X[:, 0, 4].std()
r2, expected = m.score(y)
assert r2[0, 0].sum() < 0.01 and np.allclose(expected[0, 0, :F - 1], y[0, 0, :F - 1], atol=0.05)
assert np.isnan(expected[0, 0, F - 1])
assert np.argmax(r2[3, 0]) == 4 and r2[3, 0].sum() > 50
assert ev.explain(m.features, y, r2, expected, 3, 0)[0][0] == m.features[4]

# --- persistence: the first step off the subspace alone does not pass the healthy maximum, the second one does
thr = ev.persist(m.score(X)[0].sum(2)).max(0)[0]
e = ev.persist(r2.sum(2))[:, 0]
assert e[3] < thr < e[4]

# --- protocol helpers: alarm starts, episodes, ranking (restarted first, then score), confusion matrix, F-beta
assert list(ev.starts(np.array([0, 1, 1, 0, 1], bool))) == [False, True, False, False, True]
assert ev.episodes(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
assert list(ev.rank(np.array([0.5, 3.0, 2.0]), np.array([False, False, True]))) == [2, 1, 0]
alarm, fault = np.array([1, 1, 0, 0, 1], bool), np.array([0, 0, 1, 1, 1], bool)
assert ev.counts(alarm, fault, ~fault) == (1, 2, 2, 0)
assert np.isnan(ev.precision_recall(3, 1, 0, 0)[0])  # no healthy inference: precision undefined, not 1
assert ev.precision_recall(3, 1, 1, 5) == (0.75, 0.75)

# --- absence rule: a container group gone for 3 steps is absent from its 2nd step (the 1st is filled as a miss);
#     a one-step miss is not absence
x = np.ones((8, 2, len(names)))
x[:, :, names.index("restarts")] = 0
exists = ~np.isnan(x).all(0)
C = [j for j, f in enumerate(build.FEATURES) if f[3] == "container"]
y = x.copy()
y[3:6, 0, C] = np.nan
a = ev.absent(y, names, exists)
assert list(a[:, 0]) == [False, False, False, False, True, True, False, False] and not a[:, 1].any()
y = x.copy()
y[3, 0, C] = np.nan
assert not ev.absent(y, names, exists).any()
assert np.isclose(ev.fbeta(0.5, 1.0, 2), 5 / 6) and np.isclose(ev.fbeta(1.0, 0.5, 0.5), 5 / 6)
assert np.isclose(ev.fbeta(0.5, 1.0, 1), 2 / 3) and ev.fbeta(0, 0, 1) == 0
print("ok")
