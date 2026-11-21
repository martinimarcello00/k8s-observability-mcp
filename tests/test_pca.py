# Run from the repo root: PYTHONPATH=. python tests/test_pca.py  (no cluster, no data needed)
import numpy as np

from anomaly import build, inputs
from anomaly.models import pca
from anomaly.models.evaluate import episodes, rank

rng = np.random.default_rng(0)
F = len(pca.FEATURES)

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
m = pca.PCA().fit([X]).calibrate([(X, np.ones(T, bool))])
s = m.services[0]
assert s.V.shape[1] == 2 and not s.cols[F - 1]  # two directions kept, the flat column left out

# --- a healthy step scores about 0 and is reconstructed; a step off the subspace scores on the feature that moved
y = X[:6].copy()
y[3:5, 0, 4] += 10 * X[:, 0, 4].std()
r2, expected = m.score(y)
assert r2[0, 0].sum() < 0.01 and np.allclose(expected[0, 0, :F - 1], y[0, 0, :F - 1], atol=0.05)
assert np.isnan(expected[0, 0, F - 1])
assert np.argmax(r2[3, 0]) == 4 and r2[3, 0].sum() > 50
assert pca.explain(y, r2, expected, 3, 0)[0][0] == pca.FEATURES[4]

# --- persistence: the first step off the subspace alone does not alarm, the second one does
e = m.spe(y)[:, 0]
assert e[3] < m.thr[0] < e[4]

# --- ranking: a restarted service first, then by score; episodes of consecutive alarms
assert list(rank(np.array([0.5, 3.0, 2.0]), np.array([False, False, True]))) == [2, 1, 0]
assert episodes(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
print("ok")
