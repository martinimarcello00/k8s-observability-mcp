# Run from the repo root: PYTHONPATH=. python tests/test_inputs.py  (no cluster, no data needed)
import numpy as np

from anomaly import build
from anomaly.inputs import fill_misses

F = [f[0] for f in build.FEATURES]
C = [j for j, f in enumerate(build.FEATURES) if f[3] == "container"]  # container features
N = [j for j, f in enumerate(build.FEATURES) if f[3] == "pod"]  # network features
R = F.index("restarts")


def run(T=8, S=2):
    # every cell present; the value of a cell is its step index; no restart
    x = np.arange(T, dtype=float)[:, None, None] + np.zeros((T, S, len(F)))
    x[:, :, R] = 0
    return x


# --- a container sample cAdvisor skipped: filled with the previous value; the input is not modified
x = run()
x[3, 0, C] = np.nan
y, p = fill_misses(x, F)
assert np.all(y[3, 0, C] == 2) and p[3, 0, 0] and np.isnan(x[3, 0, C]).all()
x = run()
x[3, 1, N] = np.nan  # the network series is a group of its own
y, p = fill_misses(x, F)
assert np.all(y[3, 1, N] == 2) and p[3, 1, 1]

# --- a pod gone for three steps: its first step is filled (online it cannot be told yet), then it is absent
x = run()
x[3:6, 0, C] = np.nan
y, p = fill_misses(x, F)
assert np.all(y[3, 0, C] == 2) and np.isnan(y[4:6, 0, C]).all()
assert list(p[3:7, 0, 0]) == [True, False, False, True]

# --- part of a group missing (the rates right after a restart): untouched, the container is there
x = run()
x[3, 0, [F.index("cpu"), F.index("cpu_user")]] = np.nan
y, p = fill_misses(x, F)
assert np.isnan(y[3, 0, F.index("cpu")]) and p[3, 0, 0]

# --- a restart at t or at t-1: not filled
x = run()
x[3, 0, C] = np.nan
x[3, 0, R] = 1 / 30
y, p = fill_misses(x, F)
assert np.isnan(y[3, 0, C]).all() and not p[3, 0, 0]
x = run()
x[4, 0, C] = np.nan
x[3, 0, R] = 1 / 30
assert np.isnan(fill_misses(x, F)[0][4, 0, C]).all()

# --- container and network missing together: the pod is gone, not a skipped sample
x = run()
x[3, 0, C + N] = np.nan
y, p = fill_misses(x, F)
assert np.isnan(y[3, 0, C + N]).all() and not p[3, 0].any()

# --- the first step has nothing before it
x = run()
x[0, 0, C] = np.nan
assert np.isnan(fill_misses(x, F)[0][0, 0, C]).all()

# --- a cell that never exists (throttling without a CPU limit) stays NaN and does not hide a skipped sample;
#     a group that does not exist at all counts as present
x = run()
x[:, 0, F.index("cpu_throttled")] = np.nan
x[3, 0, C] = np.nan
x[:, 1, N] = np.nan
y, p = fill_misses(x, F)
assert np.isnan(y[:, 0, F.index("cpu_throttled")]).all() and p[:, 0, 0].all() and p[:, 1, 1].all()
assert np.all(y[3, 0, [j for j in C if F[j] != "cpu_throttled"]] == 2)

# --- causal: step t computed on the steps up to t (online) equals step t computed on the whole run (offline)
rng = np.random.default_rng(0)
x = run(T=60, S=4)
for t, s in zip(*np.nonzero(rng.random(x.shape[:2]) < .2)):
    x[t, s, C] = np.nan
for t, s in zip(*np.nonzero(rng.random(x.shape[:2]) < .15)):
    x[t, s, N] = np.nan
for t, s in zip(*np.nonzero(rng.random(x.shape[:2]) < .05)):
    x[t, s, rng.choice(C)] = np.nan
for t, s in zip(*np.nonzero(rng.random(x.shape[:2]) < .05)):
    x[t, s, R] = 1 / 30
exists = ~np.isnan(x).all(axis=0)
y, p = fill_misses(x, F, exists)
for t in range(len(x)):
    yt, pt = fill_misses(x[:t + 1], F, exists)
    assert np.array_equal(yt[t], y[t], equal_nan=True) and np.array_equal(pt[t], p[t])

print("ok")
