# Run from the repo root: PYTHONPATH=. python tests/test_explore.py  (no cluster, no data needed)
from types import SimpleNamespace as NS

import numpy as np

from anomaly import explore as e

# --- robust z: median and IQR of the reference rows; a flat metric is floored instead of exploding
x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [10.0, 6.0]])[:, :, None]  # [T, S, F]
z = e.robust_z(x, np.array([True, True, True, True, False]), floor=1e-9)
assert np.isclose(z[4, 0, 0], (10 - 2.5) / 1.5)  # IQR of 1..4 is 1.5
assert np.isclose(z[4, 1, 0], (6 - 5) / 0.05)  # IQR 0: floored at 1% of the median
flat = np.zeros((5, 1, len(e.FLOOR)))
flat[4, 0, -1] = 1 / 60  # one restart in a minute, after a reference without any
assert np.isclose(e.robust_z(flat, np.arange(5) < 4)[4, 0, -1], 5.0)  # in units of one restart per 5 minutes, not millions

# --- windows around events: NaN outside the run
run = NS(t=np.arange(0.0, 300.0, 30), x=np.arange(10.0)[:, None, None], labels={"step": 30})
w, offsets = e.windows(run, [0.0, 270.0], 60, 30)
assert list(offsets) == [-60, -30, 0, 30]
assert np.allclose(w[0, :, 0, 0], [np.nan, np.nan, 0, 1], equal_nan=True) and np.allclose(w[1, :, 0, 0], [7, 8, 9, np.nan], equal_nan=True)

# --- load changes and the gap between the injection and the nearest one
run = NS(t=np.arange(6.0) * 30, rate=np.array([-1, 10, 10, 30, 30, 30]), mix=np.array([-1, 0, 0, 0, 2, 2]),
         labels={"t_inject": 100.0, "segments": [{"start": 20, "seconds": 70}, {"start": 90, "seconds": 300}]})
times, steps = e.load_changes(run)
assert list(times) == [90.0, 120.0] and steps == [(10, 30), (30, 30)]  # the second is a change of mix
assert e.gap_to_injection(run) == 10
assert list(e.steady_mask(run, skip_s=30)) == [False, False, True, False, False, True]  # after each change, wait 30 s
run.x, run.features, run.services = np.arange(6.0)[:, None, None], ["cpu"], ["geo"]
levels = e.by_level(run, "cpu", steps=run.rate >= 0)
assert levels.loc[(10, 0), "geo"] == 1.5 and levels.loc[(30, 2), "geo"] == 4.5 and len(levels) == 3

# --- fault target: "user" names user-service, not user-mention-service
run = NS(services=["user-service", "user-mention-service", "user-mongodb"], labels={"accepted_targets": None, "target": "user"})
assert e.target_services(run) == ["user-service"]

print("ok")
