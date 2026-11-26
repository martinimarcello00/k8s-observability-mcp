# Run from the repo root: PYTHONPATH=. python tests/test_orcrist.py  (no cluster, no data needed; about a minute)
from types import SimpleNamespace

import numpy as np

from anomaly.models import evaluate as ev, orcrist
from anomaly.models.orcrist import model as om

rng = np.random.default_rng(0)
F = len(om.Orcrist.features)
names = ["frontend", "mongodb-geo", "memcached-rate"]
assert [om.service_type(s) for s in names] == ["service", "mongodb", "memcached"]


W, level = rng.normal(size=(2, F)), np.array([4.0, 1.0, 2.0])


def runs(T):
    """3 services whose features follow 2 shared factors (the load), each with its own level and scale; the last
    feature of mongodb-geo is flat."""
    f = rng.normal(size=(T, 2))
    X = np.stack([level[i] + (i + 1) * 0.3 * f @ W + 0.05 * rng.normal(size=(T, F)) for i in range(3)], 1)
    X[:, 1, F - 1] = 0.5
    return X


X, V = runs(3000), runs(600)
sizes = {}
for version in orcrist.VERSIONS:
    m = om.Orcrist(version, MAX_EPOCHS=120).fit([X], [V], names)
    assert not m.cols[1, F - 1] and m.cols[0].all()  # the flat cell is left out
    assert m.card(names)["mongodb-geo"]["left_out"] == [m.features[F - 1]]

    # a healthy step scores low and is reconstructed; a feature pushed up scores high, on that feature, deterministically
    y = V[:6].copy()
    y[3:5, 0, 4] += 6 * V[:, 0, 4].std()
    r2, expected = m.score(y)
    healthy = m.score(V)[0][:, 0].sum(1).max()
    assert r2[3, 0].sum() > 3 * healthy, (version, r2[3, 0].sum(), healthy)
    assert np.argmax(r2[3, 0]) == 4 and np.array_equal(r2, m.score(y)[0])
    assert np.isnan(expected[0, 1, F - 1]) and r2[:, 1, F - 1].sum() == 0
    sd, kl = m.spread(y)
    assert (sd[:, 0] > 0).all() and np.isnan(sd[0, 1, F - 1]) and (kl >= 0).all() and kl.shape == (6, 3)

    # the localization names the feature, its direction and a normal range that holds the expected value
    r = SimpleNamespace(P=y, r2=r2, expected=expected, sd=sd)
    top = ev.why(r, 3, 0, list(m.features))[0]
    assert top["feature"] == m.features[4] and top["direction"] == "up"
    assert top["normal"][0] <= top["expected"] <= top["normal"][1] < top["observed"]
    sizes[version] = m.params["parameters"]
    print(version, m.params["best_epoch"], m.params["parameters"], round(float(r2[3, 0].sum() / healthy), 1))

# --- every parameter is per instance: other sizes, depth, KL weight and weight decay train; the history has one row per
#     epoch and the callback sees each; an unknown parameter is refused
rows = []
m = om.Orcrist("AC", MAX_EPOCHS=30, K=2, H=16, LAYERS=3, BETA=0.5, WEIGHT_DECAY=1e-4, KL_WARMUP=5, PATIENCE=5)
m.on_epoch = rows.append
m.fit([X], [V], names)
assert len(m.history) == len(rows) == m.params["epochs"] and rows[-1]["beta"] == 0.5
assert m.latent(V[:3])[0].shape == (3, 3, 2) and m.params["parameters"] != sizes["AC"]
try:
    om.Orcrist("AA", LAYER=3)
    raise AssertionError("an unknown parameter was accepted")
except ValueError:
    pass
print("ok")
