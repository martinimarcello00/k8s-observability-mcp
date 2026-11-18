"""What the model sees, made from the arrays of anomaly/build.py. The same code serves training and the online detector.

fill_misses tells a sample that cAdvisor skipped from a container that is gone. cAdvisor samples every 10-43 s, so a
30 s grid step sometimes gets no sample. When that happens, a whole group of one service is empty for one step: either
its container features or its network features, which come from a separate series. A pod that dies, never starts or
restarts leaves longer or partial holes instead, and those holes are the fault's signal.

The rule is causal: step t depends only on steps t-1 and t. Run on a whole run, it gives step by step what the online
detector computed when each step arrived.
"""
import numpy as np

from anomaly import build

# The two cAdvisor series of a service. "kube" (restarts) comes from kube-state-metrics and is never filled.
GROUPS = ("container", "pod")


def fill_misses(x, features, exists=None):
    """Fill the samples cAdvisor skipped. Returns (x_filled, present).

    x [T, S, F] holds the raw values of one run or window (NaN = no sample); `features` names its F axis.
    exists [S, F] bool marks the cells that exist at all. It defaults to the cells present at least once in x;
    online, pass the training's.

    A group of a service (its container features, or its network ones) is filled at step t with its value at t-1
    when all of these hold:
    - the whole group is missing at t;
    - the whole group was there at t-1;
    - the other group of the service is there at t;
    - the service did not restart at t-1 or t.
    Everything else stays NaN, so a gap longer than one step is filled at its first step only.

    present [T, S, 2] bool, one flag per group (container, network): True where the group has data after filling,
    or where the group does not exist at all.
    """
    level = {name: lvl for name, _, _, lvl in build.FEATURES}
    exists = ~np.isnan(x).all(axis=0) if exists is None else np.asarray(exists, bool)
    cols = {g: [j for j, f in enumerate(features) if level[f] == g] for g in GROUPS}
    restarts = np.nan_to_num(x[:, :, features.index("restarts")]) if "restarts" in features else np.zeros(x.shape[:2])
    restarted = (restarts > 0) | (np.vstack([np.zeros((1, x.shape[1])), restarts[:-1]]) > 0)

    def whole(cells, g):
        # [T, S]: every existing cell of group g satisfies `cells`; False for a service where the group does not exist
        e = exists[:, cols[g]]
        return np.where(e[None], cells[:, :, cols[g]], True).all(axis=2) & e.any(axis=1)[None]

    have = ~np.isnan(x)
    prev = np.concatenate([np.full_like(x[:1], np.nan), x[:-1]])
    out, present = x.copy(), np.ones(x.shape[:2] + (len(GROUPS),), bool)
    for k, (g, other) in enumerate(zip(GROUPS, GROUPS[::-1])):
        miss = whole(~have, g) & ~whole(~have, other) & ~restarted
        miss[1:] &= whole(have, g)[:-1]
        miss[0] = False  # no step before the first one
        c = cols[g]
        out[:, :, c] = np.where(miss[:, :, None] & exists[None][:, :, c], prev[:, :, c], out[:, :, c])
        present[:, :, k] = ~whole(np.isnan(out), g)
    return out, present
