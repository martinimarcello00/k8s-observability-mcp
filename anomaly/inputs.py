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


# B0, the feature set of the first models (2026-11-20): the 13 built features but cpu (exactly cpu_user + cpu_system) and
# cpu_throttled (0 in every healthy Hotel run, absent in Social). mem_cache and mem_rss stay for now, to be judged later.
B0 = ("cpu_user", "cpu_system", "mem_working_set", "mem_rss", "mem_cache", "threads",
      "rx_bytes", "tx_bytes", "rx_packets", "tx_packets", "restarts")
# Each feature is read in this unit before log1p, so that log1p is about linear up to 1 unit and logarithmic above it.
# Memory is a change per 30 s step (see prepare). Restarts: the rate per second times 300, restarts per 5 minutes.
UNITS = {"cpu": (1e3, "mC"), "cpu_user": (1e3, "mC"), "cpu_system": (1e3, "mC"), "cpu_throttled": (1e3, "mC"),
         "mem_working_set": (2**-20, "MiB/step"), "mem_rss": (2**-20, "MiB/step"), "mem_cache": (2**-20, "MiB/step"),
         "threads": (1, "threads"), "rx_bytes": (1, "B/s"), "tx_bytes": (1, "B/s"), "rx_packets": (1, "pkt/s"),
         "tx_packets": (1, "pkt/s"), "restarts": (300, "per 5 min")}


def prepare(x, features, keep=B0, exists=None):
    """Model input of one run, [T, S, len(keep)] from the raw x [T, S, F] of the builder; causal like fill_misses.

    Values are filled (fill_misses, with `exists` as there: online, the training's), clipped at 0, read in UNITS and
    passed through log1p. Memory is the signed log1p of its change since the previous step instead: the level of a
    store's memory is the integral of its past load, its change is not. The first step has no change, so its memory is NaN.
    """
    v = np.clip(fill_misses(x, features, exists)[0], 0, None)
    out = []
    for f in keep:
        a = v[:, :, features.index(f)] * UNITS[f][0]
        if f.startswith("mem"):
            a = np.diff(a, axis=0, prepend=np.nan)
            out.append(np.sign(a) * np.log1p(np.abs(a)))
        else:
            out.append(np.log1p(a))
    return np.stack(out, axis=2)


def to_units(o, feature):
    """Inverse of prepare for one feature: from model input back to UNITS (memory: its change per step)."""
    return np.sign(o) * np.expm1(np.abs(o)) if feature.startswith("mem") else np.expm1(o)
