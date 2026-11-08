"""Helpers of the exploration notebooks: find runs, load them through the builder, scale robustly, cut windows around events.

Everything reads the raw output of collect_dataset.py and never writes to it.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from anomaly import build

DATAGRAPH_DIR = Path(__file__).resolve().parents[1] / "service-graph"
FAULT_KINDS = ("test", "dev")
# Smallest change worth a unit of z, per feature in its own unit. A metric that is flat in the reference (throttling, restarts,
# an idle cache) has IQR 0: without a floor any blip would score millions. Calibration knobs, not measured constants.
MIN_SCALE = {"cpu": 0.002, "cpu_user": 0.002, "cpu_system": 0.002, "cpu_throttled": 0.002,  # cores, 2 mC
             "mem_working_set": 2**20, "mem_rss": 2**20, "mem_cache": 2**20,  # 1 MiB
             "threads": 1, "rx_bytes": 200, "tx_bytes": 200, "rx_packets": 2, "tx_packets": 2,  # per second
             "restarts": 1 / 300}  # one restart in 5 minutes
FLOOR = np.array([MIN_SCALE[name] for name, *_ in build.FEATURES])


def find_runs(root):
    """Last valid attempt of every run under `root`, as {run id: attempt dir}; the id is the run's path below `root`."""
    root, runs = Path(root), {}
    for run_dir in sorted({meta.parent.parent for meta in root.glob("**/attempt-*/meta.json")}):
        attempt = build.last_valid_attempt(run_dir)
        if attempt is not None:
            runs[str(run_dir.relative_to(root))] = attempt
    return runs


@lru_cache(maxsize=None)
def load(attempt, step=30):
    """One run through the builder: x [T, S, F], mask, t, rate, mix, services, features, labels, t_rel (s from the injection)."""
    arrays, labels = build.build_run(attempt, DATAGRAPH_DIR, step)
    run = SimpleNamespace(**arrays, labels=labels, attempt=Path(attempt))
    run.services, run.features = list(run.services), list(run.features)
    run.t_rel = run.t - labels["t_inject"]
    run.kind, run.id = labels["kind"], labels["id"]
    return run


def series(run, service, feature):
    return run.x[:, run.services.index(service), run.features.index(feature)]


def settle_mask(run, skip_s=60):
    """Grid steps of the settle, under load and before the injection; the first `skip_s` (at most half the settle) let wrk2 calibrate."""
    t_workload = run.labels.get("t_workload") or run.labels["t_inject"] - run.labels["settle_s"]
    skip_s = min(skip_s, (run.labels["t_inject"] - t_workload) / 2)
    return (run.t > t_workload + skip_s) & (run.t <= run.labels["t_inject"])


def robust_z(x, ref, floor=FLOOR, rel_floor=0.01):
    """(x - median) / IQR along time, both taken on the rows where `ref` is True; the last axis of x is the feature.

    The IQR is floored at `floor` (per feature, MIN_SCALE) and at `rel_floor` of the median: a flat metric must not explode.
    """
    if not np.any(ref):
        raise ValueError("robust_z: the reference has no rows")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices: services or features absent in this app
        med = np.nanmedian(x[ref], axis=0)
        iqr = np.nanpercentile(x[ref], 75, axis=0) - np.nanpercentile(x[ref], 25, axis=0)
    scale = np.maximum(np.maximum(iqr, rel_floor * np.abs(med)), floor)
    return (x - med) / scale


def windows(run, times, before_s, after_s):
    """Values around each event time: [events, steps, S, F], NaN where the run does not cover the window."""
    step = run.labels["step"]
    offsets = np.arange(-(before_s // step), after_s // step + 1)
    out = np.full((len(times), len(offsets), *run.x.shape[1:]), np.nan)
    for e, time in enumerate(times):
        idx = np.searchsorted(run.t, time) + offsets
        ok = (idx >= 0) & (idx < len(run.t))
        out[e, ok] = run.x[idx[ok]]
    return out, offsets * step


def steady_mask(run, skip_s=60):
    """Grid steps under a known load that has not changed for at least `skip_s`: away from the transients."""
    changed = np.r_[True, (np.diff(run.rate) != 0) | (np.diff(run.mix) != 0)]
    last_change = np.maximum.accumulate(np.where(changed, run.t, -np.inf))
    return (run.rate >= 0) & (run.t - last_change >= skip_s)


def by_level(run, feature, steps=None):
    """Median of `feature` per (rate, mix) on the given steps (default: steady ones): DataFrame, rows (rate, mix), columns services."""
    steps = steady_mask(run) if steps is None else steps
    values = run.x[:, :, run.features.index(feature)]
    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for level in sorted({(int(r), int(m)) for r, m in zip(run.rate[steps], run.mix[steps])}):
            on = steps & (run.rate == level[0]) & (run.mix == level[1])
            rows[level] = np.nanmedian(values[on], axis=0)
    index = pd.MultiIndex.from_tuples(list(rows), names=["rate", "mix"])
    return pd.DataFrame(np.array(list(rows.values())).reshape(len(rows), len(run.services)), index=index, columns=run.services)


def load_changes(run):
    """Grid times where the wrk2 rate or mix changes, with (rate before, rate after)."""
    i = np.where((np.diff(run.rate) != 0) | (np.diff(run.mix) != 0))[0] + 1
    i = i[(run.rate[i - 1] >= 0) & (run.rate[i] >= 0)]
    return run.t[i], list(zip(run.rate[i - 1], run.rate[i]))


def gap_to_injection(run):
    """Seconds between the injection and the nearest load change (segment boundary) of the run."""
    ends = [s["start"] + s["seconds"] for s in run.labels["segments"]] + [s["start"] for s in run.labels["segments"]]
    return min((abs(e - run.labels["t_inject"]) for e in ends), default=np.inf)


def target_services(run):
    """Datagraph services that are the fault target (the scenarios say "user" for user-service)."""
    targets = [t for t in (run.labels.get("accepted_targets") or [run.labels.get("target")]) if t]
    return [s for s in run.services if any(s in (t, f"{t}-service") for t in targets)]


def score_services(run, after_s=300, ref=None):
    """RobustScorer-like ranking: per service, the largest |robust z| over features in the `after_s` after the injection.

    The reference is the settle. Returns [(service, score, feature)] from the highest score.
    """
    ref = settle_mask(run) if ref is None else ref
    z = np.abs(robust_z(run.x, ref))
    after = (run.t_rel > 0) & (run.t_rel <= after_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(z[after], axis=0)  # [S, F]
    peak = np.where(np.isnan(peak), -np.inf, peak)
    best = peak.argmax(axis=1)
    scored = [(s, float(peak[i, best[i]]), run.features[best[i]]) for i, s in enumerate(run.services)]
    return sorted(scored, key=lambda r: -r[1])
