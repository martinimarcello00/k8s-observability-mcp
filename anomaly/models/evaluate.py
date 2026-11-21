"""Evaluate the PCA baseline on one app of the built dataset, and write down what it did.

    PYTHONPATH=. python -m anomaly.models.evaluate hotel [--data ~/datasets/full_v2-built] [--out results]

Roles come from the labels. A clean session carries its split (train, val, test, test-unseen); a run of the fault
folders is noop (No fault) or a fault (fault-dev, fault-test). Fit on train, threshold on train + val: the protocol of
the analyses of 2026-11-19/20. Cross-validation comes next.

Detection: a service is in alarm when its persistent SPE is over its threshold, or, with the restart rule, when it
restarts. Localization ranks the services: restarted first (rule only), then by SPE / threshold.
- False alarms: share of loaded steps (rate >= 0) with any service in alarm, per healthy split.
- Per fault run, in the WINDOW s after the injection: an alarm on any service; the target in alarm; minutes to the
  first alarm; the target first in the ranking at the first alarm; first, and in the top 3, by each service's largest
  score in the window. Localization counts only for runs with an alarm.

Written to OUT/pca-B0/<app>/, overwritten at each run (git keeps the history):
- config.json: data, code commit, parameters, the role of every run;
- model.json: per service, k, variance kept, threshold, columns left out;
- false_alarms.csv, faults.csv (one row per fault run, with the top 3 at the first alarm), scenarios.csv (per scenario);
- alarms.csv: one row per alarm episode (a service in alarm for consecutive steps), with the features behind its peak,
  observed and expected. Its unix times and services are the keys of the MCP tools (get_metrics_range, get_traces,
  get_logs) and of anomaly/traces.py, to go from an alarm to the raw metrics, traces and logs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from anomaly import inputs
from anomaly.models import pca

WINDOW = 900  # s after the injection where a fault must be found
REPO = Path(__file__).resolve().parents[2]
HEALTHY = ("test", "test-unseen", "noop")


def load(app, data):
    runs = []
    for p in sorted(Path(data, app).glob("*/*.npz")):
        z, L = np.load(p), json.loads(p.with_suffix(".json").read_text())
        folder = p.parent.name
        role = L["split"] if folder == "clean" else "noop" if L["fault_type"] == "No fault" else f"fault-{folder}"
        S = [str(s) for s in z["services"]]
        names = [n for n in (L.get("accepted_targets") or [L.get("target")]) if n]
        ti = L.get("t_inject") if folder != "clean" else None
        runs.append(SimpleNamespace(
            id=L["id"], role=role, labels=L, t=z["t"], on=z["rate"] >= 0, services=S, t_inject=ti,
            X=inputs.prepare(z["x"], [str(f) for f in z["features"]]),
            after=(z["t"] > ti) & (z["t"] <= ti + WINDOW) if ti else np.zeros(len(z["t"]), bool),
            targets=[i for i, s in enumerate(S) if any(s in (n, f"{n}-service") for n in names)],
            scenario=L["id"].split("/")[-1].split("__")[0].split("_hotel")[0].split("_social")[0]))
    return runs


def rank(score, restarted):
    """Service indices, restarted first, then by score, largest first."""
    return np.lexsort((-score, -restarted.astype(int)))


def episodes(a):
    """[(start, stop)] of the runs of True in a 1-D bool array, stop excluded."""
    d = np.diff(np.r_[0, a.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fault_row(r):
    row = {"run": r.id, "fold": r.role[6:], "scenario": r.scenario, "targets": " ".join(r.services[i] for i in r.targets)}
    for tag, rule in (("pca", np.zeros_like(r.restart)), ("pca+restart", r.restart)):
        n, rs = r.norm[r.after], rule[r.after]
        alarm = (n > 1) | rs
        hit = alarm.any(1)
        row[f"{tag} alarm"] = int(hit.any())
        row[f"{tag} target in alarm"] = int(alarm[:, r.targets].any())
        k = hit.argmax()
        first, window = rank(n[k], rs[k]), rank(n.max(0), rs.any(0))
        row[f"{tag} minutes"] = (r.t[r.after][k] - r.t_inject) / 60 if hit.any() else np.nan
        row[f"{tag} top1 first"] = int(hit.any() and first[0] in r.targets)
        row[f"{tag} top1 window"] = int(hit.any() and window[0] in r.targets)
        row[f"{tag} top3 window"] = int(hit.any() and bool(set(window[:3]) & set(r.targets)))
        if tag == "pca+restart" and hit.any():
            row["top3 at first alarm"] = "; ".join(
                f"{r.services[i]} {n[k, i]:.1f}" + (" restart" if rs[k, i] else "") for i in first[:3])
    return row


def alarm_rows(r):
    rows, steps = [], r.on | r.after
    for i, s in enumerate(r.services):
        for a, b in episodes(((r.norm[:, i] > 1) | r.restart[:, i]) & steps):
            p = a + int(np.argmax(r.norm[a:b, i]))
            rows.append({
                "run": r.id, "role": r.role, "scenario": r.scenario, "service": s, "is_target": int(i in r.targets),
                "start": int(r.t[a]), "start_utc": datetime.fromtimestamp(r.t[a], timezone.utc).isoformat(timespec="seconds"),
                "minutes_from_injection": round((r.t[a] - r.t_inject) / 60, 1) if r.t_inject else None,
                "minutes": (b - a) * 0.5, "peak": round(float(r.norm[p, i]), 2), "restart": int(r.restart[a:b, i].any()),
                "features at peak": "; ".join(f"{f} {sh:.0%} ({o:.4g} vs {e:.4g} {inputs.UNITS[f][1]})"
                                              for f, sh, o, e in pca.explain(r.P, r.r2, r.expected, p, i))})
    return rows


def git_state():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "anomaly", ":!anomaly/notebooks"],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()  # the code only, not results/
    return {"commit": head, "uncommitted_changes": bool(dirty)}


def evaluate(app, data, out):
    runs = load(app, data)
    P, R = [inputs.B0.index(f) for f in pca.FEATURES], inputs.B0.index("restarts")
    by = lambda *roles: [r for r in runs if r.role in roles]
    model = pca.PCA().fit([r.X[r.on][:, :, P] for r in by("train")])
    model.calibrate([(r.X[:, :, P], r.on) for r in by("train", "val")])
    for r in runs:
        r.P = r.X[:, :, P]
        r.r2, r.expected = model.score(r.P)
        r.norm = pca.persist(r.r2.sum(2)) / model.thr
        r.restart = np.nan_to_num(r.X[:, :, R]) > 0

    fa = pd.DataFrame([{"split": role, "runs": len(by(role)),
                        "pca": np.mean([(r.norm[r.on] > 1).any(1).mean() for r in by(role)]) * 100,
                        "pca+restart": np.mean([((r.norm > 1) | r.restart)[r.on].any(1).mean() for r in by(role)]) * 100}
                       for role in HEALTHY]).set_index("split").round(1)
    faults = pd.DataFrame([fault_row(r) for r in by("fault-dev", "fault-test") if r.targets])
    minutes = [c for c in faults if c.endswith("minutes")]
    shares = [c for c in faults if c.startswith("pca") and c not in minutes]
    g = faults.groupby(["fold", "scenario"])
    scen = pd.concat([g.size().rename("runs"), (g[shares].mean() * 100).round().astype(int), g[minutes].median().round(1)], axis=1)
    alarms = pd.DataFrame([row for r in runs if r.role not in ("train", "val") for row in alarm_rows(r)])

    d = Path(out, "pca-B0", app)
    d.mkdir(parents=True, exist_ok=True)
    config = {"app": app, "data": str(data), "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "code": git_state(), "model": "PCA per service, SPE persistent over 2 steps", "features": pca.FEATURES,
              "rule": "restarts > 0 -> alarm on that service", "VAR": pca.VAR, "MIN_FINITE": pca.MIN_FINITE,
              "MIN_SD": pca.MIN_SD, "threshold": "per service, max persistent SPE over train + val (loaded steps)",
              "window_s": WINDOW, "runs": {role: [r.id for r in by(role)] for role in sorted({r.role for r in runs})}}
    (d / "config.json").write_text(json.dumps(config, indent=1))
    (d / "model.json").write_text(json.dumps(model.card(runs[0].services), indent=1))
    fa.to_csv(d / "false_alarms.csv")
    faults.to_csv(d / "faults.csv", index=False)
    scen.to_csv(d / "scenarios.csv")
    alarms.to_csv(d / "alarms.csv", index=False)
    return fa, scen, faults, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("app", choices=("hotel", "social"))
    ap.add_argument("--data", default=Path.home() / "datasets/full_v2-built")
    ap.add_argument("--out", default=REPO / "results")
    a = ap.parse_args()
    fa, scen, faults, d = evaluate(a.app, a.data, a.out)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"{a.app}: false alarms, % of loaded steps\n{fa}\n\nper scenario (% runs; minutes = median)\n{scen}")
    print("\ntarget in alarm, fault runs dev + test:", f"{faults['pca target in alarm'].mean():.0%} (pca),",
          f"{faults['pca+restart target in alarm'].mean():.0%} (pca+restart)\nwritten to {d}")
