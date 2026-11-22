"""Evaluate a model on one app of the built dataset with the protocol below, and write down what it did.

    PYTHONPATH=. python -m anomaly.models.evaluate pca hotel [--data ~/datasets/full_v2-built] [--out results]

The protocol is the same for every model: only fitting and scoring belong to the model. A model in MODELS provides
- features: the B0 features it reads (names, a subset of inputs.B0);
- fit(X): X is a list of [T, S, F] arrays from inputs.prepare on those features, healthy loaded steps only;
- score(X) -> (r2, expected): for one run [T, S, F], each feature's share of the step's score (the score is their sum
  over F) and the value the model expects for it, in prepare units (NaN where it has none);
- params (its settings) and card(services) (what it learned, per service).

Protocol
- Roles from the labels: a clean session carries its split (train, val, test, test-unseen); a run of the fault folders
  is noop (No fault) or a fault (fault-dev, fault-test). Fit on train.
- Score of a service at a step: the model's score, persistent over 2 steps (the smaller of t and t-1). Threshold per
  service: the largest score over the loaded steps of train + val.
- One inference per 30 s step. The system is in alarm when a service is over its threshold or, in the model+restart
  variant, restarts (restarts is 0 in every healthy run, so it is a rule, not a model input).
- Ground truth per inference. A fault run is faulty from its injection to its end: the collector observes 900 s, then
  exports and deletes the cluster without removing the fault. Healthy are the loaded steps of noop runs, of the clean
  test and test-unseen sessions and of fault runs before the injection. train and val are never scored as healthy:
  the threshold has seen them.
- Detection per inference: confusion matrix, precision, recall, F1 (the main score), F0.5 and F2; the share of healthy
  inferences in alarm per source; the recall per scenario.
- Per fault: found when an alarm of some service starts after the injection; minutes to that first alarm; localization
  by the services ranked at it, restarted first (model+restart), then by score / threshold. top@k: an accepted target
  of the scenario (any of them) is in the first k; a fault not found is a miss.

Written to OUT/<model>-B0/<app>/, overwritten at each run (git keeps the history):
- config.json: model and parameters, data, code commit, protocol, the role of every run;
- model.json: what the model learned per service, with its threshold;
- summary.csv: one row per variant, detection per inference and per fault, localization;
- scenarios.csv (per scenario) and faults.csv (per fault run, with the top 5 at the first alarm);
- alarms.csv: one row per alarm episode (a service in alarm for consecutive steps), with the features behind its peak,
  observed and expected, and whether it began during a fault. Its unix times and services are the keys of the MCP
  tools (get_metrics_range, get_traces, get_logs) and of anomaly/traces.py, to go from an alarm to the raw data.
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

MODELS = {"pca": pca.PCA}
REPO = Path(__file__).resolve().parents[2]
SOURCES = ("test", "test-unseen", "noop", "before injection")  # where the healthy inferences come from
TOP = (1, 3, 5)
VARIANTS = ("model", "model+restart")


def load(app, data):
    runs = []
    for p in sorted(Path(data, app).glob("*/*.npz")):
        z, L = np.load(p), json.loads(p.with_suffix(".json").read_text())
        folder, t, on = p.parent.name, z["t"], z["rate"] >= 0
        role = L["split"] if folder == "clean" else "noop" if L["fault_type"] == "No fault" else f"fault-{folder}"
        S = [str(s) for s in z["services"]]
        names = [n for n in (L.get("accepted_targets") or [L.get("target")]) if n]
        ti = None if folder == "clean" else L["t_inject"]
        fault = t > ti if role.startswith("fault") else np.zeros(len(t), bool)
        runs.append(SimpleNamespace(
            id=L["id"], role=role, t=t, on=on, services=S, t_inject=ti, fault=fault,
            healthy=np.zeros(len(t), bool) if role in ("train", "val") else on & ~fault,
            source="before injection" if role.startswith("fault") else role,
            X=inputs.prepare(z["x"], [str(f) for f in z["features"]]),
            targets=[i for i, s in enumerate(S) if any(s in (n, f"{n}-service") for n in names)],
            scenario="clean" if folder == "clean" else
            L["id"].split("/")[-1].split("__")[0].split("_hotel")[0].split("_social")[0]))
    return runs


def persist(e):
    """A score holds at t when it held at t-1 too: the smaller of the two, along axis 0 (0 at the first step)."""
    return np.minimum(e, np.concatenate([np.zeros_like(e[:1]), e[:-1]]))


def starts(a):
    """Where an alarm begins along axis 0: in alarm at t and not at t-1."""
    return a & ~np.concatenate([np.zeros_like(a[:1]), a[:-1]])


def episodes(a):
    """[(start, stop)] of the runs of True in a 1-D bool array, stop excluded."""
    d = np.diff(np.r_[0, a.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def rank(score, restarted):
    """Service indices, restarted first, then by score, largest first."""
    return np.lexsort((-score, -restarted.astype(int)))


def counts(alarm, fault, healthy):
    """Confusion matrix of inferences (tp, fn, fp, tn): fault inferences in alarm or not, healthy ones in alarm or not."""
    return int((alarm & fault).sum()), int((~alarm & fault).sum()), int((alarm & healthy).sum()), int((~alarm & healthy).sum())


def fbeta(p, r, beta):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0


def explain(features, X, r2, expected, t, i, top=3):
    """The features that make the score of service i at step t, largest first: [(feature, share, observed, expected)],
    values in inputs.UNITS."""
    c = r2[t, i]
    order = [j for j in np.argsort(-c)[:top] if c[j] > 0]
    return [(features[j], float(c[j] / c.sum()), float(inputs.to_units(X[t, i, j], features[j])),
             float(inputs.to_units(expected[t, i, j], features[j]))) for j in order]


def fault_row(r):
    row = {"run": r.id, "fold": r.role[6:], "scenario": r.scenario, "targets": " ".join(r.services[i] for i in r.targets)}
    for v in VARIANTS:
        first = starts(r.alarm[v]).any(1) & r.fault
        found, k = bool(first.any()), int(first.argmax())
        order = rank(r.norm[k], r.rule[v][k]) if found else np.array([], int)
        row[f"{v} found"] = int(found)
        row[f"{v} minutes"] = round((r.t[k] - r.t_inject) / 60, 1) if found else np.nan
        row[f"{v} recall"] = round(float(r.alarm[v].any(1)[r.fault].mean()), 3)
        for n in TOP:
            row[f"{v} top@{n}"] = int(found and bool(set(order[:n].tolist()) & set(r.targets)))
        if v == VARIANTS[-1]:
            row["top 5 at the first alarm"] = "; ".join(
                f"{r.services[i]} {r.norm[k, i]:.1f}" + (" restart" if r.rule[v][k, i] else "") for i in order[:5])
    return row


def summary_row(runs, faults, v):
    tp, fn, fp, tn = np.sum([counts(r.alarm[v].any(1), r.fault, r.healthy) for r in runs], 0)
    p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    row = {"variant": v, "fault inferences": tp + fn, "healthy inferences": fp + tn, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
           "precision": round(p, 3), "recall": round(rc, 3), "F1": round(fbeta(p, rc, 1), 3),
           "F0.5": round(fbeta(p, rc, .5), 3), "F2": round(fbeta(p, rc, 2), 3)}
    for s in SOURCES:
        h = [r for r in runs if r.source == s]
        row[f"healthy in alarm % {s}"] = round(100 * sum(int((r.alarm[v].any(1) & r.healthy).sum()) for r in h)
                                               / max(sum(int(r.healthy.sum()) for r in h), 1), 1)
    row["faults found %"] = round(100 * faults[f"{v} found"].mean(), 1)
    row["minutes to alarm, median"] = faults[f"{v} minutes"].median()
    for n in TOP:
        row[f"top@{n} %"] = round(100 * faults[f"{v} top@{n}"].mean(), 1)
    return row


def alarm_rows(r, features):
    rows, v = [], VARIANTS[-1]
    for i, s in enumerate(r.services):
        for a, b in episodes(r.alarm[v][:, i] & (r.on | r.fault)):
            p = a + int(np.argmax(r.norm[a:b, i]))
            rows.append({
                "run": r.id, "role": r.role, "scenario": r.scenario, "service": s, "is_target": int(i in r.targets),
                "during_fault": int(r.fault[a]), "start": int(r.t[a]),
                "start_utc": datetime.fromtimestamp(r.t[a], timezone.utc).isoformat(timespec="seconds"),
                "minutes_from_injection": round((r.t[a] - r.t_inject) / 60, 1) if r.t_inject else None,
                "minutes": (b - a) * 0.5, "peak": round(float(r.norm[p, i]), 2), "restart": int(r.restart[a:b, i].any()),
                "features at peak": "; ".join(f"{f} {sh:.0%} ({o:.4g} vs {e:.4g} {inputs.UNITS[f][1]})"
                                              for f, sh, o, e in explain(features, r.P, r.r2, r.expected, p, i))})
    return rows


def git_state():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "anomaly", ":!anomaly/notebooks"],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()  # the code only, not results/
    return {"commit": head, "uncommitted_changes": bool(dirty)}


def evaluate(name, app, data, out):
    runs, model = load(app, data), MODELS[name]()
    cols, R = [inputs.B0.index(f) for f in model.features], inputs.B0.index("restarts")
    by = lambda *roles: [r for r in runs if r.role in roles]
    model.fit([r.X[r.on][:, :, cols] for r in by("train")])
    for r in runs:
        r.P = r.X[:, :, cols]
        r.r2, r.expected = model.score(r.P)
        r.score, r.restart = persist(r.r2.sum(2)), np.nan_to_num(r.X[:, :, R]) > 0
    thr = np.max([r.score[r.on].max(0) for r in by("train", "val")], 0)
    thr = np.where(thr > 0, thr, np.inf)  # a service that never scored cannot alarm
    for r in runs:
        r.norm = r.score / thr
        r.rule = {"model": np.zeros_like(r.restart), "model+restart": r.restart}
        r.alarm = {v: (r.norm > 1) | r.rule[v] for v in VARIANTS}

    faults = pd.DataFrame([fault_row(r) for r in by("fault-dev", "fault-test") if r.targets])
    summary = pd.DataFrame([summary_row(runs, faults, v) for v in VARIANTS]).set_index("variant")
    g = faults.groupby(["fold", "scenario"])
    minutes = [f"{v} minutes" for v in VARIANTS]
    shares = [c for c in faults if c.startswith("model") and c not in minutes]
    scen = pd.concat([g.size().rename("runs"), (g[shares].mean() * 100).round().astype(int), g[minutes].median()], axis=1)
    alarms = pd.DataFrame([row for r in runs if r.role not in ("train", "val") for row in alarm_rows(r, model.features)])
    card = model.card(runs[0].services)
    for i, s in enumerate(runs[0].services):
        if card[s]:
            card[s]["threshold"] = float(thr[i])

    d = Path(out, f"{name}-B0", app)
    d.mkdir(parents=True, exist_ok=True)
    config = {"app": app, "model": name, "params": model.params, "features": model.features, "data": str(data),
              "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code": git_state(),
              "protocol": {
                  "inference": "one per 30 s step; the system is in alarm when any service is",
                  "score": "the model's score, persistent over 2 steps (min of t and t-1)",
                  "threshold": "per service, max score over the loaded steps of train + val",
                  "restart rule": "model+restart: a service that restarts is in alarm and ranks first",
                  "ground truth": "fault runs are faulty from the injection to their end; healthy: loaded steps of noop "
                                  "runs, clean test and test-unseen sessions, fault runs before the injection",
                  "found": "an alarm of some service starts after the injection",
                  "localization": "services ranked at the first alarm (restarted first, then score / threshold); top@k "
                                  "hits when an accepted target is in the first k; a fault not found is a miss"},
              "runs": {role: [r.id for r in by(role)] for role in sorted({r.role for r in runs})}}
    (d / "config.json").write_text(json.dumps(config, indent=1))
    (d / "model.json").write_text(json.dumps(card, indent=1))
    summary.to_csv(d / "summary.csv")
    scen.to_csv(d / "scenarios.csv")
    faults.to_csv(d / "faults.csv", index=False)
    alarms.to_csv(d / "alarms.csv", index=False)
    return summary, scen, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("app", choices=("hotel", "social"))
    ap.add_argument("--data", default=Path.home() / "datasets/full_v2-built")
    ap.add_argument("--out", default=REPO / "results")
    a = ap.parse_args()
    summary, scen, d = evaluate(a.model, a.app, a.data, a.out)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"{a.app}, {a.model}\n{summary.T}\n\nper scenario (% runs; recall = % of fault inferences in alarm; minutes = median)\n{scen}")
    print(f"written to {d}")
