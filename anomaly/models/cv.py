"""Cross-validation of a model on the healthy sessions of train + val, and the Optuna helpers of the training notebooks
(anomaly/notebooks/train-pca-B0.ipynb).

    PYTHONPATH=. python -m anomaly.models.cv pca|orcrist-AA|... hotel [--k 5]   (the CV of the default parameters)
    python -m anomaly.models.cv worker <spec.json> <i>   (a background worker, started by launch())

Why: in the protocol of evaluate.py the val sessions set the threshold, so their false alarms are 0 by construction, and
the dev fold has no healthy inference, so precision and F1 cannot be measured on dev. Here every clean train and val
session is held out once, out of both the fit and the threshold.

- Folds: the clean train and val sessions, whole (steps next to each other are near copies, a random split would leak),
  in k folds balanced by loaded steps: longest first, each to the fold with the fewest steps.
- Fold i: fit on the other folds, val = fold i (it may stop the training, as Orcrist's early stopping; the PCA ignores
  it); threshold per service = the largest persistent score over the fit sessions; then the model scores fold i (healthy
  steps: false alarms) and every dev fault (no model trains on a fault, so each model scores them all). Test,
  test-unseen, noop and the test faults are never read: they stay for the report. Warm-up, persistence, rules, variants
  and top@k are those of evaluate.py.
- Pooled, every inference counts once: FP and TN summed over the folds (each healthy step is held out once), TP and FN
  averaged over the k models (each dev fault step is scored k times); found, minutes and top@k averaged over the models.
  The objectives of the tuning are F1 and top@3 of the reference system (model+restart+absence).

The faults only choose the parameters, never the weights: the models stay trained on healthy steps only.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from anomaly import build
from anomaly.models import evaluate as ev

K_FOLDS = 5
VARIANT = ev.VARIANTS[-1]  # the reference system
OBJECTIVES = ("F1", "top@3 %")  # both maximized
TOLERANCE = 4  # points of top@3 the choice may give up for F1: about one dev fault
OUT = ev.REPO / "results" / "tuning"
COLLECTION = "collect_clean.sh"  # the nightly collection: no trial starts while it runs (the CPU would bias its data)
PAUSE_UTC = ("17:45", "18:05")  # nor just before it starts. Its lock is never taken: the cron's flock -n would skip the night
MAX_WORKERS = 4  # background workers on the machine, all studies together: 4 x 8 torch threads = the 32 cores


def folds(runs, k=K_FOLDS):
    """The clean train and val sessions in k lists of whole sessions, balanced by loaded steps."""
    out, steps = [[] for _ in range(k)], np.zeros(k, int)
    for r in sorted((r for r in runs if r.role in ("train", "val")), key=lambda r: (-int(r.on.sum()), r.id)):
        i = int(np.argmin(steps))
        out[i].append(r)
        steps[i] += int(r.on.sum())
    return out


def fold_metrics(held, dev, faults, v):
    tp, fn = np.sum([ev.counts(r.alarm[v].any(1), r.fault, r.healthy)[:2] for r in dev], 0)
    fp, tn = np.sum([ev.counts(r.alarm[v].any(1), np.zeros_like(r.on), r.on)[2:] for r in held], 0)
    row = {"variant": v, "TP": tp, "FN": fn, "FP": fp, "TN": tn, "healthy in alarm %": 100 * fp / max(fp + tn, 1),
           "recall": tp / max(tp + fn, 1), "faults found %": 100 * faults[f"{v} found"].mean(),
           "minutes to alarm, median": faults[f"{v} minutes"].median()}
    return row | {f"top@{n} %": 100 * faults[f"{v} top@{n}"].mean() for n in ev.TOP}


def pooled(table):
    """One row per variant: FP, TN summed over the folds, TP, FN averaged; precision, recall and F1 from them; the
    rest averaged over the folds, with the deviation across folds of the main ones."""
    rows = []
    for v, t in table.groupby("variant", sort=False):
        tp, fn, fp, tn = t.TP.mean(), t.FN.mean(), t.FP.sum(), t.TN.sum()
        p, rc = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
        row = {"variant": v, "TP": tp, "FN": fn, "FP": fp, "TN": tn, "precision": p, "recall": rc,
               "F1": ev.fbeta(p, rc, 1), "healthy in alarm %": 100 * fp / max(fp + tn, 1)}
        row |= t[["faults found %", "minutes to alarm, median", *(f"top@{n} %" for n in ev.TOP)]].mean().to_dict()
        row |= {f"{c} sd": t[c].std() for c in ("healthy in alarm %", "recall", "faults found %", "top@3 %")}
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def cross_validate(name, runs, hp=None, k=K_FOLDS, on_epoch=None):
    """The CV of model `name` (a key of evaluate.MODELS) with parameters hp (None: its defaults) on runs from
    evaluate.load; on_epoch(fold, row), when given, sees every epoch of a model that trains by epochs (Orcrist).
    -> SimpleNamespace(folds, table, pooled, faults):
    folds: per fold, the model, the thresholds [S], the fit session ids and copies of the held-out sessions and of the
    dev faults with their scores and alarms (the runs passed in are left untouched);
    table: one row per fold and variant; pooled: one row per variant; faults: one row per fold and dev fault."""
    fs, dev = folds(runs, k), [r for r in runs if r.role == "fault-dev"]
    out, rows = [], []
    for i, held in enumerate(fs):
        train = [r for j, f in enumerate(fs) if j != i for r in f]
        start, model = time.time(), ev.MODELS[name](**(hp or {}))
        if on_epoch:
            model.on_epoch = lambda row, i=i: on_epoch(i, row)
        ev.fit(model, train, held)
        train_c, held_c, dev_c = ([copy.copy(r) for r in rs] for rs in (train, held, dev))  # arrays shared, scores not
        ev.score_runs(model, train_c + held_c + dev_c)
        thr = ev.healthy_max(train_c, "score")
        ev.set_alarms(held_c + dev_c, thr)
        faults = pd.DataFrame([ev.fault_row(r) for r in dev_c if r.targets]).assign(cv_fold=i)  # as the protocol
        extra = {p: model.params[p] for p in ("epochs", "best_epoch") if p in model.params}
        for v in ev.VARIANTS:
            rows.append({"fold": i, "fit steps": sum(int(r.on.sum()) for r in train),
                         "held-out steps": sum(int(r.on.sum()) for r in held)} | fold_metrics(held_c, dev_c, faults, v)
                        | {"seconds": round(time.time() - start, 1)} | extra)
        out.append(SimpleNamespace(i=i, model=model, thr=thr, train=[r.id for r in train], held=held_c, dev=dev_c,
                                   faults=faults))
    table = pd.DataFrame(rows)
    return SimpleNamespace(folds=out, table=table, pooled=pooled(table),
                           faults=pd.concat([f.faults for f in out], ignore_index=True))


# --- Optuna ----------------------------------------------------------------------------------------------------------

def suggest(trial, space):
    """hp from a search space {parameter: spec}: a list is a choice; (lo, hi) a range, of integers when both are int;
    (lo, hi, "log") a range in log scale; any other value is fixed."""
    hp = {}
    for p, s in space.items():
        if isinstance(s, list):
            hp[p] = trial.suggest_categorical(p, s)
        elif isinstance(s, tuple):
            lo, hi, *scale = s
            assert scale in ([], ["log"]), f"{p}: a range is (lo, hi) or (lo, hi, 'log'), not {s}"
            integer = all(isinstance(x, int) and not isinstance(x, bool) for x in (lo, hi))
            hp[p] = (trial.suggest_int if integer else trial.suggest_float)(p, lo, hi, log=bool(scale))
        else:
            hp[p] = s
    return hp


def in_space(value, s):
    return value in s if isinstance(s, list) else s[0] <= value <= s[1] if isinstance(s, tuple) else value == s


def study(name, app, runs, space, study_name, sampler="tpe", k=K_FOLDS, seed=0, out=OUT):
    """The Optuna study `study_name` of model `name` on `app`, kept in out/<name>-B0/<app>/study.journal, so it can be
    resumed. It remembers its space, its folds and its dev faults, and refuses other ones: a new space or new data is a
    new study_name. sampler: "tpe", "random" or "grid" (every value of the lists, the space must have no range).
    A new tpe or random study starts with the default parameters (the reference), when they are in the space; a grid
    has them when its lists hold them."""
    import optuna
    d = Path(out, f"{name}-B0", app)
    d.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(d / "study.journal")))
    if sampler == "grid":
        assert not any(isinstance(s, tuple) for s in space.values()), "a grid takes lists, not ranges"
        smp = optuna.samplers.GridSampler({p: s for p, s in space.items() if isinstance(s, list)}, seed=seed)
    else:
        smp = {"tpe": optuna.samplers.TPESampler, "random": optuna.samplers.RandomSampler}[sampler](seed=seed)
    s = optuna.create_study(study_name=study_name, storage=storage, sampler=smp, directions=["maximize"] * 2,
                            load_if_exists=True)
    key = {"space": repr(space), "sampler": sampler, "folds": [[[r.id, int(r.on.sum())] for r in f] for f in folds(runs, k)],
           "dev": sorted(r.id for r in runs if r.role == "fault-dev")}
    if s.user_attrs:
        for a in key:
            assert s.user_attrs[a] == key[a], f"study {study_name} was made with another {a}: give a new study_name"
        return s
    for a, v in key.items():
        s.set_user_attr(a, v)
    default = ev.MODELS[name]().params
    if sampler != "grid" and all(in_space(default[p], v) for p, v in space.items() if p in default):
        s.enqueue_trial({p: default[p] for p, v in space.items() if isinstance(v, (list, tuple))})
    return s


def collecting():
    return subprocess.run(["pgrep", "-f", COLLECTION], capture_output=True).returncode == 0


def wait_for_collection(poll=60):
    said = False
    while collecting() or PAUSE_UTC[0] <= datetime.now(timezone.utc).strftime("%H:%M") < PAUSE_UTC[1]:
        if not said:
            print(f"{datetime.now(timezone.utc):%H:%M} UTC: the nightly collection runs or is about to, waiting", flush=True)
            said = True
        time.sleep(poll)


def optimize(s, name, runs, space, n_trials=None, timeout_h=None, k=K_FOLDS):
    """Run trials of study s: each is a CV, its objectives F1 and top@3 of the reference system; every pooled metric
    and the full parameters are kept in the trial. One line per trial. Stops when the study has n_trials complete
    trials (all workers together; the trials running then end first), after timeout_h hours, or when a grid is done.
    A trial that fails (an error in the fit) is logged and skipped."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    grid = isinstance(s.sampler, optuna.samplers.GridSampler)
    assert grid or n_trials or timeout_h, "tpe and random run until n_trials or timeout_h: give one"
    done = sum(t.state.name == "COMPLETE" for t in s.trials)
    if grid and done >= np.prod([len(v) for v in space.values() if isinstance(v, list)]) or n_trials and done >= n_trials:
        print(f"study {s.study_name} is done: {done} complete trials")
        return
    default = ev.MODELS[name]().params

    def objective(trial):
        wait_for_collection()
        hp = suggest(trial, space)
        row = cross_validate(name, runs, hp, k).pooled.loc[VARIANT]
        trial.set_user_attr("hp", hp)
        trial.set_user_attr("reference", all(default[p] == v for p, v in hp.items() if p in default))
        for c, v in row.items():
            trial.set_user_attr(c, float(v))
        return tuple(float(np.nan_to_num(row[o])) for o in OBJECTIVES)

    def say(study, t):
        if t.values:
            a = t.user_attrs
            print(f"trial {t.number:>3}{' (reference)' if a.get('reference') else ''}: F1 {t.values[0]:.3f}, top@3 "
                  f"{t.values[1]:.0f}%, healthy in alarm {a['healthy in alarm %']:.2f}%, recall {a['recall']:.3f}, "
                  f"found {a['faults found %']:.0f}%  {t.params}  ({(t.datetime_complete - t.datetime_start).seconds} s)",
                  flush=True)

    stop = [optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,))] if n_trials else []
    s.optimize(objective, timeout=timeout_h * 3600 if timeout_h else None, callbacks=[say, *stop], gc_after_trial=True,
               catch=(Exception,))


def launch(s, name, app, space, sampler, n_trials=None, timeout_h=None, k=K_FOLDS, workers=MAX_WORKERS,
           data=build.DATASET / "built", out=OUT):
    """Run study s in `workers` background processes, detached from the notebook (they go on if its kernel stops),
    each with its own sampler seed. Logs: out/<name>-B0/<app>/<study>.worker<i>.log. At most MAX_WORKERS run on the
    machine, all studies together. Follow them with workers(), stop them with stop()."""
    running = int(subprocess.run(["pgrep", "-fc", "anomaly.models.cv worker"], capture_output=True, text=True).stdout or 0)
    assert running + workers <= MAX_WORKERS, f"{running} workers already run: at most {MAX_WORKERS} on the machine"
    d = Path(out, f"{name}-B0", app)
    spec = d / f"{s.study_name}.json"
    spec.write_text(json.dumps({"name": name, "app": app, "data": str(data), "space": repr(space), "study": s.study_name,
                                "sampler": sampler, "k": k, "n_trials": n_trials, "timeout_h": timeout_h,
                                "out": str(out)}, indent=1))
    env = os.environ | {"PYTHONPATH": str(ev.REPO), "OMP_NUM_THREADS": "8"}
    pids = []
    for w in range(workers):
        with open(d / f"{s.study_name}.worker{w}.log", "a") as log:
            pids.append(subprocess.Popen([sys.executable, "-m", "anomaly.models.cv", "worker", str(spec), str(w)],
                                         cwd=ev.REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True).pid)
    spec.with_suffix(".pids").write_text(json.dumps(pids))
    print(f"{workers} workers started on {s.study_name}: {pids}")


def alive(pid):
    """A worker process that runs (not gone, not a zombie, still a worker)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z" and b"anomaly.models.cv" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, IndexError):
        return False


def workers(s, name, app, out=OUT, lines=2):
    """The background workers of study s: whether each runs and the last lines of its log; the complete trials."""
    d = Path(out, f"{name}-B0", app)
    pids = json.loads((d / f"{s.study_name}.pids").read_text())
    for w, pid in enumerate(pids):
        log = (d / f"{s.study_name}.worker{w}.log").read_text().splitlines()
        print(f"worker {w} (pid {pid}): {'running' if alive(pid) else 'ended'}", *log[-lines:], sep="\n    ")
    states = pd.Series([t.state.name for t in s.trials]).value_counts().to_dict()
    print(f"trials: {states} (RUNNING includes any left by a stopped worker)")


def stop(s, name, app, out=OUT):
    """Stop the background workers of study s; their running trials are lost (the complete ones stay)."""
    for pid in json.loads(Path(out, f"{name}-B0", app, f"{s.study_name}.pids").read_text()):
        if alive(pid):
            os.kill(pid, signal.SIGTERM)


def work(spec, w):
    """A background worker: the study of spec (made before by study()), sampler seed w."""
    a = json.loads(Path(spec).read_text())
    space = ast.literal_eval(a["space"])
    runs = ev.load(a["app"], a["data"])
    s = study(a["name"], a["app"], runs, space, a["study"], a["sampler"], a["k"], seed=w, out=a["out"])
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: worker {w} on {a['study']}", flush=True)
    optimize(s, a["name"], runs, space, a["n_trials"], a["timeout_h"], a["k"])
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: worker {w} done", flush=True)


def trials(s):
    """One row per complete trial: its full parameters, the pooled metrics of the reference system, whether it is on
    the Pareto front of F1 and top@3 and whether it is the reference (the default parameters)."""
    front = {t.number for t in s.best_trials} if any(t.state.name == "COMPLETE" for t in s.trials) else set()
    return pd.DataFrame(columns=["trial"]).set_index("trial") if not front else pd.DataFrame([{"trial": t.number, **t.user_attrs["hp"],
                          **{c: v for c, v in t.user_attrs.items() if c not in ("hp", "reference")},
                          "pareto": t.number in front, "reference": bool(t.user_attrs.get("reference"))}
                         for t in s.trials if t.state.name == "COMPLETE"]).set_index("trial")


def choose(df, tol=TOLERANCE):
    """The rule fixed before the search: on the Pareto front, the largest F1 among the trials whose top@3 is within
    tol points of the best."""
    front = df[df.pareto]
    near = front[front["top@3 %"] >= front["top@3 %"].max() - tol]
    return near.loc[near["F1"].idxmax()]


def save(s, name, app, chosen, out=OUT, confirm=None):
    """trials.csv and choice.json next to the study: the chosen trial with its parameters and metrics, the reference
    when the study has it, the rule, the folds, the data and the code; confirm (the seeds check), as confirm.csv."""
    d = Path(out, f"{name}-B0", app)
    df = trials(s)
    df.to_csv(d / "trials.csv")
    if confirm is not None:
        confirm.to_csv(d / "confirm.csv")
    ref = df[df.reference]
    choice = {"model": name, "app": app, "study": s.study_name, "trials": len(df),
              "rule": f"Pareto front of {' and '.join(OBJECTIVES)} ({VARIANT}); the largest F1 within {TOLERANCE} points "
                      "of the best top@3",
              "chosen": {"trial": int(chosen.name), **chosen.to_dict()},
              "reference": {"trial": int(ref.index[0]), **ref.iloc[0].to_dict()} if len(ref) else None,
              "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code": ev.git_state(),
              "confirmed with seeds": confirm is not None, **s.user_attrs}
    (d / "choice.json").write_text(json.dumps(choice, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return d


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        work(sys.argv[2], int(sys.argv[3]))
        sys.exit()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=ev.MODELS)
    ap.add_argument("app", choices=("hotel", "social"))
    ap.add_argument("--k", type=int, default=K_FOLDS)
    ap.add_argument("--data", default=build.DATASET / "built")
    a = ap.parse_args()
    res = cross_validate(a.model, ev.load(a.app, a.data), k=a.k)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(res.table[res.table.variant == VARIANT].round(3).to_string(index=False), "\n")
    print(res.pooled.round(3).T)
