"""Turn the raw runs of SRE-agent's collect_dataset.py into model input: per-service features on a time grid.

    poetry run python -m anomaly.build RAW_APP_DIR OUT_DIR [--step 30]

RAW_APP_DIR is <out>/<app> of collect_dataset.py. Next to the arrays below, OUT_DIR/<kind>/<run>.spans.csv.gz and
.logs.csv.gz hold one row per span and per log line (anomaly/typed.py). For every run, its last valid attempt becomes OUT_DIR/<kind>/<run>.npz:
  x [T, S, F] float32   feature values, NaN where no sample fell in the grid step
  mask [T, S, F] bool   True where x holds a real value
  t [T]                 grid times (end of each step, unix seconds)
  rate, mix [T]         load of the wrk2 segment at each step (-1 outside the schedule)
  services, features    names of the S and F axes
and OUT_DIR/<kind>/<run>.json: labels, split, load schedule and the validity checks V1-V4.
Nothing is normalized or windowed here: that belongs to training, which must see only the train split.
"""
import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATAGRAPH = {"Hotel Reservation": "hotel-reservation-datagraph.txt", "Social Network": "social-network-datagraph.txt"}
# name, cAdvisor metric, counter (per-second rate) or gauge, summed over app containers or read on the pod's own series (network)
FEATURES = [
    ("cpu", "container_cpu_usage_seconds_total", "rate", "container"),
    ("cpu_user", "container_cpu_user_seconds_total", "rate", "container"),
    ("cpu_system", "container_cpu_system_seconds_total", "rate", "container"),
    ("cpu_throttled", "container_cpu_cfs_throttled_seconds_total", "rate", "container"),  # absent without CPU limits (Social)
    ("mem_working_set", "container_memory_working_set_bytes", "gauge", "container"),
    ("mem_rss", "container_memory_rss", "gauge", "container"),
    ("mem_cache", "container_memory_cache", "gauge", "container"),
    ("threads", "container_threads", "gauge", "container"),
    ("rx_bytes", "container_network_receive_bytes_total", "rate", "pod"),
    ("tx_bytes", "container_network_transmit_bytes_total", "rate", "pod"),
    ("rx_packets", "container_network_receive_packets_total", "rate", "pod"),
    ("tx_packets", "container_network_transmit_packets_total", "rate", "pod"),
    ("restarts", "kube_pod_container_status_restarts_total", "rate", "kube"),  # state channel: restarts per second
]
MIN_RATE_SHARE = 0.9  # V1: a wrk2 segment must deliver at least this share of its target rate
MAX_MASKED = 0.2  # V3: at most this share of empty cells overall, and twice this for any service and feature
# V4: at most this share of the host's CPU time waiting for the disk, on average over the run. Healthy pilot runs stay
# under 1.5% on average; a single 30 s step reaches 5% in them too, so the worst step is reported but not judged
MAX_IOWAIT = 0.05


def datagraph_services(path):
    """Services, databases and caches of the datagraph, in file order: the S axis."""
    return [name for _, name in re.findall(r"MERGE \(\w*:(\w+) \{name: '([^']+)'\}\)", Path(path).read_text())]


def datagraph_edges(path):
    """(source, relation, target) of the datagraph: CALLS between services, USES towards databases and caches.

    Cypher names nodes through variables: a statement's own declarations win over those made elsewhere in the file.
    """
    declare = re.compile(r"\((\w+):\w+ \{name: '([^']+)'\}\)")
    text = Path(path).read_text()
    names = dict(declare.findall(text))
    edges = []
    for statement in text.split(";"):
        local = {**names, **dict(declare.findall(statement))}
        for a, rel, b in re.findall(r"MERGE \((\w+)\)-\[:(\w+)\]->\((\w+)\)", statement):
            edges.append((local.get(a, a), rel, local.get(b, b)))
    return edges


def pod_service(pod):
    """Deployment pods are <service>-<replicaset hash>-<pod hash>: dropping the last two tokens gives the service."""
    return pod.rsplit("-", 2)[0]


def series_points(values, kind):
    """(time, value) of one series: the gauge itself, or the rate between consecutive real samples, at the later one.

    cAdvisor timestamps are irregular (10 to 40 s apart), so a rate uses the real interval; a counter reset is skipped.
    """
    pts = [(float(t), float(v)) for t, v in values]
    if kind == "gauge":
        return pts
    return [(t2, (v2 - v1) / (t2 - t1)) for (t1, v1), (t2, v2) in zip(pts, pts[1:]) if t2 > t1 and v2 >= v1]


def on_grid(points, grid, step):
    """Mean of the points in each grid step (t - step, t]; NaN where none fell in it."""
    out = np.full(len(grid), np.nan)
    if points:
        times = np.array([t for t, _ in points])
        vals = np.array([v for _, v in points])
        cell = np.searchsorted(grid, times, side="left")  # first grid time at or after the point
        ok = cell < len(grid)
        ok[ok] = grid[cell[ok]] - times[ok] < step
        total = np.bincount(cell[ok], weights=vals[ok], minlength=len(grid))
        count = np.bincount(cell[ok], minlength=len(grid))
        out[count > 0] = total[count > 0] / count[count > 0]
    return out


def feature_matrix(series, metric_kind, level, services, grid, step):
    """[T, S] for one feature: app containers summed per pod, network on the pod's own series, then the mean over a service's pods.

    A pod counts only in the steps where all of its containers have a value: a partial sum would look like a drop.
    A restarted container has an old and a new series: they never run at once, so they merge into one.
    """
    containers = defaultdict(list)
    for s in series:
        m = s["metric"]
        container = m.get("container", "")
        if level == "container" and container in ("", "POD"):
            continue  # pod total and pause container: the app containers are summed instead
        if level == "pod" and (container or m.get("interface", "eth0") != "eth0"):
            continue
        containers[(m["pod"], container)].append(on_grid(series_points(s["values"], metric_kind), grid, step))
    pods = defaultdict(list)
    for (pod, _), cols in containers.items():
        pods[pod].append(np.fmax.reduce(cols))  # fmax ignores the NaN of the series that is not running
    out = np.full((len(grid), len(services)), np.nan)
    index = {name: i for i, name in enumerate(services)}
    per_service = defaultdict(list)
    for pod, cols in pods.items():
        if pod_service(pod) in index:  # jaeger and consul are not in the datagraph
            per_service[index[pod_service(pod)]].append(np.sum(cols, axis=0))  # NaN if any container is missing
    for i, cols in per_service.items():
        stack = np.array(cols)
        present = ~np.isnan(stack)
        count = present.sum(axis=0)
        total = np.where(present, stack, 0).sum(axis=0)
        out[:, i] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


MS = {"us": 1e-3, "ms": 1.0, "s": 1e3, "m": 6e4}


def load_segments(wrk_log):
    """wrk2 segments in order: start, seconds, rate, mix, delivered requests/s, p99 latency in ms and non-2xx.

    delivered and p99_ms are None when wrk failed or had not finished the segment.
    """
    segments = []
    for line in wrk_log.splitlines():
        text = line.strip()
        if line.startswith("segment "):
            fields = [int(v) for v in line.split()[1:]]
            start, seconds, rate = fields[:3]
            segments.append({"start": start, "seconds": seconds, "rate": rate, "mix": fields[3] if len(fields) > 3 else 0,
                             "delivered": None, "p99_ms": None, "non_2xx": 0})
        elif not segments:
            continue
        elif text.startswith("Requests/sec:"):
            segments[-1]["delivered"] = float(text.split()[1])
        elif text.startswith("Non-2xx or 3xx responses:"):
            segments[-1]["non_2xx"] = int(text.split(":")[1])
        elif text.startswith("99.000%") and segments[-1]["p99_ms"] is None:  # the first of the two latency tables wrk2 prints
            value, unit = re.match(r"99\.000%\s+([\d.]+)([a-z]+)", text).groups()
            segments[-1]["p99_ms"] = float(value) * MS[unit]
    return segments


def load_on_grid(segments, grid):
    """Rate and mix of the segment running at each grid time; -1 where no segment covers it."""
    rate, mix = np.full(len(grid), -1), np.full(len(grid), -1)
    for seg in segments:  # a retried segment starts again with the same rate: later lines only extend it
        end = seg["start"] + seg["seconds"]
        on = (grid >= seg["start"]) & (grid < end)
        rate[on], mix[on] = seg["rate"], seg["mix"]
    return rate, mix


def checks(meta, segments, x, mask, grid, features, services):
    """V1 load delivered, V2 healthy (no restart, no HTTP error), V3 enough samples. Each is (passed, detail).

    A clean session is checked on its whole observation. A fault run is checked on its settle, before the injection:
    what follows the fault (errors, restarts, slow requests, missing samples) is the fault itself, not a broken environment.
    """
    t_inject, t_export = meta["run_meta"]["t_inject"], meta["t_export"]
    t0, t1 = (t_inject, t_export) if meta["kind"] == "clean" else (meta["run_meta"].get("t_workload", t_inject - meta["settle_s"]), t_inject)
    # segments ending inside the window; the one still running at the export had not reported yet
    observed = [s for s in segments if t0 < s["start"] + s["seconds"] <= min(t1, t_export)]
    short = [s for s in observed if s["delivered"] is None or s["delivered"] < MIN_RATE_SHARE * s["rate"]]
    covered = max((s["start"] + s["seconds"] for s in segments), default=0) >= t_export - 60  # wrk2 alive to the end
    v1 = (not short and covered, {"segments": len(observed), "short": [(s["start"], s["rate"], s["delivered"]) for s in short],
                                  "schedule_covers_run": covered})
    window = (grid > t0) & (grid <= t1)
    restarts = x[window][:, :, features.index("restarts")]
    restarted = sorted(services[i] for i in np.where(np.nansum(restarts, axis=0) > 0)[0])
    errors = sum(s["non_2xx"] for s in observed)
    v2 = (not restarted and errors == 0, {"restarted": restarted, "non_2xx": errors})
    present = mask[window].any(axis=0)  # services and features that exist at all in this app
    share = 1 - mask[window].sum(axis=0) / max(window.sum(), 1)
    overall = float(1 - mask[window][:, present].mean()) if present.any() else 1.0
    worst = {f"{services[i]}/{features[j]}": round(float(share[i, j]), 3) for i, j in zip(*np.where(present & (share > 2 * MAX_MASKED)))}
    v3 = (overall <= MAX_MASKED and not worst, {"masked_share": round(overall, 3), "over_twice_the_limit": worst})
    return {"window": [t0, t1], "V1_load": v1, "V2_health": v2, "V3_coverage": v3}


def host_iowait(series, grid, step):
    """Share of the host's CPU time spent waiting for the disk, per grid step, from node_cpu_seconds_total.

    The kind nodes share the host's kernel, so one node exporter is enough. On a slow disk the databases stall on
    their writes: the services then slow down for a reason that is neither the load nor the fault.
    """
    node = next((s["metric"].get("instance") for s in series), None)
    by_mode = defaultdict(lambda: np.zeros(len(grid)))
    for s in series:
        if s["metric"].get("instance") == node:
            by_mode[s["metric"]["mode"]] += np.nan_to_num(on_grid(series_points(s["values"], "rate"), grid, step))
    total = sum(by_mode.values(), np.zeros(len(grid)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, by_mode["iowait"] / total, np.nan)


def host_io_check(attempt_dir, grid, step):
    """V4 on the whole run, fault included: a stalled disk disturbs the response to the fault as much as the normal state."""
    path = Path(attempt_dir) / "prometheus" / "extra_node_cpu.json.gz"
    if not path.exists():
        return (False, {"missing": "extra_node_cpu"})
    with gzip.open(path, "rt") as f:
        share = host_iowait(json.load(f), grid, step)
    worst, mean = float(np.nanmax(share, initial=0)), float(np.nanmean(share)) if np.isfinite(share).any() else float("nan")
    return (mean <= MAX_IOWAIT, {"mean": round(mean, 4), "worst_step": round(worst, 4)})


def build_run(attempt_dir, datagraph_dir, step=30):
    """Arrays and labels of one valid attempt."""
    attempt_dir = Path(attempt_dir)
    meta = json.loads((attempt_dir / "meta.json").read_text())
    run_meta = meta["run_meta"]
    services = datagraph_services(Path(datagraph_dir) / DATAGRAPH[meta["app_scenario"]])
    start = run_meta.get("t_workload", run_meta["t_inject"])
    grid = np.arange(math.ceil(start / step) * step, meta["t_export"], step, dtype=float)
    x = np.full((len(grid), len(services), len(FEATURES)), np.nan, dtype=np.float64)
    for j, (_, metric, kind, level) in enumerate(FEATURES):
        path = attempt_dir / "prometheus" / f"{metric}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as f:
                x[:, :, j] = feature_matrix(json.load(f), kind, level, services, grid, step)
    mask = ~np.isnan(x)
    log = attempt_dir / "logs" / "wrk2.log.gz"
    segments = load_segments(gzip.open(log, "rt").read()) if log.exists() else []
    rate, mix = load_on_grid(segments, grid)
    features = [f[0] for f in FEATURES]
    labels = {key: meta.get(key) for key in ("id", "kind", "rep", "split", "attempt", "app_scenario", "fault_type", "target",
                                              "accepted_targets", "schedule", "settle_s", "observe_s", "versions")}
    labels.update(t_workload=run_meta.get("t_workload"), t_inject=run_meta["t_inject"], t_injected=run_meta["t_injected"], step=step,
                  checks={**checks(meta, segments, x, mask, grid, features, services), "V4_host_io": host_io_check(attempt_dir, grid, step)},
                  segments=segments)
    arrays = dict(x=x.astype(np.float32), mask=mask, t=grid, rate=rate, mix=mix, services=np.array(services), features=np.array(features))
    return arrays, labels


def last_valid_attempt(run_dir):
    for attempt in sorted(Path(run_dir).glob("attempt-*"), key=lambda p: -int(p.name.split("-")[1])):
        meta = attempt / "meta.json"
        if meta.exists() and json.loads(meta.read_text())["status"] == "valid":
            return attempt
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", type=Path, help="<out>/<app> of collect_dataset.py")
    p.add_argument("out", type=Path)
    # cAdvisor samples every 16 s on average, with gaps up to 40 s: 15 s steps leave 10-16% of the cells empty, 30 s under 1%
    p.add_argument("--step", type=int, default=30, help="grid step in seconds")
    p.add_argument("--datagraph", type=Path, default=Path(__file__).resolve().parents[1] / "service-graph")
    args = p.parse_args(argv)
    for run_dir in sorted(d for d in args.raw.glob("*/*") if d.is_dir()):
        attempt = last_valid_attempt(run_dir)
        if attempt is None:
            print(f"{run_dir.parent.name}/{run_dir.name}: no valid attempt, skipped")
            continue
        arrays, labels = build_run(attempt, args.datagraph, args.step)
        target = args.out / run_dir.parent.name / run_dir.name
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target.with_suffix(".npz"), **arrays)
        target.with_suffix(".json").write_text(json.dumps(labels, indent=1))
        from anomaly import typed  # pandas: only the typed tables need it
        typed.spans(attempt).to_csv(f"{target}.spans.csv.gz", index=False)
        typed.logs(attempt).to_csv(f"{target}.logs.csv.gz", index=False)
        verdict = {k: v[0] for k, v in labels["checks"].items() if k != "window"}
        print(f"{run_dir.parent.name}/{run_dir.name}: {arrays['x'].shape}, masked {labels['checks']['V3_coverage'][1]['masked_share']}, {verdict}")


if __name__ == "__main__":
    main()
