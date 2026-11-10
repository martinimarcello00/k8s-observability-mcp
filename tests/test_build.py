# Run from the repo root: PYTHONPATH=. python tests/test_build.py  (no cluster, no data needed)
import math
from pathlib import Path
import tempfile

import numpy as np

from anomaly import build as b

nan = float("nan")


def same(a, expected):
    return np.allclose(np.asarray(a, dtype=float), expected, equal_nan=True)


# --- counters become rates on the real timestamps; a reset is skipped, gauges stay as they are
assert b.series_points([[0, "0"], [10, "10"], [30, "30"], [40, "5"]], "rate") == [(10, 1.0), (30, 1.0)]
assert b.series_points([[0, "7"], [16, "9"]], "gauge") == [(0, 7.0), (16, 9.0)]

# --- grid: mean of the points in (t - step, t], NaN where none fell
grid = np.array([30.0, 60.0, 90.0])
assert same(b.on_grid([(10, 1.0), (30, 3.0), (70, 5.0)], grid, 30), [2.0, nan, 5.0])
assert same(b.on_grid([(95, 1.0), (-5, 1.0)], grid, 30), [nan, nan, nan])  # outside the grid
assert same(b.on_grid([], grid, 30), [nan, nan, nan])


# --- per service: app containers summed per pod, pause and pod totals ignored, restarted containers merged
def s(pod, container, values, **labels):
    return {"metric": {"pod": pod, "container": container, **labels}, "values": values}


g = [[10, "1"], [40, "2"], [70, "3"]]  # gauge, one point per step
series = [
    s("geo-5d4f-abcde", "geo", g),
    s("geo-5d4f-abcde", "POD", [[10, "100"], [40, "100"], [70, "100"]]),  # pause container
    s("geo-5d4f-abcde", "", [[10, "500"], [40, "500"], [70, "500"]]),  # pod total
    s("frontend-77-xyz", "app", g),
    s("frontend-77-xyz", "sidecar", [[10, "10"], [70, "30"]]),  # no sample in the second step: the pod is missing there
    s("rate-1-a", "rate", [[10, "4"]]),  # old container, killed
    s("rate-1-a", "rate", [[40, "5"], [70, "6"]]),  # new container after the restart
    s("jaeger-9-q", "jaeger", g),  # not in the datagraph
]
services = ["frontend", "geo", "rate", "user"]
m = b.feature_matrix(series, "gauge", "container", services, np.array([30.0, 60.0, 90.0]), 30)
assert same(m[:, 0], [11, nan, 33]) and same(m[:, 1], [1, 2, 3]) and same(m[:, 2], [4, 5, 6]) and same(m[:, 3], [nan] * 3)
net = [s("geo-5d4f-abcde", "", [[0, "0"], [16, "1600"], [32, "3200"]], interface="eth0"), s("geo-5d4f-abcde", "geo", [[0, "0"], [16, "9"]])]
assert same(b.feature_matrix(net, "rate", "pod", services, np.array([30.0, 60.0]), 30)[:, 1], [100, 100])  # the pod's own eth0 series

# --- datagraph: services, databases and caches, not the CALLS edges
graph = Path(tempfile.mkdtemp()) / "g.txt"
graph.write_text("MERGE (s:Service {name: 'frontend'});\nMERGE (db:Database {name: 'mongodb-geo'});\n"
                 "MATCH (a:Service {name: 'frontend'}), (b:Service {name: 'geo'}) MERGE (a)-[:CALLS]->(b);\n")
assert b.datagraph_services(graph) == ["frontend", "mongodb-geo"]
assert b.pod_service("compose-post-service-64f6c9b656-6ddw8") == "compose-post-service"

# --- wrk2 log: old and new segment lines, delivered rate, errors, a wrk that failed
log = """segment 1000 120 10
 99.000%    6.32ms
 99.000%    9.10ms
  Requests/sec:     10.30
segment 1120 120 30 2
 99.000%    1.20s
  Non-2xx or 3xx responses: 7
  Requests/sec:     29.96
segment 1240 120 30 1
segment 1250 120 30 1
"""
segs = b.load_segments(log)
assert [(x["start"], x["rate"], x["mix"], x["delivered"], x["p99_ms"], x["non_2xx"]) for x in segs] == [
    (1000, 10, 0, 10.3, 6.32, 0), (1120, 30, 2, 29.96, 1200.0, 7), (1240, 30, 1, None, None, 0), (1250, 30, 1, None, None, 0)]
rate, mix = b.load_on_grid(segs, np.array([990.0, 1000.0, 1130.0, 1245.0, 1500.0]))
assert list(rate) == [-1, 10, 30, 30, -1] and list(mix) == [-1, 0, 2, 1, -1]


# --- validity: a clean session on its observation, a fault run on its settle
def fake(kind, restart_at, delivered=30.0, errors=0):
    grid = np.arange(1030.0, 2030.0, 30)
    features = [f[0] for f in b.FEATURES]
    x = np.zeros((len(grid), 2, len(features)))
    x[grid == restart_at, 0, features.index("restarts")] = 0.03
    meta = {"kind": kind, "settle_s": 300, "t_export": 2000.0, "run_meta": {"t_workload": 1000.0, "t_inject": 1300.0}}
    segments = [{"start": 1000, "seconds": 300, "rate": 30, "mix": 0, "delivered": delivered, "non_2xx": errors},
                {"start": 1300, "seconds": 400, "rate": 30, "mix": 0, "delivered": 12.0, "non_2xx": 50},
                {"start": 1700, "seconds": 400, "rate": 30, "mix": 0, "delivered": None, "non_2xx": 0}]  # still running at the export
    return b.checks(meta, segments, x, np.ones_like(x, dtype=bool), grid, features, ["geo", "user"])


ok = lambda c: {k: v[0] for k, v in c.items() if k != "window"}
assert ok(fake("test", restart_at=1600.0)) == {"V1_load": True, "V2_health": True, "V3_coverage": True}  # all after the fault
assert ok(fake("test", restart_at=1150.0))["V2_health"] is False  # a restart during the settle: the environment was not healthy
assert ok(fake("test", 1600.0, delivered=20.0))["V1_load"] is False  # load not delivered before the fault
assert ok(fake("test", 1600.0, errors=3))["V2_health"] is False
clean = fake("clean", restart_at=1600.0)
assert clean["V2_health"][1]["restarted"] == ["geo"] and clean["V1_load"][1]["short"] == [(1300, 30, 12.0)]  # a clean session has no excuse

# --- V4: host I/O wait share, from per-CPU counters of one node exporter; the second node repeats the same kernel
cpu = lambda node, mode, per_s: {"metric": {"instance": node, "cpu": "0", "mode": mode},
                                  "values": [[t, per_s * (t - 1000)] for t in range(1000, 1121, 15)]}
series = [cpu("n1", "idle", 0.9), cpu("n1", "iowait", 0.1), cpu("n2", "idle", 0.0), cpu("n2", "iowait", 1.0)]
share = b.host_iowait(series, np.array([1030.0, 1060.0, 1090.0, 1120.0]), 30)
assert np.allclose(share, 0.1)  # n1 only: 0.1 of its CPU time waits for the disk

print("ok")
