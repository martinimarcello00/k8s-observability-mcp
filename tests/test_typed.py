# Run from the repo root: PYTHONPATH=. python tests/test_typed.py  (no cluster, no data needed)
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from anomaly import build, typed

# --- log lines: one real line per format seen in the pilot (timestamps of kubectl already removed)
cases = {
    '[2026-Nov-09 04:36:27.864746] <error>: (ComposePostHandler.h:137:_ComposeCreaterHelper) Failed to send compose-creator to user-service':
        ("error", "ComposePostHandler.h:137:_ComposeCreaterHelper", "Failed to send compose-creator to user-service"),
    '[2026-Nov-09 05:05:23.093114] <error>: (TextHandler.h:93:operator()) Failed to upload urls to url-shorten-service':
        ("error", "TextHandler.h:93:operator()", "Failed to upload urls to url-shorten-service"),
    '\x1b[90m2026-11-08T17:33:13Z\x1b[0m \x1b[31mERR\x1b[0m \x1b[1mservices/frontend/server.go:213\x1b[0m\x1b[36m >\x1b[0m \x1b[1mSearchHandler CheckAvailability failed\x1b[0m':
        ("error", "services/frontend/server.go:213", "SearchHandler CheckAvailability failed"),
    '{"level":"info","time":"2026-11-08T22:10:18Z","message":"TLS disabled."}': ("info", "", "TLS disabled."),
    '{"t":{"$date":"2026-11-08T22:10:53.995+00:00"},"s":"W",  "c":"ASIO",     "id":22601,   "ctx":"main","msg":"No TransportLayer configured during NetworkInterface startup"}':
        ("warning", "ASIO", "No TransportLayer configured during NetworkInterface startup"),
    '1:C 09 Nov 2026 06:12:52.688 # Redis version=6.2.4, bits=64': ("warning", "", "Redis version=6.2.4, bits=64"),
    '2026-11-08T22:09:31.671Z [WARN]  agent.server.raft: heartbeat timeout reached': ("warning", "agent.server.raft", "heartbeat timeout reached"),
    '2026/11/09 04:36:27 [error] 33#33: *779 [lua] compose.lua:62: ComposePost(): compost_post failure: No more data to read.':
        ("error", "compose.lua:62", "ComposePost(): compost_post failure: No more data to read."),
    'panic: server returned error on SASL authentication step: Authentication failed.':
        ("fatal", "", "panic: server returned error on SASL authentication step: Authentication failed."),
    '==> Starting Consul agent...': ("", "", "==> Starting Consul agent..."),
    'Thrift: Mon Nov  9 08:28:11 2026 TSocket::open() connect() <Host: text-service Port: 9090>: Connection refused':
        ("error", "TSocket::open()", "connect() <Host: text-service Port: 9090>: Connection refused"),
}
for line, expected in cases.items():
    assert typed.parse_log_line(line) == expected, (line, typed.parse_log_line(line))

# --- a run folder with two export blocks and a log file
run = Path(tempfile.mkdtemp())
(run / "jaeger").mkdir()
(run / "logs").mkdir()


def span(trace, sid, parent, start, dur, proc, op, kind, **tags):
    return {"traceID": trace, "spanID": sid, "operationName": op, "startTime": start, "duration": dur, "processID": proc,
            "references": [{"refType": "CHILD_OF", "traceID": trace, "spanID": parent}] if parent else [],
            "tags": [{"key": "span.kind", "value": kind}] + [{"key": k, "value": v} for k, v in tags.items()],
            "logs": [{"fields": [{"key": "event", "value": "error"}, {"key": "message", "value": "no connection"}]}] if tags.get("error") else []}


processes = {"p1": {"serviceName": "frontend", "tags": [{"key": "hostname", "value": "frontend-1-a"}]},
             "p2": {"serviceName": "search", "tags": [{"key": "hostname", "value": "search-1-b"}]}}
t0 = 1_000_000_000_000  # microseconds
bad = {"traceID": "aaaa", "processes": processes, "spans": [
    span("aaaa", "aaaa", "", t0, 500, "p1", "HTTP GET /hotels", "server", **{"http.status_code": 500}),
    span("aaaa", "c1", "aaaa", t0 + 10, 450, "p1", "/search.Search/Nearby", "client", error=True, response_code="Unavailable"),
    span("aaaa", "s1", "c1", t0 + 100, 40, "p2", "/search.Search/Nearby", "server", error=True, response_code="Unavailable"),
    span("aaaa", "c2", "s1", t0 + 110, 20, "p2", "/geo.Geo/Nearby", "client", error=True, response_code="Unavailable",
         **{"peer.service": "geo"})]}
good = {"traceID": "bbbb", "processes": processes, "spans": [
    span("bbbb", "bbbb", "", t0 + 31_000_000, 300, "p1", "HTTP GET /hotels", "server", **{"http.status_code": 200}),
    span("bbbb", "c3", "bbbb", t0 + 31_000_010, 250, "p1", "/search.Search/Nearby", "client"),
    span("bbbb", "s3", "c3", t0 + 31_000_100, 100, "p2", "/search.Search/Nearby", "server")]}
for name, traces in (("1000000.json.gz", [bad]), ("1000030.json.gz", [bad, good])):  # the error trace crosses two blocks
    with gzip.open(run / "jaeger" / name, "wt") as f:
        json.dump(traces, f)
with gzip.open(run / "logs" / "frontend-1-a__frontend.log.gz", "wt") as f:
    f.write('2026-11-08T17:33:13.000000001Z {"level":"error","time":"x","message":"SearchHandler failed"}\n\n')
with gzip.open(run / "logs" / "wrk2.log.gz", "wt") as f:
    f.write("segment 1 2 3 0\n")

hgraph = Path(tempfile.mkdtemp()) / "hotel.txt"  # a small Hotel-like datagraph: geo caches in memcached-geo
hgraph.write_text("MERGE (f:Service {name: 'frontend'});\nMERGE (s:Service {name: 'search'});\nMERGE (g:Service {name: 'geo'});\n"
                  "MERGE (m:Cache {name: 'memcached-geo'});\nMERGE (f)-[:CALLS]->(s);\nMERGE (s)-[:CALLS]->(g);\nMERGE (g)-[:USES]->(m);\n")
s = typed.add_callees(typed.spans(run), hgraph)
assert len(s) == 7 and s.trace.value_counts().to_dict() == {"aaaa": 4, "bbbb": 3}  # each trace once
assert s.set_index("span").callee.to_dict() == {"aaaa": "", "c1": "search", "s1": "", "c2": "", "bbbb": "", "c3": "search", "s3": ""}
assert not s.datastore.any()  # geo never answered /geo.Geo/Nearby in this run: its callee stays unknown
row = s[s.span == "c2"].iloc[0]
assert (row.parent, row.service, row.pod, row.peer, row.error, row.status, row.error_msg) == ("s1", "search", "search-1-b", "geo", 1, "Unavailable", "no connection")
assert s[s.span == "aaaa"].iloc[0].status == "500" and s[s.span == "aaaa"].iloc[0].error == 0
assert typed._kind({}, "compose_text_client") == "client" and typed._kind({}, "compose_creator_server") == "server"  # Social: no span.kind tag
assert typed._kind({}, "/wrk2-api/post/compose") == "" and typed._kind({"span.kind": "server"}, "x_client") == "server"
assert typed._kind({}, "user_mongo_insert_cilent") == "client"  # DeathStarBench's typo

# --- callees, one rule at a time: a child; the same call answered in another trace; the caller's only cache of that kind
rows = [("t1", "r1", "", 0, "frontend", "HTTP GET /hotels", "server", 0), ("t1", "k1", "r1", 10, "frontend", "/geo.Geo/Nearby", "client", 0),
        ("t1", "g1", "k1", 20, "geo", "/geo.Geo/Nearby", "server", 0),
        ("t2", "r2", "", 30, "frontend", "HTTP GET /hotels", "server", 1), ("t2", "k2", "r2", 40, "frontend", "/geo.Geo/Nearby", "client", 1),
        ("t2", "k3", "r2", 50, "geo", "memcached_get_geo", "client", 0), ("t2", "k4", "r2", 60, "geo", "redis_get_client", "client", 0),
        ("t2", "k5", "r2", 70, "frontend", "/x.X/Y", "client", 0)]
hand = pd.DataFrame([dict(zip(["trace", "span", "parent", "start_us", "service", "operation", "kind", "error"], r), duration_us=5, pod="",
                          peer="", status="", error_msg="") for r in rows], columns=typed.SPAN_FIELDS)
hand["start_us"] += 1_000_000_000_000
c = typed.add_callees(hand, hgraph).set_index("span")
assert c.callee.to_dict() == {"r1": "", "k1": "geo", "g1": "", "r2": "", "k2": "geo", "k3": "memcached-geo", "k4": "", "k5": ""}
assert c.datastore[c.datastore].index.tolist() == ["k3", "k4"]  # k4: no redis in the datagraph, but its name still marks it

lines = typed.trace_tree(s, "aaaa")
assert len(lines) == 4 and lines[0].startswith("aaaa") and lines[3].startswith("         c2") and "ERR Unavailable" in lines[3]

edges = typed.call_edges(s).set_index(["caller", "callee"])
assert edges.loc[("frontend", "search"), "calls"] == 2 and edges.loc[("frontend", "search"), "unanswered"] == 0
assert edges.loc[("frontend", "search"), "errors"] == 1  # the callee's server span failed in one of the two calls
assert edges.loc[("search", "? /geo.Geo/Nearby"), "unanswered"] == 1  # geo never answered: only the client span exists
assert len(edges) == 2  # frontend calling itself (root to client span) is not a call between services

# A caller without a client span (home-timeline reading posts in Social): the server span hangs from its server span
reads = pd.DataFrame([{"trace": "cccc", "span": "h1", "parent": "", "service": "home-timeline-service", "kind": "server", "error": 0,
                       "operation": "read_home_timeline_server", "callee": "", "datastore": False},
                      {"trace": "cccc", "span": "p1", "parent": "h1", "service": "post-storage-service", "kind": "server", "error": 0,
                       "operation": "post_storage_read_posts_server", "callee": "", "datastore": False}], columns=typed.SPAN_FIELDS + ["callee", "datastore"])
assert typed.call_edges(reads)[["caller", "callee"]].values.tolist() == [["home-timeline-service", "post-storage-service"]]

origins = typed.error_origins(s)
assert list(origins.operation) == ["/geo.Geo/Nearby"] and list(origins.service) == ["search"]  # where the error starts
assert len(typed.error_origins(s[s.trace == "bbbb"])) == 0  # a run without errors: an empty table, not a crash

grid = np.arange(1_000_000.0, 1_000_061.0, 30)  # steps ending at 1000000, 1000030, 1000060 s
f = typed.span_features(s, grid, 30, sampling=0.5, min_spans=1).set_index(["t", "service"])
assert f.loc[(1_000_030.0, "search"), "requests"] == 1 and f.loc[(1_000_030.0, "search"), "error_share"] == 1.0
assert f.loc[(1_000_060.0, "search"), "error_share"] == 0.0 and np.isclose(f.loc[(1_000_060.0, "frontend"), "p50_ms"], 0.3)
assert np.isclose(f.loc[(1_000_030.0, "search"), "rps"], 1 / 30 / 0.5)  # the sample scaled back to the traffic
assert f.loc[(1_000_030.0, "frontend"), "requests"] == 0 and np.isnan(f.loc[(1_000_030.0, "frontend"), "p95_ms"])  # served nothing: 0
assert (1_000_000.0, "geo") not in f.index  # outside the steps the export covers whole: no made-up zeros
assert f.loc[(1_000_030.0, "search"), ["calls_in", "unanswered_share", "failed_share"]].tolist() == [1, 0.0, 1.0]  # frontend's call failed
f = typed.span_features(s, grid, 30).set_index(["t", "service"])  # default min_spans and no sampling
assert np.isnan(f.loc[(1_000_060.0, "frontend"), "p50_ms"]) and f.loc[(1_000_060.0, "frontend"), "requests"] == 1 and f.rps.isna().all()
hf = typed.span_features(typed.add_callees(hand, hgraph), grid, 30).set_index(["t", "service"]).loc[1_000_030.0]
assert hf.loc["geo", "calls_in"] == 2 and hf.loc["geo", "unanswered_share"] == 0.5  # k2 is a call to geo nobody answered
assert hf.loc["memcached-geo", "calls_in"] == 1 and np.isnan(hf.loc["memcached-geo", "unanswered_share"])  # caches never answer

# --- V5: Jaeger kept its store, the roots match what wrk2 delivered, no full block under load
def v5(roots=60, restarts=0, started="2001-09-09T01:40:00Z", full=(), sampling=1.0, seconds=60):
    attempt = Path(tempfile.mkdtemp())
    (attempt / "meta.json").write_text(json.dumps({"trace_sampling": sampling, "jaeger_full_blocks": list(full)}))
    pod = {"metadata": {"name": "jaeger-1-a"}, "status": {"containerStatuses": [{"restartCount": restarts, "state": {"running": {"startedAt": started}}}]}}
    with gzip.open(attempt / "pods.json.gz", "wt") as f:
        json.dump({"items": [pod, {"metadata": {"name": "geo-1-b"}, "status": {"containerStatuses": [{"restartCount": 3}]}}]}, f)
    spans = pd.DataFrame({"parent": [""] * roots, "start_us": (1_000_001_000 + np.arange(roots)) * 10**6 // 1})
    labels = {"t_workload": 1_000_000_900, "t_inject": 1_000_002_000, "checks": {"window": [1_000_000_900, 1_000_002_000]},
              "segments": [{"start": 1_000_001_000, "seconds": seconds, "rate": 1, "delivered": 1.0},
                           {"start": 1_000_001_060, "seconds": 20, "rate": 5, "delivered": 5.0}]}  # a short ramp: not judged
    return typed.trace_check(spans, attempt, labels)


assert v5()[0] is True and v5()[1]["worst_segment"] == (1_000_001_000, 1, 1.0)  # 1 000 000 000 s is 2001-09-09T01:46:40Z
assert v5(restarts=1)[0] is False and v5(started="2001-09-09T02:05:00Z")[1]["jaeger_kept"] is False  # a pod newer than the grid lost the store
assert v5(roots=50)[0] is False and v5(full=[1_000_001_050])[1]["full_blocks_under_load"] == [1_000_001_050]
assert v5(full=[1_000_000_000])[0] is True  # a full block outside the load (Social's graph) loses nothing that counts
assert v5(sampling=None)[0] is False
assert v5(roots=0, seconds=20) == (True, v5(roots=0, seconds=20)[1]) and v5(roots=0, seconds=20)[1]["completeness_judged"] is False

lg = typed.logs(run)
assert len(lg) == 1 and lg.iloc[0].level == "error" and lg.iloc[0].service == "frontend" and abs(lg.iloc[0].t - 1794159193.0) < 1e-3

# --- datagraph edges in both styles: Hotel declares nodes in each MATCH, Social once with short variables
graph = Path(tempfile.mkdtemp()) / "g.txt"
graph.write_text("MERGE (nw:Service {name: 'nginx-thrift'});\nMERGE (cp:Service {name: 'compose-post-service'});\n"
                 "MERGE (db:Database {name: 'post-storage-mongodb'});\nMERGE (nw)-[:CALLS]->(cp);\n"
                 "MATCH (frontend:Service {name: 'frontend'}), (s:Service {name: 'search'}) MERGE (frontend)-[:CALLS]->(s);\n"
                 "MATCH (cp:Service {name: 'compose-post-service'}), (db:Database {name: 'post-storage-mongodb'}) MERGE (cp)-[:USES]->(db);\n")
assert build.datagraph_edges(graph) == [("nginx-thrift", "CALLS", "compose-post-service"), ("frontend", "CALLS", "search"),
                                        ("compose-post-service", "USES", "post-storage-mongodb")]
assert build.datagraph_nodes(graph) == [("nginx-thrift", "Service"), ("compose-post-service", "Service"), ("post-storage-mongodb", "Database")]

print("ok")
