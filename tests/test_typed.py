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

s = typed.spans(run)
assert len(s) == 7 and s.trace.value_counts().to_dict() == {"aaaa": 4, "bbbb": 3}  # each trace once
row = s[s.span == "c2"].iloc[0]
assert (row.parent, row.service, row.pod, row.peer, row.error, row.status, row.error_msg) == ("s1", "search", "search-1-b", "geo", 1, "Unavailable", "no connection")
assert s[s.span == "aaaa"].iloc[0].status == "500" and s[s.span == "aaaa"].iloc[0].error == 0
assert typed._kind({}, "compose_text_client") == "client" and typed._kind({}, "compose_creator_server") == "server"  # Social: no span.kind tag
assert typed._kind({}, "/wrk2-api/post/compose") == "" and typed._kind({"span.kind": "server"}, "x_client") == "server"

lines = typed.trace_tree(s, "aaaa")
assert len(lines) == 4 and lines[0].startswith("aaaa") and lines[3].startswith("         c2") and "ERR Unavailable" in lines[3]

edges = typed.call_edges(s).set_index(["caller", "callee"])
assert edges.loc[("frontend", "search"), "calls"] == 2 and edges.loc[("frontend", "search"), "unanswered"] == 0
assert edges.loc[("frontend", "search"), "errors"] == 1  # the callee's server span failed in one of the two calls
assert edges.loc[("search", "? /geo.Geo/Nearby"), "unanswered"] == 1  # geo never answered: only the client span exists
assert len(edges) == 2  # frontend calling itself (root to client span) is not a call between services

# A caller without a client span (home-timeline reading posts in Social): the server span hangs from its server span
reads = pd.DataFrame([{"trace": "cccc", "span": "h1", "parent": "", "service": "home-timeline-service", "kind": "server", "error": 0,
                       "operation": "read_home_timeline_server"},
                      {"trace": "cccc", "span": "p1", "parent": "h1", "service": "post-storage-service", "kind": "server", "error": 0,
                       "operation": "post_storage_read_posts_server"}], columns=typed.SPAN_FIELDS)
assert typed.call_edges(reads)[["caller", "callee"]].values.tolist() == [["home-timeline-service", "post-storage-service"]]

origins = typed.error_origins(s)
assert list(origins.operation) == ["/geo.Geo/Nearby"] and list(origins.service) == ["search"]  # where the error starts
assert len(typed.error_origins(s[s.trace == "bbbb"])) == 0  # a run without errors: an empty table, not a crash

grid = np.arange(1_000_000.0, 1_000_061.0, 30)  # steps ending at 1000000, 1000030, 1000060 s
f = typed.span_features(s, grid, 30).set_index(["t", "service"])
assert f.loc[(1_000_030.0, "search"), "requests"] == 1 and f.loc[(1_000_030.0, "search"), "error_share"] == 1.0
assert f.loc[(1_000_060.0, "search"), "error_share"] == 0.0 and np.isclose(f.loc[(1_000_060.0, "frontend"), "p50_ms"], 0.3)

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

print("ok")
