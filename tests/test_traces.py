# Run from the repo root: PYTHONPATH=. python tests/test_traces.py  (no cluster, no docker, no data needed)
import gzip
import json
from pathlib import Path
import tempfile

from anomaly import traces

# --- one Hotel trace as Jaeger exports it: a failing gRPC client call, and a Social-like span with no span.kind
trace = {"traceID": "1a2b3c4d5e6f7081", "processes": {"p1": {"serviceName": "frontend", "tags": [{"key": "hostname", "type": "string", "value": "frontend-x"}]},
                                                     "p2": {"serviceName": "user", "tags": []}},
         "spans": [
             {"traceID": "1a2b3c4d5e6f7081", "spanID": "1a2b3c4d5e6f7081", "operationName": "HTTP POST /reservation", "references": [],
              "startTime": 1000, "duration": 50, "processID": "p1", "logs": [],
              "tags": [{"key": "span.kind", "type": "string", "value": "server"}, {"key": "http.status_code", "type": "int64", "value": 200},
                       {"key": "http.url", "type": "string", "value": "http://x/reservation?username=Cornell_1&password=111&number=1"},
                       {"key": "sampler.param", "type": "bool", "value": True}, {"key": "internal.span.format", "type": "string", "value": "proto"}]},
             {"traceID": "1a2b3c4d5e6f7081", "spanID": "aa", "operationName": "/user.User/CheckUser", "processID": "p1",
              "references": [{"refType": "CHILD_OF", "traceID": "1a2b3c4d5e6f7081", "spanID": "1a2b3c4d5e6f7081"}], "startTime": 1001, "duration": 40,
              "tags": [{"key": "span.kind", "type": "string", "value": "client"}, {"key": "error", "type": "bool", "value": True}],
              "logs": [{"timestamp": 1041, "fields": [{"key": "event", "type": "string", "value": "error"}, {"key": "message", "type": "string", "value": "Unavailable"}]}]},
             {"traceID": "1a2b3c4d5e6f7081", "spanID": "bb", "operationName": "check_user_server", "processID": "p2",
              "references": [{"refType": "CHILD_OF", "traceID": "1a2b3c4d5e6f7081", "spanID": "aa"}], "startTime": 1002, "duration": 15,
              "tags": [{"key": "ratio", "type": "float64", "value": 0.5}], "logs": []}]}
body = traces.to_otlp([trace])
by = {s["spanId"][-2:]: (r["resource"]["attributes"], s) for r in body["resourceSpans"] for s in r["scopeSpans"][0]["spans"]}
res, root = by["81"]
assert len(body["resourceSpans"]) == 2 and res[0] == {"key": "service.name", "value": {"stringValue": "frontend"}}
assert root["traceId"] == "0000000000000000" + "1a2b3c4d5e6f7081" and "parentSpanId" not in root and root["kind"] == 2
assert root["startTimeUnixNano"] == "1000000" and root["endTimeUnixNano"] == "1050000"
attrs = {a["key"]: a["value"] for a in root["attributes"]}
assert attrs == {"http.status_code": {"intValue": "200"}, "sampler.param": {"boolValue": True},  # bool from the tag's type, not Python's
                 "http.url": {"stringValue": "http://x/reservation?username=Cornell_1&password=***&number=1"}}
_, call = by["aa"]
assert call["spanId"] == "00000000000000aa" and call["parentSpanId"] == "1a2b3c4d5e6f7081" and call["kind"] == 3
assert call["status"] == {"code": 2} and call["attributes"] == []  # error is the status: Jaeger turns it back into error=true
assert call["events"] == [{"timeUnixNano": "1041000", "name": "error", "attributes": [{"key": "message", "value": {"stringValue": "Unavailable"}}]}]
_, social = by["bb"]
assert social["kind"] == 0 and "status" not in social and social["attributes"] == [{"key": "ratio", "value": {"doubleValue": 0.5}}]

# --- runs by words of their path (rN exactly), blocks and traces of a window, a trace crossing two blocks counted once
root = Path(tempfile.mkdtemp())
for run in ("hotel/test/network_delay_hotel_res__r1", "hotel/test/network_delay_hotel_res__r10", "social/clean/noop_social_network__r1"):
    (root / run / "attempt-1").mkdir(parents=True)
(root / "hotel/test/empty_run").mkdir(parents=True)  # no attempt: not a run
names = lambda words: [str(d.relative_to(root)) for d in traces.find_runs(root, words)]
assert names(["network_delay", "r1"]) == ["hotel/test/network_delay_hotel_res__r1"]
assert names(["delay"]) == ["hotel/test/network_delay_hotel_res__r1", "hotel/test/network_delay_hotel_res__r10"]
assert names(["social", "r1"]) == ["social/clean/noop_social_network__r1"] and names(["empty"]) == []

attempt = root / "hotel/test/network_delay_hotel_res__r1/attempt-1"
(attempt / "jaeger").mkdir()
t = lambda tid, start_s: {"traceID": tid, "processes": {}, "spans": [{"startTime": int(start_s * 1e6)}, {"startTime": int((start_s + 1) * 1e6)}]}
for start, block in ((100, [t("a", 105)]), (110, [t("a", 105), t("b", 115)]), (120, [t("c", 125)]), (130, [t("d", 135)])):
    with gzip.open(attempt / "jaeger" / f"{start}.json.gz", "wt") as f:
        json.dump(block, f)
assert [p.name for p in traces.blocks_in(attempt, 112, 121)] == ["110.json.gz", "120.json.gz"]
assert [p.name for p in traces.blocks_in(attempt, 131, 999)] == ["130.json.gz"]  # the last block reaches the end
assert [x["traceID"] for x in traces.traces_in(attempt, 100, 120)] == ["a", "b"]  # a once; c starts after the window

# --- searches of at most SEARCH_LIMIT traces that cover every trace, cut on the start times
starts = list(range(1_000_000, 1_000_000 + 3001))  # 3001 traces, one per microsecond
links = traces.search_links("frontend", starts, 1.0)
assert len(links) == 3 and [l[2].split("&lookback")[0].split("start=")[1] for l in links] == ["1000000&end=1001399", "1001400&end=1002799", "1002800&end=1003000"]
one = traces.search_links("nginx-thrift", [5_000_000], 4.0)
assert len(one) == 1 and one[0][0] == 1.0 and "service=nginx-thrift&start=5000000&end=5000000&" in one[0][2]
print("ok")
