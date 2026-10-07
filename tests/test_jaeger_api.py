# Run from the repo root: PYTHONPATH=. python tests/test_jaeger_api.py  (no cluster/Jaeger needed)
import json
import os
from types import SimpleNamespace as NS
import api.jaeger_api as jaeger_api
from api.base_k8s_client import MECHANISM
from api.jaeger_api import JaegerAPI, strip_flagd

j = JaegerAPI.__new__(JaegerAPI)  # skip k8s/Jaeger init, process_trace is pure

# OpenTracing (hotel/social): root span without references, error in event=error logs
ot = {"traceID": "t1", "processes": {"p1": {"serviceName": "frontend"}, "p2": {"serviceName": "geo"}},
      "spans": [
          {"processID": "p1", "operationName": "op", "startTime": 100, "duration": 5000, "references": [], "tags": []},
          {"processID": "p2", "operationName": "op", "startTime": 200, "duration": 1000, "references": [{"refType": "CHILD_OF"}],
           "tags": [{"key": "error", "value": True}],
           "logs": [{"fields": [{"key": "event", "value": "error"}, {"key": "message", "value": "boom"},
                                {"key": "stack", "value": "line1\nline2"}]}]}]}
assert j.process_trace(ot) == {"traceID": "t1", "latency_ms": 5.0, "has_error": True,
                               "sequence": "frontend -> geo", "error_message": "boom; line1"}

# OTel (Astronomy Shop): no root span in Jaeger, error in otel.status_description / event=exception, wrapped upstream
otel = {"traceID": "t2", "processes": {"p1": {"serviceName": "checkout"}, "p2": {"serviceName": "payment"}},
        "spans": [
            {"processID": "p1", "operationName": "op", "startTime": 1000, "duration": 3000, "references": [{"refType": "CHILD_OF"}],
             "tags": [{"key": "error", "value": True},
                      {"key": "otel.status_description", "value": "could not charge: Invalid token."}]},
            {"processID": "p2", "operationName": "op", "startTime": 1500, "duration": 4500, "references": [{"refType": "CHILD_OF"}],
             "tags": [{"key": "error", "value": True}, {"key": "otel.status_description", "value": "Invalid token."}],
             "logs": [{"fields": [{"key": "event", "value": "exception"}, {"key": "exception.message", "value": "Invalid token."},
                                  {"key": "exception.stacktrace", "value": "Error: Invalid token.\n  at charge"}]}]}]}
assert j.process_trace(otel) == {"traceID": "t2", "latency_ms": 5.0, "has_error": True,
                                 "sequence": "checkout -> payment", "error_message": "could not charge: Invalid token."}

# No flag in it (Hotel Reservation, Social Network): the very same trace back
assert strip_flagd(ot) is ot and strip_flagd(otel) is otel

# flagd EventStream subscription (Envoy span carries it in http.url, its egress span in the operation name): dropped
stream = {"traceID": "s", "processes": {"p1": {"serviceName": "frontend-proxy"}},
          "spans": [{"processID": "p1", "operationName": "POST", "startTime": 1, "tags": [{"key": "http.url", "value": "http://frontend-proxy:8080/flagservice/flagd.evaluation.v1.Service/EventStream"}]},
                    {"processID": "p1", "operationName": "router flagservice egress", "startTime": 1, "tags": []}]}
assert strip_flagd(stream) is None

# Astronomy Shop, real traces from Jaeger (demo 3.1.0 on the kind cluster, 2026-10-01, flags injected as AIOpsLab does).
# Nothing about flagd or feature flags may be left; the real signal (sequence, latency, errors) stays.
with open(os.path.join(os.path.dirname(__file__), "astronomy_shop_traces.json")) as f:
    real = json.load(f)
for name, trace in real.items():
    after = strip_flagd(trace)
    assert after is None or not MECHANISM.search(json.dumps(after)), name
    if after:
        before, now = j.process_trace(trace), j.process_trace(after)
        hops = iter(before["sequence"].split(" -> "))
        assert all(hop in hops for hop in now["sequence"].split(" -> ")), name  # same order, only hops removed
        assert (now["latency_ms"], now["has_error"]) == (before["latency_ms"], before["has_error"]), name

# Flag plumbing only (EventStream, flag lookups answered by flagd, lookups failing while flagd restarts): the trace goes
for name in ("eventstream", "flagd_server_spans", "cart_flagd_outage"):
    assert strip_flagd(real[name]) is None, name
# A checkout flow that ends with cart looking up a flag on flagd: only those 5 flag spans go, and with them the last
# two hops (cart is there only for the lookup)
flow = real["business_trace_through_flagd"]
assert (len(flow["spans"]), len(strip_flagd(flow)["spans"])) == (51, 46)
assert j.process_trace(flow)["sequence"].endswith(" -> shipping -> cart -> flagd")
assert j.process_trace(strip_flagd(flow))["sequence"] == j.process_trace(flow)["sequence"].removesuffix(" -> cart -> flagd")
# productCatalogFailure on: same 19 spans, the flag events/attributes go, the error stays without the announcement
catalog = strip_flagd(real["product_catalog_failure"])
assert len(catalog["spans"]) == 19
assert j.process_trace(real["product_catalog_failure"])["error_message"] == "13 INTERNAL: Error: Product Catalog Fail Feature Flag Enabled"
assert j.process_trace(catalog)["error_message"] == "13 INTERNAL: Error: Product Catalog Fail"
# ad crash-looping (adFailure, adManualGc): frontend can't reach it, nothing about flags in it, untouched
assert strip_flagd(real["ad_unavailable"]) is real["ad_unavailable"]

# Page full of stream traces: search again per operation and return the real ones
def fake_get(url, params=None):
    if url.endswith("/operations"): data = ["POST", "router frontend egress"]
    elif params.get("operation") == "router frontend egress": data = [ot]
    else: data = [{**stream, "traceID": str(i)} for i in range(params["limit"])]
    return NS(raise_for_status=lambda: None, json=lambda: {"data": data})
jaeger_api.requests.get, j.jaeger_url = fake_get, "http://jaeger"
assert [t["traceID"] for t in j.get_jaeger_traces("frontend-proxy", limit=3)] == ["t1"]

# Jaeger ignores lookback: with JAEGER_START_US set the query carries start/end, without it neither
seen = []
jaeger_api.requests.get = lambda url, params=None: seen.append(params) or NS(raise_for_status=lambda: None, json=lambda: {"data": []})
j.get_jaeger_traces("frontend-proxy")
assert "start" not in seen[-1] and "end" not in seen[-1]
os.environ["JAEGER_START_US"] = "1000"
j.get_jaeger_traces("frontend-proxy")
assert seen[-1]["start"] == "1000" and int(seen[-1]["end"]) > 1000
del os.environ["JAEGER_START_US"]
print("ok")
