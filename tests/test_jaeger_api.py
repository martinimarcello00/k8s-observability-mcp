# Run from the repo root: PYTHONPATH=. python tests/test_jaeger_api.py  (no cluster/Jaeger needed)
from types import SimpleNamespace as NS
import api.jaeger_api as jaeger_api
from api.jaeger_api import JaegerAPI, is_stream_trace

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
# flagd EventStream subscription (Envoy span carries it in http.url, its egress span has no marker): dropped
stream = {"spans": [{"operationName": "POST", "startTime": 1, "tags": [{"key": "http.url", "value": "http://frontend-proxy:8080/flagservice/flagd.evaluation.v1.Service/EventStream"}]},
                    {"operationName": "router flagservice egress", "startTime": 1, "tags": []}]}
assert is_stream_trace(stream) and not is_stream_trace(ot) and not is_stream_trace(otel)

# Page full of stream traces: search again per operation and return the real ones
def fake_get(url, params=None):
    if url.endswith("/operations"): data = ["POST", "router frontend egress"]
    elif params.get("operation") == "router frontend egress": data = [ot]
    else: data = [{**stream, "traceID": str(i)} for i in range(params["limit"])]
    return NS(raise_for_status=lambda: None, json=lambda: {"data": data})
jaeger_api.requests.get, j.jaeger_url = fake_get, "http://jaeger"
assert [t["traceID"] for t in j.get_jaeger_traces("frontend-proxy", limit=3)] == ["t1"]
print("ok")
