"""Look at the traces of a collected run in a local Jaeger, or at the Jaeger of the run being collected.

    poetry run python -m anomaly.traces pick network_delay r1     every trace from 60 s before to 120 s after the injection
    poetry run python -m anomaly.traces live                      the Jaeger of the run being collected
    poetry run python -m anomaly.traces down                      stops both and leaves nothing behind

Nothing is written to disk: the local Jaeger keeps the traces in memory and loses them when it stops. Every port listens on
127.0.0.1 only, reached through an SSH tunnel. The copy in the viewer masks the passwords of the URLs (the raw files keep them).
"""
import argparse
import getpass
import gzip
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from anomaly.build import last_valid_attempt

VIEWER, IMAGE = "jaeger-viewer", "jaegertracing/all-in-one:1.57"  # the collected runs' Jaeger version
UI, OTLP, LIVE = 16690, 14318, 16687  # never 16686: the collector port-forwards its Jaeger there at readiness and export
MAX_SPANS = 150_000  # in its 1 GB: 235k Hotel spans took 590 MB at rest and 1009 MB in a search, at the edge of the limit
SEARCH_LIMIT = 1400  # the UI shows the newest 1500 traces of a search; a trace whose spans cross a cut can show in both searches
BATCH_SPANS = 2000
PASSWORD = re.compile(r"(password=)[^&]*")
LIVE_LOOP = (f"while true; do ns=$(kubectl get svc -A --field-selector metadata.name=jaeger -o jsonpath='{{.items[0].metadata.namespace}}' "
             f"2>/dev/null) && kubectl port-forward -n \"$ns\" svc/jaeger {LIVE}:16686; sleep 15; done")  # each run brings a new cluster


def find_runs(root, words):
    """Run directories under root (<app>/<kind>/<run>) whose path holds every word; a word rN is the repetition, exactly."""
    runs = sorted(d for d in Path(root).glob("*/*/*") if d.is_dir() and any(d.glob("attempt-*")))
    def match(path):
        name = str(path.relative_to(root))
        return all(name.endswith(f"__{w}") if re.fullmatch(r"r\d+", w) else w in name for w in words)
    return [d for d in runs if match(d)]


def blocks_in(attempt, t0, t1):
    """The export blocks (named by their start, in seconds) that overlap [t0, t1]: a block ends where the next one starts."""
    blocks = sorted((attempt / "jaeger").glob("*.json.gz"), key=lambda p: int(p.name.split(".")[0]))
    starts = [int(p.name.split(".")[0]) for p in blocks] + [float("inf")]
    return [p for p, s, e in zip(blocks, starts, starts[1:]) if s < t1 and e > t0]


def traces_in(attempt, t0, t1):
    """Every trace that starts in [t0, t1], once: a trace that crosses two blocks comes back in both."""
    seen, out = set(), []
    for block in blocks_in(attempt, t0, t1):
        with gzip.open(block, "rt") as f:
            for trace in json.load(f):
                if trace["traceID"] not in seen and t0 <= min(s["startTime"] for s in trace["spans"]) / 1e6 <= t1:
                    seen.add(trace["traceID"])
                    out.append(trace)
    return out


def _value(tag):
    kind = tag.get("type", "string")
    if kind == "bool":
        return {"boolValue": tag["value"]}
    if kind == "int64":
        return {"intValue": str(tag["value"])}
    if kind == "float64":
        return {"doubleValue": tag["value"]}
    if kind == "binary":
        return {"bytesValue": tag["value"]}
    value = str(tag["value"])
    return {"stringValue": PASSWORD.sub(r"\1***", value) if tag["key"] == "http.url" else value}


def _attributes(tags):
    return [{"key": t["key"], "value": _value(t)} for t in tags]


KINDS = {"internal": 1, "server": 2, "client": 3, "producer": 4, "consumer": 5}


def to_otlp(traces):
    """The traces as an OTLP/JSON body, as Jaeger exported them: span.kind and error become the OTLP kind and status (Jaeger turns
    them back into the same tags), the first CHILD_OF the parent, the logs the events; internal.span.format is Jaeger's own."""
    resource_spans = []
    for trace in traces:
        by_process = {}
        for s in trace["spans"]:
            tags = [t for t in s.get("tags", []) if t["key"] != "internal.span.format"]
            kind = next((t["value"] for t in tags if t["key"] == "span.kind"), "")
            failed = any(t["key"] == "error" and str(t["value"]).lower() == "true" for t in tags)
            tags = [t for t in tags if t["key"] != "span.kind" and not (t["key"] == "error" and failed)]
            span = {"traceId": s["traceID"].rjust(32, "0"), "spanId": s["spanID"].rjust(16, "0"), "name": s["operationName"],
                    "kind": KINDS.get(kind, 0), "startTimeUnixNano": str(s["startTime"] * 1000),
                    "endTimeUnixNano": str((s["startTime"] + s["duration"]) * 1000), "attributes": _attributes(tags),
                    "events": [{"timeUnixNano": str(log["timestamp"] * 1000),
                                "name": next((f["value"] for f in log["fields"] if f["key"] == "event"), ""),
                                "attributes": _attributes([f for f in log["fields"] if f["key"] != "event"])} for log in s.get("logs", [])]}
            parent = next((r["spanID"] for r in s.get("references", []) if r["refType"] == "CHILD_OF"), "")
            if parent:
                span["parentSpanId"] = parent.rjust(16, "0")
            if failed:
                span["status"] = {"code": 2}
            by_process.setdefault(s["processID"], []).append(span)
        for pid, spans in by_process.items():
            process = trace["processes"][pid]
            resource = [{"key": "service.name", "value": {"stringValue": process["serviceName"]}}] + _attributes(process.get("tags", []))
            resource_spans.append({"resource": {"attributes": resource}, "scopeSpans": [{"spans": spans}]})
    return {"resourceSpans": resource_spans}


def _post(body):
    request = urllib.request.Request(f"http://127.0.0.1:{OTLP}/v1/traces", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()


def _docker(*args, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout.strip()


def start_viewer(keep):
    """The local Jaeger, empty unless keep and already running. Memory only, capped in CPU and memory: a collection may be running.

    Its queue holds every span of a pick (the default 2000 dropped 2% of a 47k-span pick, and the OTLP answer is 200 all the same),
    and it does not trace itself.
    """
    if keep and _docker("ps", "-q", "-f", f"name=^{VIEWER}$"):
        return
    _docker("rm", "-f", VIEWER, check=False)
    _docker("run", "-d", "--rm", "--name", VIEWER, "--cpus", "0.5", "--memory", "1g", "-e", "COLLECTOR_OTLP_ENABLED=true",
            "-e", f"COLLECTOR_QUEUE_SIZE={MAX_SPANS}", "-e", "OTEL_TRACES_SAMPLER=always_off", "-e", "GOMEMLIMIT=800MiB",
            "-p", f"127.0.0.1:{UI}:16686", "-p", f"127.0.0.1:{OTLP}:4318", IMAGE)
    for _ in range(60):
        try:
            return _post({"resourceSpans": []})
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"the local Jaeger did not start: docker logs {VIEWER}")


def stored():
    """(spans saved, spans dropped) by the local Jaeger since it started, from its own metrics."""
    metrics = _docker("exec", VIEWER, "wget", "-qO-", "localhost:14269/metrics")
    count = lambda pattern: sum(float(line.split()[-1]) for line in metrics.splitlines() if re.match(pattern, line))
    return count(r'jaeger_collector_spans_saved_by_svc_total\{.*result="ok"'), count(r"jaeger_collector_spans_dropped_total")


def search_links(service, starts_us, t_inject):
    """(from, to in seconds from the injection, link) of the searches that together show every trace, each up to SEARCH_LIMIT."""
    starts = sorted(starts_us)
    cuts = starts[::SEARCH_LIMIT] + [starts[-1] + 1]
    return [(a / 1e6 - t_inject, b / 1e6 - t_inject,
             f"http://localhost:{UI}/search?service={service}&start={a}&end={b - 1}&lookback=custom&limit=1500") for a, b in zip(cuts, cuts[1:])]


def tunnel(*ports):
    server = (os.environ.get("SSH_CONNECTION") or "").split()[2:3] or ["<server>"]
    return "ssh -N " + " ".join(f"-L {p}:localhost:{p}" for p in ports) + f" {getpass.getuser()}@{server[0]}"


def pick(args):
    root = Path(args.root).expanduser()
    runs = find_runs(root, args.run)
    if len(runs) != 1:
        print(f"{len(runs)} runs under {root} match {' '.join(args.run)}" + (": add a word" if runs else ""))
        for d in runs:
            print("  ", d.relative_to(root))
        raise SystemExit(1)
    attempt = last_valid_attempt(runs[0])
    if attempt is None:
        raise SystemExit(f"{runs[0].relative_to(root)}: no valid attempt")
    t_inject = json.loads((attempt / "meta.json").read_text())["run_meta"]["t_inject"]
    t0, t1 = t_inject + args.start, t_inject + args.end
    traces = traces_in(attempt, t0, t1)
    spans = sum(len(t["spans"]) for t in traces)
    if not traces:
        raise SystemExit(f"{runs[0].relative_to(root)}/{attempt.name}: no trace from {args.start:+g} to {args.end:+g} s around the injection")
    if spans > MAX_SPANS:
        raise SystemExit(f"{len(traces)} traces, {spans} spans: over the {MAX_SPANS} the local Jaeger holds, take a shorter window")
    start_viewer(args.add)
    saved0, dropped0 = stored()
    if saved0 + spans > MAX_SPANS:
        raise SystemExit(f"the local Jaeger already holds {saved0:.0f} spans, {spans} more go over {MAX_SPANS}: pick without --add")
    batch, size = [], 0
    for trace in traces:
        batch.append(trace)
        size += len(trace["spans"])
        if size >= BATCH_SPANS:
            _post(to_otlp(batch))
            batch, size = [], 0
    if batch:
        _post(to_otlp(batch))
    for _ in range(120):  # the OTLP answer comes before the spans are stored
        saved, dropped = stored()
        if saved - saved0 + dropped - dropped0 >= spans:
            break
        time.sleep(0.5)
    if dropped > dropped0 or saved - saved0 < spans:
        raise SystemExit(f"the local Jaeger stored {saved - saved0:.0f} of {spans} spans ({dropped - dropped0:.0f} dropped): not every trace is there")
    roots = [next(t["processes"][s["processID"]]["serviceName"] for s in t["spans"] if not s.get("references")) for t in traces
             if any(not s.get("references") for s in t["spans"])]
    service = max(set(roots), key=roots.count) if roots else ""
    print(f"{runs[0].relative_to(root)}/{attempt.name}: {len(traces)} traces, {spans} spans, from {args.start:+g} to {args.end:+g} s around the injection")
    links = search_links(service, [min(s["startTime"] for s in t["spans"]) for t in traces], t_inject)
    if len(links) > 1:
        print(f"  Jaeger shows at most 1500 traces a search: these {len(links)} searches show them all, in seconds from the injection")
    for a, b, link in links:
        print(f"  {a:+7.1f} .. {b:+7.1f} s  open  {link}" if len(links) > 1 else f"  open  {link}")
    print(f"  tunnel (on your laptop, if not open yet)  {tunnel(UI)}")


def live(_):
    if not subprocess.run(["pgrep", "-f", f"{LIVE}:16686"], capture_output=True).stdout:
        subprocess.Popen(["bash", "-c", LIVE_LOOP], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  open  http://localhost:{LIVE}  (the run being collected; it comes back about 15 s after each new cluster)")
    print(f"  tunnel (on your laptop, if not open yet)  {tunnel(LIVE)}")


def down(_):
    viewer = _docker("rm", "-f", VIEWER, check=False)
    forward = subprocess.run(["pkill", "-f", f"{LIVE}:16686"]).returncode == 0
    print(f"local Jaeger {'stopped, its traces gone' if viewer else 'was not running'}; live link {'stopped' if forward else 'was not running'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="poetry run python -m anomaly.traces", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    q = sub.add_parser("pick", help="load every trace of a window of a collected run into the local Jaeger")
    q.add_argument("run", nargs="+", help="words of the run's path, e.g. network_delay r1, or social noop r6")
    q.add_argument("--from", dest="start", type=float, default=-60, help="seconds from the injection (default -60)")
    q.add_argument("--to", dest="end", type=float, default=120, help="seconds from the injection (default 120)")
    q.add_argument("--add", action="store_true", help="keep the traces already loaded, e.g. a clean run next to a faulty one")
    q.add_argument("--root", default="~/datasets/full", help="where the runs are (default ~/datasets/full)")
    q.set_defaults(func=pick)
    sub.add_parser("live", help="link to the Jaeger of the run being collected").set_defaults(func=live)
    sub.add_parser("down", help="stop the local Jaeger and the live link").set_defaults(func=down)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
