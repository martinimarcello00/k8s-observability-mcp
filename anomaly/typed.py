"""Typed tables from the raw traces and logs of a run, and the analyses the notebooks build on them.

One row per span and one row per log line. The raw files under the run stay the source of truth:
these tables are derived and can be rebuilt with more columns at any time.
"""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPAN_FIELDS = ["trace", "span", "parent", "start_us", "duration_us", "service", "pod", "operation", "kind", "peer",
               "error", "status", "error_msg"]
LOG_FIELDS = ["t", "service", "pod", "container", "previous", "level", "source", "message"]


# --- spans

def _tags(obj):
    return {t["key"]: t["value"] for t in obj.get("tags", [])}


def _kind(tags, operation):
    """client or server. Social's C++ services set no span.kind tag, but name every operation ..._client or ..._server."""
    if tags.get("span.kind"):
        return tags["span.kind"]
    return "client" if operation.endswith("_client") else "server" if operation.endswith("_server") else ""


def _span_row(trace, s):
    tags, process = _tags(s), trace["processes"][s["processID"]]
    message = next((f["value"] for log in s.get("logs", []) for f in log["fields"]
                    if f["key"] in ("message", "error.object") and f["value"]), "")
    return {"trace": s["traceID"], "span": s["spanID"],
            "parent": next((r["spanID"] for r in s.get("references", []) if r["refType"] == "CHILD_OF"), ""),
            "start_us": s["startTime"], "duration_us": s["duration"], "service": process["serviceName"],
            "pod": _tags(process).get("hostname", ""), "operation": s["operationName"], "kind": _kind(tags, s["operationName"]),
            "peer": tags.get("peer.service") or tags.get("peer.hostname") or tags.get("peer.address") or "",
            "error": int(str(tags.get("error", "")).lower() == "true"),
            "status": str(tags.get("http.status_code", tags.get("response_code", ""))), "error_msg": message}


SHARED = ("service", "pod", "operation", "kind", "peer", "status", "error_msg")  # few distinct values in a run


def spans(run_dir):
    """Every span exported for the run, once: a trace that crosses two export blocks comes back in both.

    Column by column, one string object per distinct value: a dict per span, with its own copy of every string,
    took about 1 kB a span, over 5 GB for a 20-min Social session traced at 100%.
    """
    columns, seen, strings = {f: [] for f in SPAN_FIELDS}, set(), {}
    for block in sorted((Path(run_dir) / "jaeger").glob("*.json.gz")):
        with gzip.open(block, "rt") as f:
            for trace in json.load(f):
                if trace["traceID"] in seen:
                    continue
                seen.add(trace["traceID"])
                ids = {}  # the trace, span and parent ids of this trace
                for s in trace["spans"]:
                    row = _span_row(trace, s)
                    for field in ("trace", "span", "parent"):
                        row[field] = ids.setdefault(row[field], row[field])
                    for field in SHARED:
                        row[field] = strings.setdefault(row[field], row[field])
                    for field, value in row.items():
                        columns[field].append(value)
    return pd.DataFrame(columns, columns=SPAN_FIELDS)


# --- logs

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LEVELS = {"trc": "debug", "trace": "debug", "dbg": "debug", "debug": "debug", "d": "debug", "-": "debug", ".": "debug",
          "inf": "info", "info": "info", "notice": "info", "i": "info", "*": "info",
          "wrn": "warning", "warn": "warning", "warning": "warning", "w": "warning", "#": "warning",
          "err": "error", "error": "error", "e": "error",
          "ftl": "fatal", "fatal": "fatal", "pnc": "fatal", "panic": "fatal", "crit": "fatal", "critical": "fatal",
          "alert": "fatal", "emerg": "fatal", "f": "fatal", "thrift": "error"}
# (pattern, level group, source group, message group), tried in order on the line without its kubectl timestamp
FORMATS = [
    # DeathStarBench C++: [date] <level>: (file:line:func) msg; the function may hold parentheses, e.g. operator()
    (re.compile(r"^\[[^\]]+\] <(\w+)>: \((.*?)\) (.*)$"), 1, 2, 3),
    (re.compile(r"^\S+ (TRC|DBG|INF|WRN|ERR|FTL|PNC) (\S+) > (.*)$"), 1, 2, 3),  # zerolog console: time LVL file:line > msg
    (re.compile(r"^\d+:\w \d\d \w{3} \d{4} [\d:.]+ ([.*#-]) ()(.*)$"), 1, 2, 3),  # Redis: pid:role date sign msg
    (re.compile(r"^\S+ \[(\w+)\]\s+(?:([\w.-]+): )?(.*)$"), 1, 2, 3),  # consul (hclog): time [LEVEL] module: msg
    (re.compile(r"^\d{4}/\d\d/\d\d [\d:]+ \[(\w+)\] \S+ (?:\*\d+ )?(?:\[lua\] )?(\S+:\d+:)? ?(.*)$"), 1, 2, 3),  # nginx error log
    # Thrift's own output, e.g. "Thrift: Mon Nov  9 08:28:11 2026 TSocket::open() connect() <Host: text-service ...": failures only
    (re.compile(r"^(Thrift): \w{3} \w{3} +\d+ [\d:]+ \d{4} (\S+) (.*)$"), 1, 2, 3),
]


def parse_log_line(content):
    """(level, source, message) of one log line in any of the formats seen in Hotel and Social; level '' when unknown."""
    content = ANSI.sub("", content).strip()
    if content.startswith("{"):
        try:
            entry = json.loads(content)
        except ValueError:
            entry = None
        if isinstance(entry, dict):  # zerolog, zap (Jaeger) and MongoDB structured logs
            level = str(entry.get("level", entry.get("s", ""))).lower()
            level = LEVELS.get(level, LEVELS.get(level[:1], "")) if level else ""
            source = str(entry.get("caller", entry.get("c", "")))
            message = str(entry.get("message", entry.get("msg", "")))
            return level, source.strip(), message.strip()
    for pattern, lg, sg, mg in FORMATS:
        m = pattern.match(content)
        if m:
            return LEVELS.get(m.group(lg).lower(), ""), (m.group(sg) or "").rstrip(":"), m.group(mg).strip()
    if content.startswith(("panic:", "fatal error:")):
        return "fatal", "", content
    return "", "", content


def logs(run_dir, service_of=lambda pod: pod.rsplit("-", 2)[0]):
    """Every container log line of the run (wrk2's excluded: it is the load, parsed by the builder), with kubectl's timestamp."""
    rows = []
    for path in sorted((Path(run_dir) / "logs").glob("*.log.gz")):
        if path.name == "wrk2.log.gz":
            continue
        name = path.name[:-len(".log.gz")]
        previous = name.endswith(".previous")
        pod, container = name.removesuffix(".previous").split("__", 1)
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                stamp, _, content = line.rstrip("\n").partition(" ")
                if not content.strip():
                    continue
                level, source, message = parse_log_line(content)
                rows.append({"t": stamp, "service": service_of(pod), "pod": pod, "container": container, "previous": int(previous),
                             "level": level, "source": source, "message": message})
    frame = pd.DataFrame(rows, columns=LOG_FIELDS)
    frame["t"] = pd.to_datetime(frame["t"], utc=True, format="ISO8601").map(lambda x: x.timestamp()) if len(frame) else frame["t"]
    return frame


# --- analyses on the span table

def span_features(spans, grid, step):
    """Per grid step (t - step, t] and service: requests served, p50 and p95 latency in ms, error share.

    A request served = a server span, or a root span (Social's nginx tags no span kind). The exported traces are a sample
    (Jaeger sampling, then at most 5 traces/s kept by the export): requests count the sample, not the traffic.
    """
    server = spans[(spans.kind == "server") | (spans.parent == "")].copy()
    cell = np.searchsorted(grid, server.start_us / 1e6, side="left")
    ok = (cell < len(grid)) & (grid[np.minimum(cell, len(grid) - 1)] - server.start_us / 1e6 < step)
    server = server[ok].assign(t=grid[cell[ok]], ms=lambda d: d.duration_us / 1000)
    return server.groupby(["t", "service"]).agg(requests=("span", "size"), p50_ms=("ms", "median"),
                                                 p95_ms=("ms", lambda v: v.quantile(0.95)), error_share=("error", "mean")).reset_index()


def trace_tree(spans, trace_id):
    """Indented lines of one trace, children under their parent, in start order."""
    t = spans[spans.trace == trace_id].sort_values("start_us")
    children = t.groupby("parent")["span"].apply(list).to_dict()
    by_id = t.set_index("span")
    lines = []

    def walk(span_id, depth):
        r = by_id.loc[span_id]
        mark = f"  ERR {r.status} {r.error_msg[:70]}" if r.error else (f"  {r.status}" if r.status else "")
        lines.append(f"{'   ' * depth}{span_id[:8]}  {r.service:<22} {r.kind:<6} {r.duration_us:>8} µs  {r.operation}{mark}")
        for child in children.get(span_id, []):
            walk(child, depth + 1)

    for root in t[~t.parent.isin(t.span)].span:  # no parent, or a parent the export did not keep
        walk(root, 0)
    return lines


def call_edges(spans):
    """Service-to-service calls seen in the traces: every parent and child span that sit in two different services.

    Not every caller opens a client span (Social's nginx, or home-timeline reading posts), so the link is what counts.
    A client span with no child at all is a call nobody answered, shown as callee '? <operation>': databases and caches
    are not traced, so their calls always land there; a service that should answer and does not is a fault.
    """
    child = spans[["trace", "parent", "service", "error"]].rename(columns={"parent": "span", "service": "callee", "error": "failed"})
    pairs = spans[["trace", "span", "service"]].merge(child, on=["trace", "span"])
    answered = (pairs[pairs.service != pairs.callee].groupby(["service", "callee"])
                .agg(calls=("span", "size"), errors=("failed", "sum")).reset_index().assign(unanswered=0))
    lonely = unanswered_calls(spans)
    unanswered = (lonely.assign(callee="? " + lonely.operation).groupby(["service", "callee"])
                  .agg(calls=("span", "size"), errors=("error", "sum")).reset_index().assign(unanswered=lambda d: d.calls))
    return (pd.concat([answered, unanswered]).rename(columns={"service": "caller"})[["caller", "callee", "calls", "unanswered", "errors"]]
            .sort_values("calls", ascending=False).reset_index(drop=True))


def unanswered_calls(spans):
    """Client spans with no child at all: calls to untraced databases and caches, and calls to a service that never answered."""
    has_child = set(zip(spans.trace, spans.parent))
    lonely = np.array([(t, s) not in has_child for t, s in zip(spans.trace, spans.span)], dtype=bool)
    return spans[(spans.kind == "client").to_numpy() & lonely]


def error_origins(spans):
    """For each trace with errors, the deepest failing span (no failing child): where the error starts, not where it surfaces."""
    failing = spans[spans.error == 1]
    has_failing_child = set(zip(failing.trace, failing.parent))
    deepest = np.array([(t, s) not in has_failing_child for t, s in zip(failing.trace, failing.span)], dtype=bool)
    return failing[deepest][["trace", "start_us", "service", "operation", "kind", "status", "error_msg"]]
