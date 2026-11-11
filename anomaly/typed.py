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
    """client or server. Social's C++ services set no span.kind tag, but name every operation ..._client or ..._server
    (user-service once writes user_mongo_insert_cilent)."""
    if tags.get("span.kind"):
        return tags["span.kind"]
    return "client" if operation.endswith(("_client", "_cilent")) else "server" if operation.endswith("_server") else ""


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


DATASTORE_OP = re.compile(r"mongo|redis|memcached|mmc", re.I)  # Social names its memcached calls mmc_...
STORES = ("mongodb", "redis", "memcached")  # the kind of datastore, as it appears in datagraph node names


def add_callees(spans, graph):
    """The spans with two columns more: callee, the datagraph node each client span calls ('' when unknown), and datastore.

    The callee is, in order: the service of a child span in another service; for a call nobody answered, the service that
    answered the same caller and operation elsewhere in the run; for a database or cache call (never traced), the only node of
    that kind the caller USES in the datagraph. datastore also reads the operation name, so a USES edge missing from the
    datagraph (Social's user-mention-service and user-memcached) leaves the callee '' but still marks the call.
    """
    from anomaly import build
    stores = {name for name, kind in build.datagraph_nodes(graph) if kind in ("Database", "Cache")}
    uses = {}
    for caller, relation, node in build.datagraph_edges(graph):
        if relation == "USES":
            uses.setdefault((caller, next((s for s in STORES if s in node), "")), []).append(node)
    client = spans.kind == "client"
    children = spans.loc[spans.parent != "", ["trace", "parent", "service"]].rename(columns={"parent": "span", "service": "child"})
    answered = (spans.loc[client, ["trace", "span", "service", "operation"]].reset_index().merge(children, on=["trace", "span"])
                .query("child != service").drop_duplicates("index").set_index("index"))
    callee = pd.Series("", index=spans.index, dtype=object)
    callee[answered.index] = answered.child
    # ponytail: learned within the run, so an operation never answered in the whole run stays ''; a mapping across runs if that bites
    learned = (answered.value_counts(["service", "operation", "child"]).reset_index()
               .drop_duplicates(["service", "operation"]).set_index(["service", "operation"]).child.to_dict())
    store_ops = {op for op in spans.operation.unique() if DATASTORE_OP.search(op)}
    is_store_op = client & spans.operation.isin(store_ops)
    for caller, operation in spans.loc[is_store_op, ["service", "operation"]].drop_duplicates().itertuples(index=False):
        kind = "mongodb" if "mongo" in operation else "redis" if "redis" in operation else "memcached"
        nodes = uses.get((caller, kind), [])
        if len(nodes) == 1:
            learned.setdefault((caller, operation), nodes[0])
    todo = client & (callee == "")
    if todo.any() and learned:
        pairs = pd.MultiIndex.from_frame(spans.loc[todo, ["service", "operation"]])
        callee[todo] = pd.Series(learned).reindex(pairs).fillna("").to_numpy()
    return spans.assign(callee=callee, datastore=is_store_op | callee.isin(stores))


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
    """Every container log line of the run (wrk2's excluded: it is the load, parsed by the builder), with kubectl's timestamp.

    A container waiting in CrashLoopBackOff has no current instance: kubectl gives the dead one's log both with and without
    --previous, so the current file is skipped when it equals the previous one and the lines are kept once, as previous.
    """
    rows = []
    for path in sorted((Path(run_dir) / "logs").glob("*.log.gz")):
        if path.name == "wrk2.log.gz":
            continue
        name = path.name[:-len(".log.gz")]
        previous = name.endswith(".previous")
        pod, container = name.removesuffix(".previous").split("__", 1)
        dead = path.with_name(f"{name}.previous.log.gz")
        if not previous and dead.exists() and gzip.open(dead).read() == gzip.open(path).read():
            continue
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

MIN_SPANS = 20  # below it p50 and p95 are masked: a p95 of 4 spans is their max. A calibration knob, not a measured constant


def span_features(spans, grid, step, sampling=None, min_spans=MIN_SPANS):
    """Per grid step (t - step, t] and node, from the spans through add_callees.

    Served side, for the traced services: requests (spans in the sample), rps (requests per second of real traffic:
    requests / step / sampling, NaN when the sampling is unknown), p50_ms and p95_ms (NaN under min_spans requests),
    error_share. A request served = a server span, or a root span (Social's nginx tags no span kind).
    Called side, for every node client spans call, databases and caches included: calls_in, failed_share (the client
    span or its child failed), unanswered_share (no child at all; NaN for databases and caches, which never answer
    with a span). It puts a fault on the broken node, which often has no span of its own (a crashed service).
    Every trace is exported (Hotel samples 100%, Social 60%), so the shares and percentiles need no rescaling.
    Inside the steps the export covers, a node that served or received nothing has requests or calls_in 0, not a gap.
    """
    def on_grid(frame):
        cell = np.searchsorted(grid, frame.start_us / 1e6, side="left")
        ok = (cell < len(grid)) & (grid[np.minimum(cell, len(grid) - 1)] - frame.start_us / 1e6 < step)
        return frame[ok].assign(t=grid[cell[ok]])

    def filled(frame, count):  # 0 for every node of the frame in every step between the first and the last span
        full = pd.MultiIndex.from_product([covered, frame.index.get_level_values("service").unique()], names=["t", "service"])
        frame = frame.reindex(frame.index.union(full))
        frame[count] = frame[count].fillna(0)
        return frame

    covered = grid[(grid - step >= spans.start_us.min() / 1e6) & (grid <= spans.start_us.max() / 1e6)]
    server = on_grid(spans.loc[(spans.kind == "server") | (spans.parent == ""), ["start_us", "duration_us", "service", "error"]])
    ms = (server.duration_us / 1000).groupby([server.t, server.service])
    served = pd.DataFrame({"requests": ms.size(), "p50_ms": ms.median(), "p95_ms": ms.quantile(0.95),
                           "error_share": server.error.groupby([server.t, server.service]).mean()})
    kids = (spans.loc[spans.parent != "", ["trace", "parent", "error"]].groupby(["trace", "parent"]).error.max()
            .rename("child_error").rename_axis(["trace", "span"]).reset_index())
    calls = on_grid(spans.loc[(spans.kind == "client") & (spans.callee != ""), ["trace", "span", "start_us", "callee", "error", "datastore"]]
                    .merge(kids, on=["trace", "span"], how="left"))
    calls = calls.assign(unanswered=calls.child_error.isna(), failed=(calls.error == 1) | (calls.child_error == 1)).rename(columns={"callee": "service"})
    by = calls.groupby(["t", "service"])
    called = pd.DataFrame({"calls_in": by.size(), "unanswered_share": by.unanswered.mean(), "failed_share": by.failed.mean()})
    stores = set(calls.service[calls.datastore])
    called.loc[called.index.get_level_values("service").isin(stores), "unanswered_share"] = np.nan
    out = filled(served, "requests").join(filled(called, "calls_in"), how="outer")
    out.loc[out.requests < min_spans, ["p50_ms", "p95_ms"]] = np.nan
    out.insert(1, "rps", out.requests / step / sampling if sampling else np.nan)
    return out.rename_axis(["t", "service"]).reset_index()


def trace_check(spans, attempt, labels, min_share=0.9, block_s=10):
    """V5: are the run's traces whole? (passed, detail), like V1-V4, but it gates the trace features only, never the metrics.

    - Jaeger kept its memory store: no restart and no new pod since before the grid starts (both would wipe every trace);
    - in each observed load segment of the window of V1-V3 (30 s or longer, wrk2 reported), the root spans are at least
      min_share of the requests wrk2 delivered times the sampling;
    - no export block reached its limit inside a load segment (a full block lost its oldest traces).
    A run without trace_sampling comes from the export before pilot2 (at most 5 traces/s kept): it does not pass.
    """
    attempt = Path(attempt)
    meta = json.loads((attempt / "meta.json").read_text())
    sampling = meta.get("trace_sampling")
    if not sampling:
        return False, {"reason": "no trace_sampling: export before pilot2, at most 5 traces/s kept"}
    with gzip.open(attempt / "pods.json.gz", "rt") as f:
        jaeger = [c for p in json.load(f)["items"] if "jaeger" in p["metadata"]["name"] for c in p["status"].get("containerStatuses", [])]
    start = labels["t_workload"] or labels["t_inject"]
    started = [pd.Timestamp(c["state"]["running"]["startedAt"]).timestamp() for c in jaeger if "running" in c.get("state", {})]
    kept = bool(jaeger) and len(started) == len(jaeger) and all(c["restartCount"] == 0 for c in jaeger) and max(started) <= start
    t0, t1 = labels["checks"]["window"]
    segments = [s for s in labels["segments"] if t0 < s["start"] + s["seconds"] <= t1 and s["seconds"] >= 30 and s["delivered"]]
    roots = np.sort(spans.start_us[spans.parent == ""].to_numpy() / 1e6)
    shares = [(s["start"], s["rate"], round(float(np.searchsorted(roots, s["start"] + s["seconds"]) - np.searchsorted(roots, s["start"]))
                                            / (s["delivered"] * s["seconds"] * sampling), 3)) for s in segments]
    full = [b for b in meta.get("jaeger_full_blocks") or [] if any(s["start"] < b + block_s and b < s["start"] + s["seconds"] for s in labels["segments"])]
    passed = kept and all(share >= min_share for *_, share in shares) and not full
    # completeness_judged False: no long segment ended in the window (a 300 s settle often holds none), so only Jaeger and the
    # full blocks were checked
    return passed, {"jaeger_kept": kept, "jaeger_restarts": sum(c["restartCount"] for c in jaeger), "completeness_judged": bool(shares),
                    "worst_segment": min(shares, key=lambda s: s[2], default=None), "segments": len(shares), "full_blocks_under_load": full}


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
    A client span with no child at all is a call nobody answered, shown with its callee from add_callees, or as
    '? <operation>' when unknown: databases and caches are not traced, so their calls always land there; a service that
    should answer and does not is a fault. Needs the spans through add_callees.
    """
    child = spans[["trace", "parent", "service", "error"]].rename(columns={"parent": "span", "service": "callee", "error": "failed"})
    pairs = spans[["trace", "span", "service"]].merge(child, on=["trace", "span"])
    answered = (pairs[pairs.service != pairs.callee].groupby(["service", "callee"])
                .agg(calls=("span", "size"), errors=("failed", "sum")).reset_index().assign(unanswered=0))
    lonely = unanswered_calls(spans)
    unanswered = (lonely.assign(callee=lonely.callee.where(lonely.callee != "", "? " + lonely.operation)).groupby(["service", "callee"])
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
