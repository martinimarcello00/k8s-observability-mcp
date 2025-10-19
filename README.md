# ☸️ K8s Observability MCP

Small MCP server that lets you explore Kubernetes metrics, logs, traces, and service graph data via simple tools.

- 🐍 Python 3.13
- 📈 Prometheus
- 🔎 Jaeger
- 🕸️ Neo4j
- ☸️ Kubernetes API

## Features

- 📊 Get pod/service metrics (instant and range)
- 📜 Read pod/service logs with important-line filtering
- 🔗 Service map from Neo4j (uses/depends)
- 🧭 Cluster overview (pods and services)
- 🧵 Trace summaries and details from Jaeger

## Requirements

- 🐍 Python 3.13+
- 📦 Poetry
- ☸️ Access to your cluster (kubeconfig on this machine)
- 📈 Prometheus URL
- 🔎 Jaeger URL
- 🕸️ Neo4j URI, user, password

## Setup

- Install (Poetry)

```bash
poetry install
```

- Configure env

```bash
cp .env.example .env
# edit .env with your values
```

## Run

```bash
poetry run python mcp_server.py
```
Then connect with your MCP client to use the tools.


## Tools

- get_metrics / get_metrics_range
- get_pods_from_service
- get_cluster_overview
- get_services_used_by
- get_dependencies
- get_logs
- get_traces / get_trace

## Notes

- Uses your default kubeconfig. Set TARGET_NAMESPACE in .env to scope queries.

- 🕸️ Service graph docs: see `service-graph/README.md` for how the Neo4j service graph is built (Jaeger CALLS + static USES), how to load it, and the result image.
