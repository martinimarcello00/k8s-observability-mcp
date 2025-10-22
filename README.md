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

### 🔍 **Kubernetes Resource Inspection**

- **`get_pods_from_service(service)`**
  - Returns all pods belonging to a specific service
  - Shows pod names and current status (Running, Pending, etc.)

- **`get_cluster_pods_and_services()`**
  - Comprehensive cluster overview
  - Lists all pods and services with counts

### 📊 **Metrics & Observability**

- **`get_metrics(resource_name, resource_type)`**
  - Retrieves instant Prometheus metrics for a pod or service
  - Parameters:
    - `resource_name`: The exact name of the Kubernetes resource
    - `resource_type`: Either "pod" or "service"
  - Returns CPU, memory, network, thread, and container specifications

- **`get_metrics_range(resource_name, resource_type, time_range_minutes)`**
  - Historical metrics over a specified time range from Prometheus
  - Parameters:
    - `resource_name`: The exact name of the Kubernetes resource
    - `resource_type`: Either "pod" or "service"
    - `time_range_minutes`: Historical lookback in minutes (minimum 1)

- **`get_logs(resource_name, resource_type, tail=100, important=True)`**
  - Retrieve pod/service logs with optional keyword filtering
  - Parameters:
    - `resource_name`: The exact name of the Kubernetes resource
    - `resource_type`: Either "pod" or "service"
    - `tail`: Number of recent log lines to retrieve (default: 100)
    - `important`: If true, filter for ERROR, WARN, CRITICAL keywords (default: true)

### 🔗 **Service Dependencies & Graph**

- **`get_services_used_by(service)`**
  - Returns downstream services called by the given service
  - Shows service dependency chain (who calls whom)

- **`get_dependencies(service)`**
  - Retrieves infrastructure dependencies for a service
  - Includes databases, caches, message queues, etc.

### 🧵 **Distributed Tracing**

- **`get_traces(service_name, only_errors=False)`**
  - Retrieves traces for a specific service from Jaeger
  - Parameters:
    - `service_name`: The name of the service to retrieve traces for
    - `only_errors`: If true, return only traces containing errors (default: false)
  - Returns: traceID, latency_ms, has_error, service sequence

- **`get_trace(trace_id)`**
  - Retrieves detailed information for a specific trace by ID
  - Parameters:
    - `trace_id`: The unique trace ID to retrieve
  - Includes all spans with timestamps, durations, tags, and errors

## Notes

- Uses your default kubeconfig. Set TARGET_NAMESPACE in .env to scope queries.

- 🕸️ Service graph docs: see `service-graph/README.md` for how the Neo4j service graph is built (Jaeger CALLS + static USES), how to load it, and the result image.
