from mcp.server.fastmcp import FastMCP
import logging
from pydantic import Field
from typing import Literal
from typing_extensions import Annotated
from config_manager import ConfigManager
from prometheus_api import PrometheusAPI
from log_api import LogAPI
from jaeger_api import JaegerAPI
from datagraph import DataGraph

# Global instances - initialized lazily
prometheus_api = None
datagraph = None
log_api = None
jaeger_api = None

def get_apis():
    """Get or create all API instances using singleton pattern"""
    config_manager = ConfigManager()
    config = config_manager.config
    
    global prometheus_api, datagraph, log_api, jaeger_api
    
    if prometheus_api is None:
        prometheus_api = PrometheusAPI(config.prometheus_url, config.target_namespace)
    if datagraph is None:
        datagraph = DataGraph(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
    if log_api is None:
        log_api = LogAPI(config.target_namespace)
    if jaeger_api is None:
        jaeger_api = JaegerAPI(config.jaeger_url)
    
    return prometheus_api, datagraph, log_api, jaeger_api

def get_prometheus_api():
    """Get or create the PrometheusAPI instance"""
    prometheus_api, _, _, _ = get_apis()
    return prometheus_api

def get_datagraph():
    """Get or create the DataGraph instance"""
    _, datagraph, _, _ = get_apis()
    return datagraph

def get_log_api():
    """Get or create the LogAPI instance"""
    _, _, log_api, _ = get_apis()
    return log_api

def get_jaeger_api():
    """Get or create the JaegerAPI instance"""
    _, _, _, jaeger_api = get_apis()
    return jaeger_api

mcp = FastMCP("Cluster API MCP")

@mcp.tool(
    title="get_metrics",
    description="Retrieve all instant Prometheus metrics for a specific Kubernetes pod or service. Returns comprehensive metrics including CPU, memory, network, and container specifications."
)
def get_metrics(
    resource_name: Annotated[str, Field(description="The exact name of the Kubernetes resource to retrieve metrics for.")],
    resource_type: Annotated[Literal["pod","service"], Field(description="Type of Kubernetes resource. 'pod' returns metrics for a single pod. 'service' returns aggregated metrics for all pods behind the service.")]
        
) -> dict:
    """Get all the Prometheus metrics associated with a specific cluster resource"""
    api = get_prometheus_api()
    if resource_type == "pod":
        return api.get_pod_metrics(resource_name)
    else:
        pods = api.get_pods_from_service(resource_name)
        
        if "error" in pods.keys():
            return pods
        
        # Aggregate metrics from all pods in the service
        service_metrics = {
            "service_name": resource_name,
            "pods": []
        }
        
        for pod in pods["pods"]:
            pod_metrics = api.get_pod_metrics(pod["pod_name"])
            service_metrics["pods"].append(pod_metrics)
        
        return service_metrics

@mcp.tool(
    title="get_metrics_range",
    description="Retrieve historical Prometheus metrics for a specific Kubernetes pod or service over a time range."
)  
def get_metrics_range(
    resource_name: Annotated[str, Field(description="The exact name of the Kubernetes resource to retrieve metrics for.")],
    resource_type: Annotated[Literal["pod","service"], Field(description="Type of Kubernetes resource. 'pod' returns metrics for a single pod. 'service' returns aggregated metrics for all pods behind the service.")],
    time_range_minutes: Annotated[int, Field(description="The time range in minutes to retrieve historical metrics from now. Must be at least 1 minute.", ge=1)]
) -> dict:
    """Get historical Prometheus metrics for a resource over a specified time range from now"""
    api = get_prometheus_api()
    if resource_type == "pod":
        return api.get_pod_metrics_range(resource_name, time_range_minutes)
    else:
        pods = api.get_pods_from_service(resource_name)
        
        if "error" in pods.keys():
            return pods
        
        # Aggregate metrics from all pods in the service
        service_metrics = {
            "service_name": resource_name,
            "time_range_minutes": time_range_minutes,
            "pods": []
        }
        
        for pod in pods["pods"]:
            pod_metrics = api.get_pod_metrics_range(pod["pod_name"], time_range_minutes)
            service_metrics["pods"].append(pod_metrics)
        
        return service_metrics

@mcp.tool(
    title="get_pods_from_service",
    description="Retrieve all Kubernetes pods that belong to a specific service. Returns pod names and their current status (Running, Pending, etc.)."
)
def get_pods_from_service(
    service_name: Annotated[str, Field(description="The exact name of the Kubernetes service to find associated pods for.")]
) -> dict:
    """Get all the pods associated with a specific service"""
    api = get_prometheus_api()
    return api.get_pods_from_service(service_name)

@mcp.tool(
    title="get_cluster_overview",
    description="Get a comprehensive overview of the Kubernetes cluster including all pods and services. Returns counts and complete lists for cluster analysis."
)
def get_cluster_pods_and_services() -> dict:
    """Get the complete list of all pods and services in the target Kubernetes namespace"""
    api = get_prometheus_api()
    pods = api.get_pods_list()
    services = api.get_services_list()
    return {
        "namespace": api.namespace,
        "pods": pods,
        "services": services,
        "summary": f"Found {len(pods)} pods and {len(services)} services in namespace '{api.namespace}'"
    }

@mcp.tool(
    title="get_services_used_by",
    description="Return all the services that are used by the given service to complete its tasks. This shows the service dependency chain - which services the target service calls to fulfill requests."
)
def get_services_used_by(
    service: Annotated[str, Field(description="The name of the service to analyze for its service dependencies.")]
) -> dict:
    """Return all the services that are used by the given service to complete its tasks"""
    graph = get_datagraph()
    services_used = graph.get_services_used_by(service)
    
    if isinstance(services_used, str):
        # Error case
        return {"error": services_used, "service": service}
    
    return {
        "service": service,
        "services_used": services_used,
        "count": len(services_used),
        "summary": f"Service '{service}' uses {len(services_used)} other services to complete its tasks"
    }

@mcp.tool(
    title="get_dependencies",
    description="Retrieves all dependencies for a specified service from kubernetes cluster. Dependencies include databases and other infrastructure components."
)
def get_dependencies(
    service: Annotated[str, Field(description="The name of the service to analyze for its infrastructure dependencies.")]
) -> dict:
    """Retrieves all dependencies (databases, external services, etc.) for a specified service"""
    graph = get_datagraph()
    dependencies = graph.get_dependencies(service)
    
    if isinstance(dependencies, str):
        # Error case
        return {"error": dependencies, "service": service}
    
    return {
        "service": service,
        "dependencies": dependencies,
        "summary": f"Service '{service}' has {len(dependencies)} infrastructure dependencies"
    }

@mcp.tool(
    title="get_logs",
    description="Retrieve logs from a Kubernetes pod or service with optional filtering for important messages."
)
def get_logs(
    resource_name: Annotated[str, Field(description="The exact name of the Kubernetes resource to retrieve logs from.")],
    resource_type: Annotated[Literal["pod","service"], Field(description="Type of Kubernetes resource. 'pod' returns logs for a single pod. 'service' returns logs for all pods behind the service.")],
    tail: Annotated[int, Field(description="Number of recent log lines to retrieve.", ge=1)] = 100,
    important: Annotated[bool, Field(description="If True, filter logs to only include lines with ERROR, WARN, or CRITICAL keywords.")] = True,
) -> str:
    """Retrieves the last log entries of a pod or service with optional filtering for important messages"""
    log_api_instance = get_log_api()
    
    if resource_type == "pod":
        return log_api_instance.get_pod_logs(resource_name, tail, important)
    else:
        # Get pods from service first
        api = get_prometheus_api()
        pods = api.get_pods_from_service(resource_name)
        
        if "error" in pods.keys():
            return f"Error getting pods for service '{resource_name}': {pods['error']}"
        
        # Collect logs from all pods in the service
        service_logs = f"=== Logs for service '{resource_name}' ===\n\n"
        
        for pod in pods["pods"]:
            pod_logs = log_api_instance.get_pod_logs(pod["pod_name"], tail, important)
            service_logs += f"--- Pod: {pod['pod_name']} ---\n"
            service_logs += pod_logs
            service_logs += "\n\n"
        
        return service_logs

@mcp.tool(
    title="get_traces",
    description="Retrieve traces for a specific service, with an option to filter for traces that contain errors. Returns a list of traces, each containing: traceID (unique trace identifier), latency_ms (total trace duration in milliseconds), has_error (boolean indicating if the trace contains errors), and sequence (string showing the service call chain, e.g., 'serviceA -> serviceB -> serviceC')."
)
def get_traces(
        service_name: Annotated[str, Field(description="The name of the service for which to retrieve traces.")],
        only_errors: Annotated[bool, Field(description="If True, return only traces that contain errors.")] = False
) -> dict:
    
    jaeger = get_jaeger_api()
    traces = jaeger.get_processed_traces(service_name, only_errors=only_errors)

    return traces

@mcp.tool(
    title="get_trace",
    description="Retrieve detailed information for a specific trace by its trace ID. Returns the complete trace with all spans, including service names, operation names, timestamps, durations, tags, and any errors."
)
def get_trace(
    trace_id: Annotated[str, Field(description="The unique trace ID to retrieve detailed information for.")]
) -> dict:
    """Get detailed information for a specific trace by trace ID"""
    jaeger = get_jaeger_api()
    result = jaeger.get_trace(trace_id)
    if result is None:
        return {"error": f"Trace with ID '{trace_id}' not found"}
    return result

if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.config
    logging.info(f"Target namespace: {config.target_namespace}")
    mcp.run(transport="streamable-http")