import json
import os
import time
import requests
import logging
from typing import Optional, Dict, List, Any
from .base_k8s_client import BaseK8sClient, MECHANISM, hide_flag_announcement
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

def strip_flagd(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Hide the fault-injection mechanism (see MECHANISM) from a trace, None when nothing is left.
    Flag evaluation spans (flagd's own, the callers' Resolve*, OFREP and Envoy /flagservice/ calls) are leaves and go;
    flag attributes and evaluation events go; error messages announcing a flag keep only their symptom.
    Astronomy Shop services and browsers also keep a stream open to flagd (EventStream) in a trace of its own: it lasts
    as long as the connection and fails at every deadline or flagd restart, noise for slow/error trace triage."""
    if not MECHANISM.search(json.dumps(trace)):
        return trace

    def clean(value):
        return hide_flag_announcement(value) if isinstance(value, str) else value

    hidden = {pid for pid, process in trace["processes"].items() if MECHANISM.search(process["serviceName"])}
    spans = []
    for span in trace["spans"]:
        tags = [{**tag, "value": clean(tag.get("value"))} for tag in span.get("tags", [])]
        if (span["processID"] in hidden or MECHANISM.search(span["operationName"])
                or any(MECHANISM.search(str(tag["value"])) for tag in tags)):
            continue
        logs = [{**log, "fields": [{**field, "value": clean(field.get("value"))} for field in log.get("fields", [])]}
                for log in span.get("logs", [])]
        spans.append({**span,
                      "tags": [tag for tag in tags if not MECHANISM.search(tag["key"])],
                      "logs": [log for log in logs
                               if not any(MECHANISM.search(f"{field['key']}={field['value']}") for field in log["fields"])]})
    if not spans:
        return None
    return {**trace, "spans": spans, "processes": {pid: p for pid, p in trace["processes"].items() if pid not in hidden}}

class JaegerAPI(BaseK8sClient):
    def __init__(self, jaeger_url: Optional[str] = None):
        config_manager = ConfigManager()
        self.jaeger_url = jaeger_url or config_manager.config.jaeger_url
        
        # Initialize with namespace=None to get all services across namespaces
        super().__init__(namespace=None)
        self.services = self.get_services_list()
    
    def get_jaeger_traces(self, service: str, limit: int = 20, lookback: str = "15m", min_latency_ms: Optional[float] = None, only_errors: bool = False):
        """Fetches traces from the Jaeger Query API, optionally filtering by minimum latency (ms) and error traces using Jaeger API parameters."""
        logger.info(f"Querying Jaeger for '{service}' traces...")
        api_url = f"{self.jaeger_url}/api/traces"

        params = {
            "service": service,
            "limit": limit,
            "lookback": lookback,
        }

        if min_latency_ms is not None:
            params["minDuration"] = f"{int(min_latency_ms)}ms"
        if only_errors:
            params["tags"] = '{"error":"true"}'
        if os.environ.get("JAEGER_START_US"):
            # Jaeger ignores lookback (it searches the whole store): the harness gives the instant the environment was ready
            params["start"], params["end"] = os.environ["JAEGER_START_US"], int(time.time() * 1e6)

        def search(extra: Dict[str, str] = {}) -> List[Dict[str, Any]]:
            response = requests.get(api_url, params={**params, **extra})
            response.raise_for_status()
            return response.json().get("data") or []

        try:
            traces = search()
            kept = [trace for trace in map(strip_flagd, traces) if trace]
            if len(traces) == limit and len(kept) < limit:
                # EventStream traces can fill the whole page (frontend-proxy: ~15/min from browser sessions) and push
                # real ones out: search again per operation, as real requests have their own
                # (e.g. Envoy's "router frontend egress"), then keep the most recent ones
                response = requests.get(f"{self.jaeger_url}/api/services/{service}/operations")
                response.raise_for_status()
                for operation in response.json().get("data") or []:
                    kept += [trace for trace in map(strip_flagd, search({"operation": operation})) if trace]
                kept = sorted({trace["traceID"]: trace for trace in kept}.values(),
                              key=lambda trace: min(span["startTime"] for span in trace["spans"]), reverse=True)
            return kept[:limit]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Jaeger: {e}")
            return None
        except KeyError:
            logger.error("Unexpected response format from Jaeger. 'data' key not found.")
            return None

    def process_trace(self, trace: Dict[str, Any]):
        """Extracts latency, service sequence, and error details from a single trace."""
        
        # Find the Root Span and Total Latency
        root_span = None
        for span in trace["spans"]:
            if not span.get("references"):
                root_span = span
                break
                
        if root_span:
            latency_ms = root_span["duration"] / 1000.0
        else:
            # OTel (e.g. Astronomy Shop): the root span lives in a client that doesn't export to Jaeger
            start = min(s["startTime"] for s in trace["spans"])
            end = max(s["startTime"] + s["duration"] for s in trace["spans"])
            latency_ms = (end - start) / 1000.0

        # Check for Errors and Extract Messages
        has_error = False
        error_message = "N/A"
        error_details = [] # Store multiple error messages if they exist

        for span in trace["spans"]:
            is_error_span = False
            for tag in span.get("tags", []):
                if tag.get("key") == "error" and tag.get("value") is True:
                    has_error = True
                    is_error_span = True
                    break
            
            # If this span has the error, search its logs for the reason
            if is_error_span:
                # OTel puts the error reason in the span status
                for tag in span.get("tags", []):
                    if tag.get("key") == "otel.status_description":
                        error_details.append(tag["value"])
                for log in span.get("logs", []):
                    # OpenTracing: event=error with message/stack; OTel: event=exception with exception.*
                    log_fields = {field['key']: field['value'] for field in log.get("fields", [])}
                    if log_fields.get("event") in ("error", "exception"):
                        message = log_fields.get("message") or log_fields.get("exception.message")
                        if message:
                            error_details.append(message)
                        # OTel stacktrace's first line just repeats exception.message, so only use it as a fallback
                        stack = log_fields.get("stack") or (None if message else log_fields.get("exception.stacktrace"))
                        if stack: # Stack traces can be verbose but useful
                            error_details.append(stack.split('\n')[0]) # Get first line of stack
        
        if error_details:
            # OTel repeats (and wraps) the same message on every span up the call chain:
            # drop duplicates and messages already contained in a longer one
            unique = list(dict.fromkeys(error_details))
            error_message = "; ".join(d for d in unique if not any(d != o and d in o for o in unique))

        # Determine the Sequence of Services
        service_map = {p_id: p_info["serviceName"] for p_id, p_info in trace["processes"].items()}
        sorted_spans = sorted(trace["spans"], key=lambda s: s["startTime"])
        
        service_sequence = []
        last_service = None
        for span in sorted_spans:
            service_name = service_map.get(span["processID"])
            if service_name and service_name != last_service:
                service_sequence.append(service_name)
                last_service = service_name
                
        result = {
            "traceID": trace["traceID"],
            "latency_ms": latency_ms,
            "has_error": has_error,
            "sequence": " -> ".join(service_sequence)
        }
        
        if has_error:
            result["error_message"] = error_message
        
        return result
    
    def get_processed_traces(self, service: str, limit: int = 20, lookback: str = "15m", only_errors: bool = False) -> Dict[str, Any]:
        results = {}

        if service not in self.services:
            results["error"] = f"The service {service} does not exist"
            return results

        results["service"] = service
        results["traces"] = []

        traces = self.get_jaeger_traces(service, limit, lookback, only_errors=only_errors)

        if traces is None:
            logger.error(f"Failed to retrieve traces for service '{service}'. Check Jaeger connectivity and service name.")
            results["error"] = "Failed to fetch traces from Jaeger"
            return results

        if not traces:
            logger.warning(f"No traces found for service '{service}' with lookback '{lookback}'.")
            results["info"] = f"No traces found for service '{service}' with lookback '{lookback}'."
            return results

        for trace in traces:
            trace_data = self.process_trace(trace)
            if trace_data:
                results["traces"].append(trace_data)

        results["traces_count"] = len(results["traces"])

        return results
        
    def get_trace(self, trace_id: str):
        """Fetches a single trace by trace ID from Jaeger."""
        logger.info(f"Querying Jaeger for trace ID: {trace_id}")
        api_url = f"{self.jaeger_url}/api/traces/{trace_id}"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            trace_data = response.json()
            
            if "data" in trace_data and len(trace_data["data"]) > 0:
                return strip_flagd(trace_data["data"][0])
            else:
                logger.warning(f"No trace found with ID: {trace_id}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Jaeger: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Jaeger: {e}")
            return None
    
    def get_slow_traces(
        self,
        service: str,
        min_duration_ms: float,
        limit: int = 30,
        lookback: str = "15m",
        only_errors: bool = False
    ) -> Dict[str, Any]:
        """
        Args:
            service: Name of the service to query
            min_duration_ms: Minimum latency threshold in milliseconds
            limit: Maximum number of traces to return
            lookback: Time duration to look back (e.g., "1h", "30m", "5m")
            only_errors: If True, only return traces with errors
        Returns:
            List of processed trace dictionaries with high latency
        """

        results = {}

        if service not in self.services:
            results["error"] = f"The service {service} does not exist"
            return results

        results["service"] = service
        results["traces"] = []

        # Fetch traces using Jaeger's native duration and error filter
        traces = self.get_jaeger_traces(
            service=service,
            limit=limit,
            lookback=lookback,
            min_latency_ms=min_duration_ms,
            only_errors=only_errors
        )

        if not traces:
            logger.warning(f"No slow traces found for service '{service}' with min duration {min_duration_ms}ms")
            results["info"] = f"No traces found for service '{service}' with a minimum duration of {min_duration_ms}ms in the last {lookback}."
            return results

        # Process and return only the slow traces
        for trace in traces:
            trace_data = self.process_trace(trace)
            if trace_data:
                results["traces"].append(trace_data)

        # Sort by latency (slowest first)
        results["traces"].sort(key=lambda x: x["latency_ms"], reverse=True)

        results["traces_count"] = len(results["traces"])

        return results