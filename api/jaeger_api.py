import requests
import logging
from typing import Optional, Dict, List, Any
from .base_k8s_client import BaseK8sClient
from .config_manager import ConfigManager

class JaegerAPI(BaseK8sClient):
    def __init__(self, jaeger_url: Optional[str] = None):
        config_manager = ConfigManager()
        self.jaeger_url = jaeger_url or config_manager.config.jaeger_url
        
        # Initialize with namespace=None to get all services across namespaces
        super().__init__(namespace=None)
        self.services = self.get_services_list()
    
    def get_jaeger_traces(self, service: str, limit: int = 20, lookback: str = "15m", min_latency_ms: Optional[float] = None, only_errors: bool = False):
        """Fetches traces from the Jaeger Query API, optionally filtering by minimum latency (ms) and error traces using Jaeger API parameters."""
        logging.info(f"Querying Jaeger for '{service}' traces...")
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

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            traces = response.json().get("data", [])
            return traces
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to Jaeger: {e}")
            return None
        except KeyError:
            logging.error("Unexpected response format from Jaeger. 'data' key not found.")
            return None

    def process_trace(self, trace: Dict[str, Any]):
        """Extracts latency, service sequence, and error details from a single trace."""
        
        # Find the Root Span and Total Latency
        root_span = None
        for span in trace["spans"]:
            if not span.get("references"):
                root_span = span
                break
                
        if not root_span:
            return None

        latency_ms = root_span["duration"] / 1000.0

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
                for log in span.get("logs", []):
                    # Find fields like 'event: error', 'message', or 'stack'
                    log_fields = {field['key']: field['value'] for field in log.get("fields", [])}
                    if log_fields.get("event") == "error":
                        if "message" in log_fields:
                            error_details.append(log_fields["message"])
                        if "stack" in log_fields: # Stack traces can be verbose but useful
                            error_details.append(log_fields["stack"].split('\n')[0]) # Get first line of stack
        
        if error_details:
            error_message = "; ".join(error_details) # Join multiple messages

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
            logging.error(f"Failed to retrieve traces for service '{service}'. Check Jaeger connectivity and service name.")
            results["error"] = "Failed to fetch traces from Jaeger"
            return results

        if not traces:
            logging.warning(f"No traces found for service '{service}' with lookback '{lookback}'.")
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
        logging.info(f"Querying Jaeger for trace ID: {trace_id}")
        api_url = f"{self.jaeger_url}/api/traces/{trace_id}"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            trace_data = response.json()
            
            if "data" in trace_data and len(trace_data["data"]) > 0:
                return trace_data["data"][0]
            else:
                logging.warning(f"No trace found with ID: {trace_id}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to Jaeger: {e}")
            return None
        except (KeyError, IndexError) as e:
            logging.error(f"Unexpected response format from Jaeger: {e}")
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
            logging.warning(f"No slow traces found for service '{service}' with min duration {min_duration_ms}ms")
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