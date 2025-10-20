from prometheus_api_client import PrometheusConnect
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from .base_k8s_client import BaseK8sClient
from .config_manager import ConfigManager

class PrometheusAPI(BaseK8sClient):

    normal_metrics = [
        # cpu
        "container_cpu_usage_seconds_total",
        "container_cpu_user_seconds_total",
        "container_cpu_system_seconds_total",
        "container_cpu_cfs_throttled_seconds_total",
        "container_cpu_cfs_throttled_periods_total",
        "container_cpu_cfs_periods_total",
        "container_cpu_load_average_10s",
        # memory
        "container_memory_cache",
        "container_memory_usage_bytes",
        "container_memory_working_set_bytes",
        "container_memory_rss",
        "container_memory_mapped_file",
        # spec
        "container_spec_cpu_period",
        "container_spec_cpu_quota",
        "container_spec_memory_limit_bytes",
        "container_spec_cpu_shares",
        # threads
        "container_threads",
        "container_threads_max",
        # network
        "container_network_receive_errors_total",
        "container_network_receive_packets_dropped_total",
        "container_network_receive_packets_total",
        "container_network_receive_bytes_total",
        "container_network_transmit_bytes_total",
        "container_network_transmit_errors_total",
        "container_network_transmit_packets_dropped_total",
        "container_network_transmit_packets_total",
    ]

    network_metrics = [
        # network
        "container_network_receive_errors_total",
        "container_network_receive_packets_dropped_total",
        "container_network_receive_packets_total",
        "container_network_receive_bytes_total",
        "container_network_transmit_bytes_total",
        "container_network_transmit_errors_total",
        "container_network_transmit_packets_dropped_total",
        "container_network_transmit_packets_total",
    ]   

    def __init__(self, url: Optional[str] = None, namespace: Optional[str] = None) -> None:
        # Use ConfigManager if parameters not provided
        if not url or not namespace:
            config_manager = ConfigManager()
            config = config_manager.config
            url = url or config.prometheus_url
            namespace = namespace or config.target_namespace
        
        self.url = url
        # Create prometheus client
        try:
            self.prometheusClient = PrometheusConnect(self.url, disable_ssl=True)
        except Exception as e:
            logging.error("Error connecting to prometheus server: ", e)
        
        # Initialize base class with namespace
        super().__init__(namespace)
        
        # Get list of pods and services using inherited methods
        self.pods = self.get_pods_list()
        self.services = self.get_services_list()
    
    def get_pod_metrics(self, pod_name: str) -> Dict[str, Any]:
        """
            Get all metrics (no Istio) for given pods - INSTANT VALUES ONLY.
            
            Args:
                pod_name (str): Pod name
            
            Returns:
                dict: {resource_info, metrics: {metric_name: current_value}}
        """
        
        all_metrics = self.normal_metrics + self.network_metrics
        # Remove duplicates
        all_metrics = list(set(all_metrics))
        
        results = {
            "resource_type": "pod",
            "resource_namespace": self.namespace,
            "resource_name": pod_name,
            "metrics": {}
        }

        # Check if the pod exists
        if pod_name not in self.pods:
            results["error"] = f"The pod {pod_name} does not exist in the {self.namespace} namespace."
            return results

        for metric in all_metrics:
            try:
                # Instant query
                query = f'{metric}{{namespace="{self.namespace}", pod=~".*{pod_name}.*"}}'
                data = self.prometheusClient.custom_query(query=query)
                
                if data:
                    # Extract just the value from first result
                    if len(data) > 0 and 'value' in data[0]:
                        results["metrics"][metric] = float(data[0]['value'][1])
                    else:
                        results["metrics"][metric] = None
                else:
                    results["metrics"][metric] = None
                    
            except Exception as e:
                results["metrics"][metric] = f"Error: {str(e)}"

        return results
    
    def get_pod_metrics_range(self, pod_name: str, range_minutes: int, step: str = "1m") -> Dict[str, Any]:
        """
            Get all metrics (no Istio) for given pod over a time range - TIME SERIES VALUES.
            
            Args:
                pod_name (str): Pod name
                range_minutes (int): Time range in minutes from now backwards
                step (str): Query resolution step (e.g., "15s", "1m", "5m")
        """
        
        all_metrics = self.normal_metrics + self.network_metrics
        # Remove duplicates
        all_metrics = list(set(all_metrics))
        
        results = {
            "resource_type": "pod",
            "resource_namespace": self.namespace,
            "resource_name": pod_name,
            "time_range_minutes": range_minutes,
            "step": step,
            "metrics": {}
        }

        # Check if the pod exists
        if pod_name not in self.pods:
            results["error"] = f"The pod {pod_name} does not exist in the {self.namespace} namespace."
            return results
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=range_minutes)

        for metric in all_metrics:
            try:
                # Range query
                query = f'{metric}{{namespace="{self.namespace}", pod=~".*{pod_name}.*"}}'
                data = self.prometheusClient.custom_query_range(
                    query=query,
                    start_time=start_time,
                    end_time=end_time,
                    step=step
                )
                
                if data:
                    # Extract values from time series data
                    if len(data) > 0 and 'values' in data[0]:
                        # Extract just the values (not timestamps) from the time series
                        values = [float(value[1]) for value in data[0]['values']]
                        results["metrics"][metric] = values
                    else:
                        results["metrics"][metric] = None
                else:
                    results["metrics"][metric] = None
                    
            except Exception as e:
                results["metrics"][metric] = f"Error: {str(e)}"

        return results
    
    def get_pod_triage_metrics(self, pod_name: str) -> Dict[str, Any]:
        """
        Performs a simple triage based on universal, instant metrics without pod specs.

        This function checks for clear, high-confidence anomaly signals that do not require
        knowledge of the pod's resource limits. It's designed for a quick first-pass
        health check.

        Args:
            pod_name (str): The name of the pod to triage.

        Returns:
            dict: A dictionary containing the triage result.
                  {'is_anomalous': bool, 'reasons': [str], 'checked_metrics': {}}
        """
        pod_data = self.get_pod_metrics(pod_name)
        
        triage_result = {
            "is_anomalous": False,
            "reasons": [],
            "checked_metrics": pod_data.get("metrics", {})
        }

        if "error" in pod_data:
            triage_result["is_anomalous"] = True
            triage_result["reasons"].append(pod_data["error"])
            return triage_result

        metrics = pod_data.get("metrics", {})

        # Triage Rule 1: Thread Saturation
        threads = metrics.get("container_threads")
        threads_max = metrics.get("container_threads_max")
        if threads is not None and threads_max is not None and threads_max > 0:
            thread_ratio = threads / threads_max
            if thread_ratio > 0.95:
                triage_result["is_anomalous"] = True
                triage_result["reasons"].append(
                    f"CRITICAL: Thread usage is at {thread_ratio:.2%} of the maximum ({int(threads)}/{int(threads_max)}). Application may hang or crash."
                )

        # Triage Rule 2: High CPU Load
        cpu_load = metrics.get("container_cpu_load_average_10s")
        if cpu_load is not None and cpu_load > 10.0:
            triage_result["is_anomalous"] = True
            triage_result["reasons"].append(
                f"WARNING: High CPU load average of {cpu_load:.2f}. The CPU is likely saturated, causing high latency."
            )

        # Triage Rule 3: Network Errors & Drops
        # This checks if any errors have occurred during the pod's lifetime.
        network_checks = {
            "container_network_receive_errors_total": "receive errors",
            "container_network_transmit_errors_total": "transmit errors",
            "container_network_receive_packets_dropped_total": "dropped received packets",
            "container_network_transmit_packets_dropped_total": "dropped transmitted packets",
        }
        for metric, description in network_checks.items():
            value = metrics.get(metric)
            if value is not None and value > 1:
                triage_result["is_anomalous"] = True
                triage_result["reasons"].append(
                    f"INFO: Pod has a history of {int(value)} network {description}."
                )

        return triage_result