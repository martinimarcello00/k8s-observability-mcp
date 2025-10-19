from typing import Optional
from base_k8s_client import BaseK8sClient

class LogAPI(BaseK8sClient):
    def __init__(self, namespace: Optional[str] = None):
        super().__init__(namespace)
        # Initialize pod and service lists using inherited methods
        self.pods = self.get_pods_list()
        self.services = self.get_services_list()

    def get_pods_from_service(self, service: str):
        """Return all the pods connected to a service"""
        results = {
            "service_name": service,
            "namespace": self.namespace,
            "pods": []
        }
        
        if service not in self.services:
            results["error"] = f"The service {service} does not exist in the {self.namespace} namespace."
            return results
        
        try:
            requested_svc = self.k8s_client.read_namespaced_service(service, self.namespace)
            # Kubernetes objects have complex types, use type: ignore
            if not hasattr(requested_svc, 'spec') or not hasattr(requested_svc.spec, 'selector'):  # type: ignore
                results["error"] = f"Service {service} does not have a valid selector."
                return results
                
            selector = requested_svc.spec.selector  # type: ignore
            if not selector:
                results["error"] = f"Service {service} has no selector configured."
                return results
                
            label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])
            pods = self.k8s_client.list_namespaced_pod(self.namespace, label_selector=label_selector)
            
            results["pods"] = [
                {
                    "pod_name": pod.metadata.name,  # type: ignore
                    "pod_status": pod.status.phase  # type: ignore
                }
                for pod in pods.items  # type: ignore
            ]
        except Exception as e:
            results["error"] = f"Failed to get pods for service {service}: {str(e)}"
            
        return results
    
    def get_pod_logs(self, pod_name: str, tail: int = 100, important: bool = True) -> str:
        # Check if the pod exists
        if pod_name not in self.pods:
            return f"The pod {pod_name} does not exist in the {self.namespace} namespace."
        
        try:
            logs = self.k8s_client.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=tail,
            )
        except Exception as e:
            return f"Failed to get logs for pod {pod_name}: {str(e)}"

        if important:
            # Split logs into lines
            log_lines = logs.split('\n')

            # Extended list of important keywords
            important_keywords = [
                "ERROR", "WARN", "CRITICAL", "FATAL", "PANIC",
                "EXCEPTION", "FAILURE", "FAILED", "TIMEOUT",
                "REFUSED", "DENIED", "UNREACHABLE", "RESTART",
                "CRASH", "KILLED", "OOM", "5xx", "500", "503", "502",
                "4xx", "401", "403", "404", "CONNECTION", "DISK"
            ]

            # Return only the log lines that contains the important keywords (case-insensitive)
            filtered_logs = [line for line in log_lines if any(keyword in line.upper() for keyword in important_keywords)]

            results = ""

            if len(filtered_logs) > 0:
                results = f"Found {len(filtered_logs)} important log entries:\n\n"
                results += "\n".join(filtered_logs)
            else:
                results += "No important log entries found, full log entries are appended\n"
                results += "\n".join(log_lines)
            
            return results
        else:
            return logs