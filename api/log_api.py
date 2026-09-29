import re
from typing import Optional
from .base_k8s_client import BaseK8sClient

class LogAPI(BaseK8sClient):
    def __init__(self, namespace: Optional[str] = None):
        super().__init__(namespace)
        # Initialize pod and service lists using inherited methods
        self.pods = self.get_pods_list()
        self.services = self.get_services_list()
    
    def get_pod_logs(self, pod_name: str, tail: int = 100, important: bool = True) -> str:
        # Check if the pod exists
        if pod_name not in self.pods:
            return f"The pod {pod_name} does not exist in the {self.namespace} namespace."
        
        try:
            # Multi-container pods need the container name, otherwise the API returns 400
            containers = [c.name for c in self.k8s_client.read_namespaced_pod(pod_name, self.namespace).spec.containers]
            lines = []
            for container in containers:
                container_logs = self.k8s_client.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.namespace,
                    container=container,
                    tail_lines=tail,
                )
                # Tell containers apart only when there is more than one
                prefix = f"[{container}] " if len(containers) > 1 else ""
                lines += [prefix + line for line in container_logs.split('\n')]
            logs = "\n".join(lines)
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

            # Return only the log lines that contains the important keywords (case-insensitive).
            # Status codes (401, 5xx, ...) only as whole words: as substrings they match inside timestamps and IDs
            # ("time":1790674934741 contains 401). Text keywords stay substrings (WARN -> WARNING, OOM -> OOMKilled).
            pattern = re.compile("|".join(rf"\b{k}\b" if k[0].isdigit() else k for k in important_keywords), re.IGNORECASE)
            filtered_logs = [line for line in log_lines if pattern.search(line)]

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