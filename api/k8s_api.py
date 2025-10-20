import logging
from typing import Optional, Dict, List, Any
from .base_k8s_client import BaseK8sClient
from .config_manager import ConfigManager
import json

class K8sAPI(BaseK8sClient):
    def __init__(self, namespace: Optional[str] = None):
        config_manager = ConfigManager()
        self._target_namespace = namespace or config_manager.config.target_namespace
        super().__init__(self._target_namespace)
        self.services = self.get_services_list()
        self.pods = self.get_pods_list()

    def get_problematic_pods(self) -> dict:
        """
        Scans a Kubernetes namespace for problematic pods and returns them
        in a JSON format suitable for an LLM.

        Args:
            namespace: The Kubernetes namespace to scan.

        Returns:
            A JSON string listing problematic pods and their issues.
        """
        
        report = {}
        report["problematic_pods"] = []
        
        try:
            pod_list = self.k8s_client.list_namespaced_pod(self.namespace)
        except Exception as e:
            return {"error": f"Could not list pods in namespace '{self.namespace}': {e.body}"} #type: ignore

        for pod in pod_list.items:
            pod_issues = []

            # A pod might not have container_statuses if it's still pending scheduling
            if not pod.status.container_statuses:
                # Check if the pod is stuck in a Pending state for a non-container reason
                if pod.status.phase == 'Pending':
                    pod_issues.append({
                        "container_name": "N/A",
                        "issue_type": "Pod Pending",
                        "reason": pod.status.reason or "Unknown",
                        "message": pod.status.message or "Waiting for scheduling or resources."
                    })
                continue # Go to the next pod

            for container in pod.status.container_statuses:
                # 1. Check for containers in a waiting state
                if container.state.waiting:
                    pod_issues.append({
                        "container_name": container.name,
                        "issue_type": "Waiting",
                        "reason": container.state.waiting.reason,
                        "message": container.state.waiting.message,
                        "restart_count": container.restart_count
                    })

                # 2. Check for containers that terminated with an error code
                elif container.state.terminated and container.state.terminated.exit_code != 0:
                    pod_issues.append({
                        "container_name": container.name,
                        "issue_type": "Terminated With Error",
                        "reason": container.state.terminated.reason,
                        "message": container.state.terminated.message,
                        "exit_code": container.state.terminated.exit_code,
                        "restart_count": container.restart_count
                    })

                # 3. Check for high restart counts (a strong sign of a crash loop)
                # A threshold like 3 is good to avoid flagging pods that had a single transient restart.
                elif container.restart_count > 3:
                    # Often, a crashing container will be in a 'Running' state just before
                    # the next crash, so we check this separately.
                    reason = "High Restart Count"
                    # If the last state was termination, we can get more info
                    if container.last_state and container.last_state.terminated:
                        reason = container.last_state.terminated.reason or reason

                    pod_issues.append({
                        "container_name": container.name,
                        "issue_type": "High Restarts",
                        "reason": reason,
                        "message": "Container is restarting frequently, indicating a potential crash loop.",
                        "restart_count": container.restart_count
                    })

            # If we found any issues for this pod, add it to our report
            if pod_issues:
                report["problematic_pods"].append({
                    "pod_name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "pod_phase": pod.status.phase,
                    "container_issues": pod_issues
                })

        if len(report["problematic_pods"]) == 0:
            report["info"] = "No problematic pods detected based on status analysis. All pods appear healthy."
                
        return report

    