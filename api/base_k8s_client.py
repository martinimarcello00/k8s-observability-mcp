from kubernetes import client, config
from abc import ABC
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class BaseK8sClient(ABC):
    """Base class for Kubernetes API interactions"""
    
    def __init__(self, namespace: Optional[str] = None):
        self.namespace = namespace or os.environ.get("TARGET_NAMESPACE", "default")
        self._k8s_client = None
        self._services_cache = None
        self._pods_cache = None
    
    @property
    def k8s_client(self):
        """Lazy initialization of Kubernetes client"""
        if self._k8s_client is None:
            try:
                config.load_kube_config()
                self._k8s_client = client.CoreV1Api()
            except Exception as e:
                logger.error(f"Failed to initialize Kubernetes client: {e}")
                raise
        return self._k8s_client
    
    def get_services_list(self, use_cache: bool = True):
        """Get all service names in the namespace with caching"""
        if not use_cache or self._services_cache is None:
            try:
                if self.namespace:
                    service_list = self.k8s_client.list_namespaced_service(self.namespace)
                else:
                    service_list = self.k8s_client.list_service_for_all_namespaces()
                
                self._services_cache = [service.metadata.name for service in service_list.items]
            except Exception as e:
                logger.error(f"Failed to get services list: {e}")
                return []
        
        return self._services_cache
    
    def get_pods_list(self, use_cache: bool = True):
        """Get all pod names in the namespace with caching"""
        if not use_cache or self._pods_cache is None:
            try:
                pod_list = self.k8s_client.list_namespaced_pod(self.namespace)
                self._pods_cache = [pod.metadata.name for pod in pod_list.items]
            except Exception as e:
                logger.error(f"Failed to get pods list: {e}")
                return []
        
        return self._pods_cache

    def get_pods_from_service(self, service: str):
        """Return all the pods connected to a service"""
        results = {
            "service_name": service,
            "namespace": self.namespace,
            "pods": []
        }
        
        if service not in self.get_services_list():
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
    
    def get_services_from_pod(self, pod_name: str):
        """Return the service(s) that select a given pod."""

        results = {
            "pod_name": pod_name,
            "namespace": self.namespace,
            "services": []
        }
        
        if pod_name not in self.get_pods_list():
            results["error"] = f"The pod {pod_name} does not exist in the {self.namespace} namespace."
            return results
        
        try:
            # Get the pod and its labels
            pod = self.k8s_client.read_namespaced_pod(pod_name, self.namespace)
            pod_labels = pod.metadata.labels  # type: ignore
            
            if not pod_labels:
                results["error"] = f"Pod {pod_name} has no labels."
                return results
            
            # Get all services in the namespace
            service_list = self.k8s_client.list_namespaced_service(self.namespace)
            
            # Check which services select this pod
            for service in service_list.items:  # type: ignore
                if not hasattr(service, 'spec') or not hasattr(service.spec, 'selector'):  # type: ignore
                    continue
                
                selector = service.spec.selector  # type: ignore
                if not selector:
                    continue
                
                # Check if all selector labels match the pod's labels
                if all(pod_labels.get(k) == v for k, v in selector.items()):
                    results["services"].append({
                        "service_name": service.metadata.name,  # type: ignore
                        "selector": selector
                    })
            
            if not results["services"]:
                results["info"] = f"No services found selecting pod {pod_name}."
                
        except Exception as e:
            results["error"] = f"Failed to get services for pod {pod_name}: {str(e)}"
            
        return results
    
    def refresh_cache(self):
        """Refresh the cached services and pods"""
        self._services_cache = None
        self._pods_cache = None