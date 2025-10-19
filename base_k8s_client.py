from kubernetes import client, config
from abc import ABC
import os
import logging
from typing import Optional

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
                logging.error(f"Failed to initialize Kubernetes client: {e}")
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
                logging.error(f"Failed to get services list: {e}")
                return []
        
        return self._services_cache
    
    def get_pods_list(self, use_cache: bool = True):
        """Get all pod names in the namespace with caching"""
        if not use_cache or self._pods_cache is None:
            try:
                pod_list = self.k8s_client.list_namespaced_pod(self.namespace)
                self._pods_cache = [pod.metadata.name for pod in pod_list.items]
            except Exception as e:
                logging.error(f"Failed to get pods list: {e}")
                return []
        
        return self._pods_cache
    
    def refresh_cache(self):
        """Refresh the cached services and pods"""
        self._services_cache = None
        self._pods_cache = None