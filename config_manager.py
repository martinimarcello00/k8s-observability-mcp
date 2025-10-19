import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    target_namespace: str
    prometheus_url: str
    jaeger_url: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

class ConfigManager:
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            load_dotenv()
            self._config = Config(
                target_namespace=os.environ.get("TARGET_NAMESPACE", "default"),
                prometheus_url=os.environ.get("PROMETHEUS_SERVER_URL", "http://localhost:9090"),
                jaeger_url=os.environ.get("JAEGER_URL", "http://localhost:16686"),
                neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
                neo4j_password=os.environ.get("NEO4J_PASSWORD", "neo4j")
            )
    
    @property
    def config(self) -> Config:
        if self._config is None:
            raise RuntimeError("Configuration not initialized")
        return self._config