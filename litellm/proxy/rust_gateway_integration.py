"""
Rust Gateway Integration Module

Provides integration between the Python proxy and the Rust gateway.
The Rust gateway runs as a sidecar process and handles high-throughput
chat completion requests, while the Python proxy handles complex logic.
"""

import os
import subprocess
import time
import requests
from typing import Optional, Dict, Any
from litellm import verbose_logger


class RustGatewayManager:
    """Manages the Rust gateway sidecar process."""
    
    def __init__(
        self,
        binary_path: Optional[str] = None,
        port: int = 4001,
        host: str = "127.0.0.1",
        master_key: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.binary_path = binary_path or os.environ.get(
            "RUST_GATEWAY_BINARY",
            "litellm-rust/target/release/litellm-ai-gateway"
        )
        self.port = port
        self.host = host
        self.master_key = master_key or os.environ.get("LITELLM_MASTER_KEY")
        self.config_path = config_path or os.environ.get("LITELLM_YAML_CONFIG")
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"
        
    def start(self) -> bool:
        """Start the Rust gateway process."""
        if self.process is not None:
            verbose_logger.warning("Rust gateway already running")
            return True
            
        if not os.path.exists(self.binary_path):
            verbose_logger.error(f"Rust gateway binary not found: {self.binary_path}")
            return False
            
        env = os.environ.copy()
        if self.master_key:
            env["LITELLM_MASTER_KEY"] = self.master_key
        if self.config_path:
            env["LITELLM_YAML_CONFIG"] = self.config_path
        env["PORT"] = str(self.port)
        env["HOST"] = self.host
        
        try:
            self.process = subprocess.Popen(
                [self.binary_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            verbose_logger.info(f"Rust gateway started on {self.host}:{self.port}")
            
            # Wait for gateway to be ready
            for _ in range(10):
                if self.is_healthy():
                    return True
                time.sleep(0.5)
                
            verbose_logger.warning("Rust gateway started but not responding to health checks")
            return True
            
        except Exception as e:
            verbose_logger.error(f"Failed to start Rust gateway: {e}")
            self.process = None
            return False
            
    def stop(self):
        """Stop the Rust gateway process."""
        if self.process is not None:
            verbose_logger.info("Stopping Rust gateway")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            
    def is_healthy(self) -> bool:
        """Check if the Rust gateway is healthy."""
        if self.process is None:
            return False
            
        try:
            response = requests.get(
                f"{self.base_url}/health/liveness",
                timeout=2.0
            )
            return response.status_code == 200
        except Exception:
            return False
            
    def forward_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None,
    ) -> Optional[requests.Response]:
        """Forward a request to the Rust gateway."""
        if not self.is_healthy():
            return None
            
        try:
            url = f"{self.base_url}{path}"
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=body,
                timeout=30.0,
            )
            return response
        except Exception as e:
            verbose_logger.error(f"Failed to forward request to Rust gateway: {e}")
            return None


# Global instance
_rust_gateway: Optional[RustGatewayManager] = None


def get_rust_gateway() -> Optional[RustGatewayManager]:
    """Get the global Rust gateway instance."""
    global _rust_gateway
    return _rust_gateway


def init_rust_gateway(**kwargs) -> Optional[RustGatewayManager]:
    """Initialize and start the Rust gateway."""
    global _rust_gateway
    
    # Check if Rust gateway is enabled
    if not os.environ.get("ENABLE_RUST_GATEWAY", "").lower() in ("true", "1", "yes"):
        verbose_logger.info("Rust gateway not enabled (set ENABLE_RUST_GATEWAY=true)")
        return None
        
    _rust_gateway = RustGatewayManager(**kwargs)
    if _rust_gateway.start():
        verbose_logger.info("Rust gateway integration initialized")
        return _rust_gateway
    else:
        verbose_logger.error("Failed to initialize Rust gateway")
        _rust_gateway = None
        return None


def shutdown_rust_gateway():
    """Shutdown the Rust gateway."""
    global _rust_gateway
    if _rust_gateway is not None:
        _rust_gateway.stop()
        _rust_gateway = None
