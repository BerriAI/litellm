"""
AgentOps integration for LiteLLM - Provides OpenTelemetry tracing for LLM calls
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

@dataclass
class AgentOpsConfig:
    endpoint: str = "https://otlp.agentops.cloud/v1/traces"
    api_key: Optional[str] = None
    service_name: Optional[str] = None
    deployment_environment: Optional[str] = None
    auth_endpoint: str = "https://api.agentops.ai/v3/auth/token"

    @classmethod
    def from_env(cls):
        return cls(
            endpoint="https://otlp.agentops.cloud/v1/traces",
            api_key=os.getenv("AGENTOPS_API_KEY"),
            service_name=os.getenv("AGENTOPS_SERVICE_NAME", "agentops"),
            deployment_environment=os.getenv("AGENTOPS_ENVIRONMENT", "production"),
            auth_endpoint="https://api.agentops.ai/v3/auth/token"
        )

class AgentOps(OpenTelemetry):
    """
    AgentOps integration - built on top of OpenTelemetry

    Example usage:
        ```python
        import litellm
        
        litellm.success_callback = ["agentops"]

        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        ```
    """
    def __init__(
        self,
        config: Optional[AgentOpsConfig] = None,
    ):
        if config is None:
            config = AgentOpsConfig.from_env()

        # Prefetch JWT token for authentication
        jwt_token = None
        project_id = None
        if config.api_key:
            try:
                response = self._fetch_auth_token(config.api_key, config.auth_endpoint)
                jwt_token = response.get("token")
                project_id = response.get("project_id")
            except Exception:
                pass

        headers = f"Authorization=Bearer {jwt_token}" if jwt_token else None
        
        otel_config = OpenTelemetryConfig(
            exporter="otlp_http",
            endpoint=config.endpoint,
            headers=headers
        )

        # Initialize OpenTelemetry with our config
        super().__init__(
            config=otel_config,
            callback_name="agentops"
        )

        # Set AgentOps-specific resource attributes
        resource_attrs = {
            "service.name": config.service_name or "litellm",
            "deployment.environment": config.deployment_environment or "production",
            "telemetry.sdk.name": "agentops",
        }
        
        if project_id:
            resource_attrs["project.id"] = project_id
            
        self.resource_attributes = resource_attrs

    def _fetch_auth_token(self, api_key: str, auth_endpoint: str) -> Dict[str, Any]:
        """
        Fetch JWT authentication token from AgentOps API
        
        Args:
            api_key: AgentOps API key
            auth_endpoint: Authentication endpoint
            
        Returns:
            Dict containing JWT token and project ID
        """
        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }
        
        client = _get_httpx_client()
        try:
            response = client.post(
                url=auth_endpoint,
                headers=headers,
                json={"api_key": api_key},
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch auth token: {response.text}")
            
            return response.json()
        finally:
            client.close() 