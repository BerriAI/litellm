"""
Schema definitions for JSON-driven endpoint configuration.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class Operation(BaseModel):
    """Defines a single API operation (generate, edit, etc.)."""
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    path: str
    path_params: List[str] = []


class AuthConfig(BaseModel):
    """Defines authentication configuration."""
    type: Literal["query_param", "header", "bearer"]
    param_name: Optional[str] = None
    header_name: Optional[str] = None
    env_vars: List[str]
    gcloud_auth: bool = False


class EndpointDefinition(BaseModel):
    """Complete definition of an API endpoint."""
    endpoint_name: str
    provider: str
    base_url: str
    api_version: str
    operations: Dict[str, Operation]
    auth: AuthConfig
    defaults: Optional[Dict[str, str]] = None
