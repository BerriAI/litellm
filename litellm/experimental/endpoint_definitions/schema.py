"""
Schema definitions for JSON-driven endpoint configuration.

This defines the structure of the JSON files that describe API endpoints.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class UsageMapping(BaseModel):
    """Maps response paths to standard usage fields."""
    input_tokens_path: str
    output_tokens_path: str
    total_tokens_path: Optional[str] = None
    cached_tokens_path: Optional[str] = None


class OperationRequest(BaseModel):
    """Defines request parameters for an operation."""
    required: List[str] = []
    required_one_of: List[str] = []
    optional: List[str] = []
    
    # Request body structure template (for transformation mode)
    # Example: {"instances": [{"prompt": "$prompt"}], "parameters": {"sampleCount": "$sample_count"}}
    body_template: Optional[Dict[str, Any]] = None
    
    # Request body schema (for documentation and passthrough mode)
    # When defined without body_template, kwargs are passed through as-is
    body_schema: Optional[Dict[str, Any]] = None


class OperationResponse(BaseModel):
    """Defines response type for an operation."""
    type: str  # Pydantic model name from types module
    streaming_type: Optional[str] = None
    
    # Response template showing how to extract/transform data
    # Example: {"images": "$predictions[*].bytesBase64Encoded"}
    response_template: Optional[Dict[str, Any]] = None
    
    # Response body schema (for documentation)
    # Defines the expected structure of the response body
    body_schema: Optional[Dict[str, Any]] = None


class Operation(BaseModel):
    """Defines a single API operation (create, get, delete, etc.)."""
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    path: str
    streaming_path: Optional[str] = None
    streaming_query_params: Optional[Dict[str, str]] = None
    path_params: List[str] = []
    supports_streaming: bool = False
    request: Optional[OperationRequest] = None
    response: OperationResponse
    usage: Optional[UsageMapping] = None


class AuthConfig(BaseModel):
    """Defines authentication configuration."""
    type: Literal["query_param", "header", "bearer"]
    param_name: Optional[str] = None
    header_name: Optional[str] = None
    env_vars: List[str]
    gcloud_auth: bool = False  # Use gcloud auth for bearer token


class EndpointDefinition(BaseModel):
    """
    Complete definition of an API endpoint.
    
    This is the root schema for JSON endpoint definition files.
    """
    endpoint_name: str
    provider: str
    base_url: str
    api_version: str
    operations: Dict[str, Operation]
    auth: AuthConfig
    defaults: Optional[Dict[str, str]] = None

