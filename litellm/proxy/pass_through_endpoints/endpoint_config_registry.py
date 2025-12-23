"""
Centralized registry for endpoint configurations loaded from JSON.

This module provides a declarative way to define pass-through endpoints using JSON configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger


class AuthConfig(BaseModel):
    """
    Authentication configuration for an endpoint.

    Supports multiple auth types:
    - bearer_token: Standard Bearer token in Authorization header
    - custom_header: Custom header with configurable name and format
    - query_param: API key passed as query parameter
    - custom_handler: Complex auth requiring custom Python handler
    """

    type: str = Field(
        ...,
        description="Auth type: bearer_token, custom_header, query_param, custom_handler",
    )
    env_var: str = Field(..., description="Environment variable for API key")
    header_name: Optional[str] = Field(None, description="Header name for auth")
    header_format: Optional[str] = Field(
        None, description="Format string for header value, e.g., 'Bearer {api_key}'"
    )
    param_name: Optional[str] = Field(None, description="Query param name for API key")
    handler_function: Optional[str] = Field(
        None, description="Custom auth handler function name"
    )


class StreamingConfig(BaseModel):
    """
    Streaming detection configuration.

    Defines how to detect if a request is streaming and how to handle it.
    """

    detection_method: str = Field(
        ...,
        description="Method: request_body_field, url_contains, header, or none",
    )
    field_name: Optional[str] = Field(None, description="Request body field to check")
    pattern: Optional[str] = Field(None, description="Pattern to match in URL")
    query_param_suffix: Optional[str] = Field(
        None, description="Query param to append for streaming (e.g., '?alt=sse')"
    )
    response_content_type: Optional[str] = Field(
        default="text/event-stream", description="Content-Type for streaming responses"
    )


class FeaturesConfig(BaseModel):
    """
    Feature flags for endpoint configuration.

    Controls various endpoint behaviors like header forwarding, auth requirements, etc.
    """

    forward_headers: bool = Field(
        default=False, description="Forward incoming request headers to target"
    )
    merge_query_params: bool = Field(
        default=False, description="Merge query params from request"
    )
    require_litellm_auth: bool = Field(
        default=True, description="Require LiteLLM API key authentication"
    )
    subpath_routing: bool = Field(
        default=True, description="Support wildcard subpath routing"
    )
    custom_auth_handler: bool = Field(
        default=False, description="Use custom authentication handler"
    )
    dynamic_base_url: bool = Field(
        default=False,
        description="Base URL is dynamically constructed (e.g., from request params)",
    )
    custom_query_params: bool = Field(
        default=False, description="Endpoint uses custom query param handling"
    )


class UrlTransformationConfig(BaseModel):
    """
    Configuration for URL transformation and parameter extraction.

    Used for complex endpoints that need to extract parameters from URLs
    or inject parameters into target URLs.
    """

    extract_params: Optional[Dict[str, Dict[str, str]]] = Field(
        None, description="Parameters to extract from URL using regex"
    )
    inject_params_to_url: bool = Field(
        default=False, description="Inject extracted params into target URL"
    )


class AuthExtractionConfig(BaseModel):
    """
    Configuration for extracting authentication from multiple sources.

    Allows checking multiple sources for auth credentials (query params, headers, etc.)
    """

    from_query_param: Optional[str] = Field(
        None, description="Query parameter name to extract auth from"
    )
    from_header: Optional[str] = Field(
        None, description="Header name to extract auth from"
    )


class EndpointConfig(BaseModel):
    """
    Complete endpoint configuration.

    Defines all aspects of a pass-through endpoint including routing, authentication,
    streaming support, and feature flags.
    """

    provider_slug: str = Field(..., description="Unique identifier for the provider")
    route_prefix: str = Field(..., description="FastAPI route prefix (e.g., '/provider/{endpoint:path}')")
    target_base_url: Optional[str] = Field(
        None, description="Static base URL for the provider API"
    )
    target_base_url_template: Optional[str] = Field(
        None,
        description="Template for dynamic base URL (e.g., 'https://{location}-api.example.com')",
    )
    target_base_url_env: Optional[str] = Field(
        None, description="Environment variable for base URL override"
    )
    auth: AuthConfig = Field(..., description="Authentication configuration")
    streaming: StreamingConfig = Field(..., description="Streaming configuration")
    features: FeaturesConfig = Field(
        default_factory=FeaturesConfig, description="Feature flags"
    )
    tags: List[str] = Field(
        default_factory=lambda: ["pass-through"],
        description="OpenAPI tags for documentation",
    )
    docs_url: Optional[str] = Field(
        None, description="Link to provider documentation"
    )
    custom_transformations: Optional[Dict[str, Any]] = Field(
        None, description="Custom transformation functions for request/response"
    )
    url_transformation: Optional[UrlTransformationConfig] = Field(
        None, description="URL transformation configuration"
    )
    auth_extraction: Optional[AuthExtractionConfig] = Field(
        None, description="Auth extraction from multiple sources"
    )


class EndpointConfigRegistry:
    """
    Registry for endpoint configurations loaded from JSON.

    Loads endpoint configurations once on startup and provides methods to
    retrieve and list configurations.
    """

    _configs: Dict[str, EndpointConfig] = {}
    _loaded = False

    @classmethod
    def load(cls, config_path: Optional[Path] = None):
        """
        Load endpoint configurations from JSON file.

        Args:
            config_path: Optional path to JSON config file. If None, uses default path.
        """
        if cls._loaded:
            return

        if config_path is None:
            config_path = Path(__file__).parent / "endpoints_config.json"

        if not config_path.exists():
            verbose_proxy_logger.debug(
                f"Endpoint config file not found at {config_path}, skipping JSON-based endpoint registration"
            )
            cls._loaded = True
            return

        try:
            with open(config_path) as f:
                data = json.load(f)

            for slug, config_data in data.items():
                config_data["provider_slug"] = slug
                cls._configs[slug] = EndpointConfig(**config_data)

            cls._loaded = True
            verbose_proxy_logger.info(
                f"Loaded {len(cls._configs)} endpoint configurations from JSON"
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"Failed to load endpoint configs from {config_path}: {e}"
            )
            cls._loaded = True

    @classmethod
    def get(cls, slug: str) -> Optional[EndpointConfig]:
        """
        Get endpoint configuration by provider slug.

        Args:
            slug: Provider slug (e.g., 'cohere', 'anthropic')

        Returns:
            EndpointConfig if found, None otherwise
        """
        return cls._configs.get(slug)

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider slugs.

        Returns:
            List of provider slug strings
        """
        return list(cls._configs.keys())

    @classmethod
    def get_all_configs(cls) -> Dict[str, EndpointConfig]:
        """
        Get all endpoint configurations.

        Returns:
            Dictionary mapping provider slugs to EndpointConfig objects
        """
        return cls._configs.copy()

    @classmethod
    def reload(cls, config_path: Optional[Path] = None):
        """
        Reload endpoint configurations from JSON file.

        Useful for hot-reloading configurations during development.

        Args:
            config_path: Optional path to JSON config file
        """
        cls._loaded = False
        cls._configs.clear()
        cls.load(config_path)


# Load configurations on module import
EndpointConfigRegistry.load()
