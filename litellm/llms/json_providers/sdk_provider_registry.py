"""
SDK-level provider configuration registry with transformation support.

This module provides Pydantic models and registry for JSON-configured providers
at the SDK level (not just proxy pass-through).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from litellm._logging import verbose_logger


class AuthenticationConfig(BaseModel):
    """
    Authentication configuration for provider.
    
    Supports multiple auth types:
    - bearer_token: Authorization: Bearer <token>
    - query_param: ?key=<token>
    - custom_header: Custom header name
    """

    type: str = Field(..., description="Auth type: bearer_token, query_param, custom_header")
    env_var: str = Field(..., description="Environment variable containing API key")
    param_name: Optional[str] = Field(None, description="Query param name (for query_param type)")
    header_name: Optional[str] = Field(None, description="Header name (for custom_header type)")


class EndpointConfig(BaseModel):
    """API endpoint configuration"""

    path: str = Field(..., description="URL path with optional {model} placeholder")
    method: str = Field(default="POST", description="HTTP method")
    supported_models: List[str] = Field(default_factory=list, description="List of supported model names")


class TransformationConfig(BaseModel):
    """
    Transformation configuration for request/response conversion.
    
    Supports:
    - jinja: Jinja2 template rendering
    - jsonpath: JSONPath field extraction
    - function: Python function (module.function)
    """

    type: str = Field(..., description="Transformation type: jinja, jsonpath, function")
    template: Optional[Dict[str, Any]] = Field(None, description="Jinja2 template (for type=jinja)")
    mappings: Optional[Dict[str, str]] = Field(None, description="JSONPath mappings (for type=jsonpath)")
    module: Optional[str] = Field(None, description="Python module (for type=function)")
    function: Optional[str] = Field(None, description="Function name (for type=function)")
    filters: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Custom Jinja filters")


class CostTrackingConfig(BaseModel):
    """
    Cost tracking configuration.
    
    Supports per-image and per-token pricing.
    """

    enabled: bool = Field(default=True, description="Enable cost tracking")
    cost_per_image: Dict[str, float] = Field(
        default_factory=dict, description="Cost per image for each model"
    )
    cost_per_token: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Cost per token (prompt/completion) for each model"
    )
    unit: str = Field(default="per_image", description="Cost unit: per_image, per_token")


class SDKProviderConfig(BaseModel):
    """
    Complete SDK provider configuration.
    
    This defines everything needed to integrate a provider with LiteLLM SDK:
    - API endpoints and authentication
    - Request/response transformations
    - Cost tracking
    - Provider capabilities
    """

    provider_name: str = Field(..., description="Unique provider identifier")
    provider_type: str = Field(
        ..., description="Provider type: image_generation, chat, embeddings, etc."
    )
    api_base: str = Field(..., description="Base API URL")
    api_base_env: Optional[str] = Field(
        None, description="Environment variable for API base override"
    )
    authentication: AuthenticationConfig = Field(..., description="Authentication configuration")
    endpoints: Dict[str, EndpointConfig] = Field(
        ..., description="API endpoints (generate, chat, embed, etc.)"
    )
    transformations: Dict[str, TransformationConfig] = Field(
        ..., description="Request/response transformations"
    )
    cost_tracking: CostTrackingConfig = Field(
        default_factory=CostTrackingConfig, description="Cost tracking configuration"
    )
    capabilities: Dict[str, Any] = Field(
        default_factory=dict, description="Provider capabilities and limits"
    )


class SDKProviderRegistry:
    """
    Registry for SDK-level provider configurations.
    
    Loads provider configurations from JSON and provides methods to
    retrieve and list configurations.
    """

    _configs: Dict[str, SDKProviderConfig] = {}
    _loaded = False

    @classmethod
    def load(cls, config_path: Optional[Path] = None):
        """
        Load SDK provider configurations from JSON file.
        
        Args:
            config_path: Optional path to JSON config file. If None, uses default path.
        """
        if cls._loaded:
            return

        if config_path is None:
            config_path = Path(__file__).parent / "sdk_providers.json"

        if not config_path.exists():
            verbose_logger.debug(
                f"SDK provider config file not found at {config_path}, skipping JSON-based SDK providers"
            )
            cls._loaded = True
            return

        try:
            with open(config_path) as f:
                data = json.load(f)

            # Skip metadata fields
            for slug, config_data in data.items():
                if slug.startswith("_"):
                    continue
                cls._configs[slug] = SDKProviderConfig(**config_data)

            cls._loaded = True
            verbose_logger.info(f"Loaded {len(cls._configs)} SDK provider configurations from JSON")
        except Exception as e:
            verbose_logger.error(f"Failed to load SDK provider configs from {config_path}: {e}")
            cls._loaded = True

    @classmethod
    def get(cls, provider_name: str) -> Optional[SDKProviderConfig]:
        """
        Get provider configuration by name.
        
        Args:
            provider_name: Provider identifier (e.g., 'google_imagen')
        
        Returns:
            SDKProviderConfig if found, None otherwise
        """
        return cls._configs.get(provider_name)

    @classmethod
    def list_providers(cls, provider_type: Optional[str] = None) -> List[str]:
        """
        List registered providers, optionally filtered by type.
        
        Args:
            provider_type: Optional filter by provider type (e.g., 'image_generation')
        
        Returns:
            List of provider names
        """
        if provider_type:
            return [
                name
                for name, config in cls._configs.items()
                if config.provider_type == provider_type
            ]
        return list(cls._configs.keys())

    @classmethod
    def get_all_configs(cls) -> Dict[str, SDKProviderConfig]:
        """
        Get all provider configurations.
        
        Returns:
            Dictionary mapping provider names to configurations
        """
        return cls._configs.copy()

    @classmethod
    def reload(cls, config_path: Optional[Path] = None):
        """
        Reload provider configurations from JSON file.
        
        Args:
            config_path: Optional path to JSON config file
        """
        cls._loaded = False
        cls._configs.clear()
        cls.load(config_path)


# Load configurations on module import
SDKProviderRegistry.load()
