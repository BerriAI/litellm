"""
JSON-based provider configuration loader for OpenAI-compatible providers.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.types.utils import ProviderSpecificModelInfo

_PROVIDER_INFO_ADAPTER: Final = TypeAdapter(ProviderSpecificModelInfo)
_SPECIAL_HANDLING_ADAPTER: Final = TypeAdapter(Mapping[str, bool])
_MODEL_ENDPOINTS_ADAPTER: Final = TypeAdapter(Mapping[str, tuple[str, ...]])


class SimpleProviderConfig:
    """Simple data class for JSON provider config"""

    def __init__(self, slug: str, data: dict):
        self.slug = slug
        self.base_url = data["base_url"]
        self.api_key_env: str = data["api_key_env"]
        self.api_base_env: str | None = data.get("api_base_env")
        self.base_class = data.get("base_class", "openai_gpt")
        self.param_mappings = data.get("param_mappings", {})
        self.constraints = data.get("constraints", {})
        self.special_handling: Mapping[str, bool] = _SPECIAL_HANDLING_ADAPTER.validate_python(
            data.get("special_handling", {})
        )
        self.supported_endpoints: Sequence[str] = data.get("supported_endpoints", [])
        self.model_supported_endpoints: Mapping[str, tuple[str, ...]] = _MODEL_ENDPOINTS_ADAPTER.validate_python(
            data.get("model_supported_endpoints", {})
        )
        self.provider_info: ProviderSpecificModelInfo = (
            _PROVIDER_INFO_ADAPTER.validate_python(data["provider_info"]) if "provider_info" in data else {}
        )


class JSONProviderRegistry:
    """Load providers from JSON once on import"""

    _providers: dict[str, SimpleProviderConfig] = {}
    _loaded = False

    @classmethod
    def load(cls):
        """Load providers from JSON configuration file"""
        if cls._loaded:
            return

        json_path: Final = Path(__file__).parent / "providers.json"

        if not json_path.exists():
            # No JSON file yet, that's okay
            cls._loaded = True
            return

        try:
            with open(json_path) as f:
                data: Final = json.load(f)

            for slug, config in data.items():
                cls._providers[slug] = SimpleProviderConfig(slug, config)

            cls._loaded = True
        except Exception as e:
            verbose_logger.warning("Warning: Failed to load JSON provider configs: %s", e)
            cls._loaded = True

    @classmethod
    def get(cls, slug: str) -> SimpleProviderConfig | None:
        """Get a provider configuration by slug"""
        return cls._providers.get(slug)

    @classmethod
    def exists(cls, slug: str) -> bool:
        """Check if a provider is defined via JSON"""
        return slug in cls._providers

    @classmethod
    def get_by_base_url(cls, base_url: str) -> SimpleProviderConfig | None:
        """Get a provider configuration by its default base url"""
        return next((provider for provider in cls._providers.values() if provider.base_url == base_url), None)

    @classmethod
    def supports_responses_api(cls, slug: str) -> bool:
        """Check if a JSON provider supports the Responses API"""
        provider: Final = cls._providers.get(slug)
        if provider is None:
            return False
        return "/v1/responses" in provider.supported_endpoints

    @classmethod
    def supports_embeddings(cls, slug: str) -> bool:
        provider: Final = cls._providers.get(slug)
        if provider is None:
            return False
        return "/v1/embeddings" in provider.supported_endpoints

    @classmethod
    def serves_endpoint(cls, slug: str, model: str | None, endpoint: str) -> bool:
        provider: Final = cls._providers.get(slug)
        if provider is None:
            return False
        model_endpoints: Final = (
            provider.model_supported_endpoints.get(model.removeprefix(f"{slug}/"), ()) if model else ()
        )
        return endpoint in provider.supported_endpoints or endpoint in model_endpoints

    @classmethod
    def list_providers(cls) -> list:
        """List all registered provider slugs"""
        return list(cls._providers.keys())


# Load on import
JSONProviderRegistry.load()
