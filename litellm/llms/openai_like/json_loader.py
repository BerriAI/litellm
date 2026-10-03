"""
JSON-based provider configuration loader for OpenAI-compatible providers.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_logger
from litellm.constants import openai_compatible_providers

OPENAI_AUDIO_ENDPOINTS: Final = frozenset({"/v1/audio/transcriptions", "/v1/audio/speech"})


class SimpleProviderConfig:
    """Simple data class for JSON provider config"""

    def __init__(self, slug: str, data: dict):
        self.slug = slug
        self.base_url = data["base_url"]
        self.api_key_env = data["api_key_env"]
        self.api_base_env = data.get("api_base_env")
        self.base_class = data.get("base_class", "openai_gpt")
        self.param_mappings = data.get("param_mappings", {})
        self.constraints = data.get("constraints", {})
        self.special_handling = data.get("special_handling", {})
        self.supported_endpoints: Final[Sequence[str]] = data.get("supported_endpoints", [])


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
    def declared_endpoints(cls) -> Mapping[str, Sequence[str]]:
        """Endpoints each JSON provider declares for itself"""
        return MappingProxyType({slug: provider.supported_endpoints for slug, provider in cls._providers.items()})

    @classmethod
    def list_providers(cls) -> list:
        """List all registered provider slugs"""
        return list(cls._providers.keys())


# Load on import
JSONProviderRegistry.load()


def derive_openai_audio_transcription_providers(
    compatible_providers: Sequence[str],
    declared_endpoints: Mapping[str, Sequence[str]],
) -> frozenset[str]:
    """Providers allowed to drive the OpenAI `/v1/audio/*` transport.

    A JSON provider joins only when it declares an audio endpoint for itself, so a chat-only
    provider can never be sent the caller's OpenAI credentials against its third-party base url.
    """
    non_json_providers: Final = frozenset(
        provider for provider in compatible_providers if provider not in declared_endpoints
    )
    declaring_providers: Final = frozenset(
        slug
        for slug, endpoints in declared_endpoints.items()
        if any(endpoint in OPENAI_AUDIO_ENDPOINTS for endpoint in endpoints)
    )
    return frozenset({"openai"}) | non_json_providers | declaring_providers


OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS: Final = derive_openai_audio_transcription_providers(
    compatible_providers=openai_compatible_providers,
    declared_endpoints=JSONProviderRegistry.declared_endpoints(),
)
