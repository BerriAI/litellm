"""
JSON-based provider configuration loader for OpenAI-compatible providers.
"""

import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

from pydantic import Field, TypeAdapter

from litellm._logging import verbose_logger
from litellm.types.utils import ProviderSpecificModelInfo

_PROVIDER_INFO_ADAPTER: Final = TypeAdapter(ProviderSpecificModelInfo)
_SPECIAL_HANDLING_ADAPTER: Final = TypeAdapter(Mapping[str, bool])
_STR_TUPLE_ADAPTER: Final = TypeAdapter(tuple[str, ...])
_MODEL_SPECIAL_HANDLING_ADAPTER: Final = TypeAdapter(Mapping[str, Mapping[str, bool]])
AcceptedValue: TypeAlias = bool | int | float | str
_ACCEPTED_VALUES_ADAPTER: Final = TypeAdapter(Mapping[str, tuple[AcceptedValue, ...]])
_MODEL_UNRESTRICTED_PARAMS_ADAPTER: Final = TypeAdapter(Mapping[str, frozenset[str]])
_API_KEY_ENV_ADAPTER: Final[TypeAdapter[str | tuple[str, ...]]] = TypeAdapter(
    str | Annotated[tuple[str, ...], Field(min_length=1)]
)


def _matching_model_key(keys: Collection[str], name: str) -> str | None:
    if name in keys:
        return name
    return next((key for key in keys if key.endswith("/*") and name.startswith(key.removesuffix("*"))), None)


class SimpleProviderConfig:
    """Simple data class for JSON provider config"""

    def __init__(self, slug: str, data: dict):
        self.slug = slug
        self.base_url = data["base_url"]
        api_key_env: Final = _API_KEY_ENV_ADAPTER.validate_python(data["api_key_env"])
        self.api_key_envs: Final[tuple[str, ...]] = (api_key_env,) if isinstance(api_key_env, str) else api_key_env
        self.api_key_env: str = self.api_key_envs[0]
        self.api_base_env = data.get("api_base_env")
        self.base_class = data.get("base_class", "openai_gpt")
        self.param_mappings = data.get("param_mappings", {})
        self.constraints = data.get("constraints", {})
        self.special_handling: Mapping[str, bool] = _SPECIAL_HANDLING_ADAPTER.validate_python(
            data.get("special_handling", {})
        )
        self.unsupported_params: Final[frozenset[str]] = frozenset(
            _STR_TUPLE_ADAPTER.validate_python(data["unsupported_params"]) if "unsupported_params" in data else ()
        )
        self.extra_supported_params: Final[tuple[str, ...]] = (
            _STR_TUPLE_ADAPTER.validate_python(data["extra_supported_params"])
            if "extra_supported_params" in data
            else ()
        )
        self.responses_unsupported_params: Final[frozenset[str]] = frozenset(
            _STR_TUPLE_ADAPTER.validate_python(data["responses_unsupported_params"])
            if "responses_unsupported_params" in data
            else ()
        )
        self.accepted_values: Final[Mapping[str, tuple[AcceptedValue, ...]]] = (
            _ACCEPTED_VALUES_ADAPTER.validate_python(data["accepted_values"])
            if "accepted_values" in data
            else MappingProxyType({})
        )
        self.model_unrestricted_params: Final[Mapping[str, frozenset[str]]] = (
            _MODEL_UNRESTRICTED_PARAMS_ADAPTER.validate_python(data["model_unrestricted_params"])
            if "model_unrestricted_params" in data
            else MappingProxyType({})
        )
        self.model_special_handling: Final[Mapping[str, Mapping[str, bool]]] = (
            _MODEL_SPECIAL_HANDLING_ADAPTER.validate_python(data["model_special_handling"])
            if "model_special_handling" in data
            else MappingProxyType({})
        )
        self.supported_endpoints = data.get("supported_endpoints", [])
        self.provider_info: ProviderSpecificModelInfo = (
            _PROVIDER_INFO_ADAPTER.validate_python(data["provider_info"]) if "provider_info" in data else {}
        )

    def api_key_from(self, lookup: Callable[[str], str | None]) -> str | None:
        return next(filter(None, map(lookup, self.api_key_envs)), None)

    def special_handling_for(self, model: str) -> Mapping[str, bool]:
        key: Final = _matching_model_key(self.model_special_handling, model.removeprefix(f"{self.slug}/"))
        if key is None:
            return self.special_handling
        return MappingProxyType({**self.special_handling, **self.model_special_handling[key]})

    def accepted_values_for(self, model: str) -> Mapping[str, tuple[AcceptedValue, ...]]:
        key: Final = _matching_model_key(self.model_unrestricted_params, model.removeprefix(f"{self.slug}/"))
        if key is None:
            return self.accepted_values
        unrestricted: Final = self.model_unrestricted_params[key]
        return MappingProxyType(
            {param: values for param, values in self.accepted_values.items() if param not in unrestricted}
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
    def list_providers(cls) -> list:
        """List all registered provider slugs"""
        return list(cls._providers.keys())


# Load on import
JSONProviderRegistry.load()
