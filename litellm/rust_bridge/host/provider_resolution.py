"""The Python tables and settings `get_llm_provider` reads, for the native resolver."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from operator import attrgetter
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # litellm module globals are untyped

import litellm
from litellm.constants import NADIR_DEFAULT_API_BASE, REPLICATE_MODEL_NAME_WITH_ID_LENGTH
from litellm.litellm_core_utils.fallback_generalizations import routing_generalization_rules
from litellm.litellm_core_utils.get_llm_provider_logic import provider_set_writes
from litellm.litellm_core_utils.model_cost_table import model_cost_writes
from litellm.litellm_core_utils.model_latest_aliases import latest_alias_rows
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.secret_managers.main import (
    _should_read_secret_from_secret_manager,  # pyright: ignore[reportPrivateUsage]  # the gate get_secret applies
)

_MODEL_LISTS: Final = (
    "open_ai_chat_completion_models",
    "openai_video_generation_models",
    "open_ai_text_completion_models",
    "anthropic_models",
    "cohere_models",
    "cohere_embedding_models",
    "cohere_chat_models",
    "replicate_models",
    "openrouter_models",
    "maritalk_models",
    "vertex_chat_models",
    "vertex_code_chat_models",
    "vertex_text_models",
    "vertex_code_text_models",
    "vertex_language_models",
    "vertex_embedding_models",
    "vertex_vision_models",
    "vertex_ai_image_models",
    "vertex_ai_video_models",
    "ai21_chat_models",
    "ai21_models",
    "aleph_alpha_models",
    "baseten_models",
    "nlp_cloud_models",
    "petals_models",
    "bedrock_models",
    "bedrock_embedding_models",
    "bedrock_converse_models",
    "watsonx_models",
    "open_ai_embedding_models",
    "empower_models",
    "gradient_ai_models",
    "mistral_chat_models",
    "provider_list",
    "model_list_set",
)
_SETTINGS: Final = attrgetter(
    "use_litellm_proxy", "suppress_debug_info", "secret_manager_client", "_key_management_settings"
)


def _strings(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(str.__str__(value) for value in values if isinstance(value, str))


def _model_list(name: str) -> Iterable[object]:
    return cast("Iterable[object]", getattr(litellm, name))  # cast-ok: every name in _MODEL_LISTS is a collection


def _custom_provider(item: object) -> str | None:
    if not isinstance(item, Mapping) or "provider" not in item:
        raise LookupError("provider")
    provider: Final[object] = cast("Mapping[str, object]", item)["provider"]  # cast-ok: keys are only compared
    return str.__str__(provider) if isinstance(provider, str) else None


def _custom_providers() -> tuple[str, ...] | None:
    try:
        providers: Final = tuple(map(_custom_provider, litellm.custom_provider_map))
    except LookupError:
        return None
    return tuple(provider for provider in providers if provider is not None)


def version() -> tuple[object, ...]:
    """Moves whenever a table or setting `snapshot` reads is replaced, resized or rewritten by LiteLLM."""
    model_cost: Final = cast("Mapping[str, object]", litellm.model_cost)  # cast-ok: only identity and size are read
    custom_provider_map: Final = litellm.custom_provider_map
    return (
        id(model_cost),
        len(model_cost),
        model_cost_writes(),
        provider_set_writes.count,
        custom_provider_map,
        len(custom_provider_map),
        *_SETTINGS(litellm),
        routing_generalization_rules(),
    )


def snapshot() -> Mapping[str, object]:
    """Every table and setting `get_llm_provider` reads."""
    return MappingProxyType(
        {
            "model_lists": {name: _strings(_model_list(name)) for name in _MODEL_LISTS},
            "openai_compatible_endpoints": _strings(litellm.openai_compatible_endpoints),
            "json_providers": [
                {
                    "slug": provider.slug,
                    "base_url": provider.base_url,
                    "api_key_env": provider.api_key_env,
                    "api_base_env": provider.api_base_env,
                }
                for slug in JSONProviderRegistry.list_providers()
                if (provider := JSONProviderRegistry.get(slug)) is not None
            ],
            "custom_providers": _custom_providers(),
            "latest_aliases": [
                {"key": key, "target": target, "provider": provider} for key, target, provider in latest_alias_rows()
            ],
            "routing_rules": [
                {"pattern": pattern, "provider": provider} for pattern, provider in routing_generalization_rules()
            ],
            "use_litellm_proxy": litellm.use_litellm_proxy is True,
            "suppress_debug_info": litellm.suppress_debug_info is not False,
            "secret_manager": _should_read_secret_from_secret_manager(),
            "replicate_model_name_with_id_length": REPLICATE_MODEL_NAME_WITH_ID_LENGTH,
            "nadir_default_api_base": NADIR_DEFAULT_API_BASE,
            "nscale_api_base_url": litellm.NscaleConfig.API_BASE_URL,
        }
    )
