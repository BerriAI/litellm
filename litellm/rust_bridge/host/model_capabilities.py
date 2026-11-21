from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Final, TypeAlias

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils import fallback_generalizations
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.get_model_cost_map import FALLBACK_GENERALIZATIONS_KEY
from litellm.litellm_core_utils.model_cost_table import ModelCostVersion, model_cost_version

_DECODED_JSON: Final = TypeAdapter(object)


class _Held:
    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Held) and other.value is self.value


_CatalogVersion: TypeAlias = tuple[int, int, int]
_Fingerprint: TypeAlias = tuple[str, str | None, _Held, ModelCostVersion, int, tuple[_Held, ...]]


def _resolved_provider(model: str, custom_llm_provider: str | None) -> tuple[str, str]:
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # an unroutable model still shapes as a bare Anthropic id
        return model, custom_llm_provider or "anthropic"
    return resolved_model, provider


def _fingerprint(model: str, custom_llm_provider: str | None, resolved_model: str, provider: str) -> _Fingerprint:
    """Changes whenever an answer could: every write moves the cost map version, and replacing
    any entry the probes look up (``litellm.model_cost[key] = ...``) changes that entry's identity.
    The key holds the entries it names, so a cached answer's entries are never freed and their
    addresses never reused."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    model_cost: Final = as_str_mapping(litellm.model_cost) or {}  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # the model map is an untyped dict of rows
    keys: Final = dict.fromkeys(
        (
            *AnthropicModelInfo._model_map_lookup_candidates(model),  # pyright: ignore[reportPrivateUsage]  # the keys the capability walk reads
            resolved_model,
            f"{provider}/{resolved_model}",
        )
    )
    return (
        model,
        custom_llm_provider,
        _Held(model_cost),
        model_cost_version(),
        len(model_cost),
        tuple(_Held(model_cost.get(key)) for key in keys),
    )


def anthropic_model_capabilities(model: str, custom_llm_provider: str | None) -> dict[str, object]:
    """The capability probes the Rust messages route reads, once per model-map state: each probe
    walks the model map (``get_model_info`` plus generalization matching), and the route asks on
    every request."""
    resolved_model, provider = _resolved_provider(model, custom_llm_provider)
    cached: Final = _capabilities(
        _fingerprint(model, custom_llm_provider, resolved_model, provider), resolved_model, provider
    )
    tiers: Final = as_str_mapping(cached["effort_tiers"])
    return {**cached, "effort_tiers": dict(tiers) if tiers is not None else cached["effort_tiers"]}


@lru_cache(maxsize=256)
def _capabilities(fingerprint: _Fingerprint, resolved_model: str, provider: str) -> Mapping[str, object]:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    model: Final = fingerprint[0]

    def supports(flag: str) -> bool:
        return AnthropicModelInfo._supports_model_capability(model, flag, provider)  # pyright: ignore[reportPrivateUsage]  # same probes the Python transform runs; forking them would drift

    def tier(level: str) -> bool:
        return AnthropicConfig._supports_effort_level(model, level, provider)  # pyright: ignore[reportPrivateUsage]  # same probe the Python transform runs

    return {
        "supports_reasoning": supports("supports_reasoning"),
        "supports_adaptive_thinking": supports("supports_adaptive_thinking"),
        "thinking_always_on": supports("thinking_always_on"),
        "thinking_on_by_default": AnthropicModelInfo.thinks_by_default(model, provider),
        "supports_legacy_thinking": supports("supports_legacy_thinking"),
        "supports_output_config": supports("supports_output_config"),
        "supports_sampling_params": AnthropicModelInfo._supports_sampling_params(resolved_model),  # pyright: ignore[reportPrivateUsage]  # same gate the handler applies
        "supports_speed": AnthropicConfig._model_supports_speed_param(resolved_model, provider),  # pyright: ignore[reportPrivateUsage]  # same gate the handler applies
        "supports_mid_conversation_system": supports("supports_mid_conversation_system"),
        "effort_tiers": {level: tier(level) for level in ("minimal", "low", "medium", "high", "xhigh", "max")},
    }


def model_cost_document(known: _CatalogVersion | None) -> tuple[_CatalogVersion, Mapping[str, object] | None]:
    """The live model map and its active fallback rules as one document, which the native catalog reads, or
    ``None`` when the map is still at version ``known``."""
    version: Final = model_cost_version()
    current: Final = (version.table, version.size, version.writes)
    if current == known:
        return current, None
    rows: Final = as_str_mapping(litellm.model_cost) or {}  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # the model map is an untyped dict of rows
    rules: Final = as_object_list(fallback_generalizations.get_fallback_generalization_rules()) or []  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # the rule registry keeps the raw untyped list
    document: Final = _DECODED_JSON.validate_python(
        json.loads(json.dumps({**rows, FALLBACK_GENERALIZATIONS_KEY: {"rules": list(rules)}}, default=str))
    )
    return current, as_str_mapping(document)
