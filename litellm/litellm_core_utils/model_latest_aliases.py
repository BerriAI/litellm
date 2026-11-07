from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # litellm.model_cost is an untyped module global

import litellm
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry


def _cost_map() -> Mapping[str, object]:
    return cast("Mapping[str, object]", litellm.model_cost)  # cast-ok: per-lookup catalog validation is too slow


def _alias_entry(key: str) -> AliasedCostEntry | None:
    entry: Final = _cost_map().get(key)
    return entry if isinstance(entry, AliasedCostEntry) and entry.names[:1] not in ((), (key,)) else None


def _alias_provider(key: str, entry: AliasedCostEntry) -> str | None:
    """The provider an alias is requested through: its key prefix for a hosted alias such as
    ``bedrock/claude-haiku-latest`` (whose target row belongs to ``bedrock_converse``), else the target's provider"""
    prefix, separator, _ = key.partition("/")
    if separator and prefix in litellm.provider_list:
        return prefix
    provider: Final = entry.get("litellm_provider")
    return provider if isinstance(provider, str) else None


def _alias_target(key: str, provider: str) -> str | None:
    entry: Final = _alias_entry(key)
    if entry is None or _alias_provider(key, entry) != provider:
        return None
    return entry.names[0].removeprefix(f"{provider}/")


def get_model_latest_aliases() -> Mapping[str, Mapping[str, str]]:
    """``{provider: {alias: target}}`` for every alias row in the cost map, with provider prefixes stripped"""
    rows: Final = tuple(
        (provider, key, entry.names[0])
        for key in _cost_map()
        if (entry := _alias_entry(key)) is not None and (provider := _alias_provider(key, entry)) is not None
    )
    return MappingProxyType(
        {
            provider: MappingProxyType(
                {
                    alias.removeprefix(f"{provider}/"): target.removeprefix(f"{provider}/")
                    for row_provider, alias, target in rows
                    if row_provider == provider
                }
            )
            for provider in dict.fromkeys(provider for provider, _, _ in rows)
        }
    )


def resolve_model_latest_alias(model: str, provider: str) -> str:
    return _alias_target(f"{provider}/{model}", provider) or _alias_target(model, provider) or model


def expand_model_latest_alias(model: str) -> tuple[str, ...]:
    provider, separator, name = model.partition("/")
    if not separator:
        entry: Final = _alias_entry(model)
        return (model,) if entry is None else (model, entry.names[0])
    target: Final = resolve_model_latest_alias(name, provider)
    return (model, f"{provider}/{target}") if target != name else (model,)


def is_model_latest_alias(model: str, custom_llm_provider: str | None = None) -> bool:
    prefix, separator, name = model.partition("/")
    provider: Final = custom_llm_provider or (prefix if separator else None)
    if provider is None:
        return _alias_entry(model) is not None
    candidate: Final = name if separator and prefix == provider else model
    return resolve_model_latest_alias(candidate, provider) != candidate
