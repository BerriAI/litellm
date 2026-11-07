from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

_ALIASES_ADAPTER: Final = TypeAdapter(dict[str, dict[str, str]])


@cache
def get_model_latest_aliases() -> Mapping[str, Mapping[str, str]]:
    aliases: Final = _ALIASES_ADAPTER.validate_json(files("litellm").joinpath("model_latest_aliases.json").read_bytes())
    return MappingProxyType({provider: MappingProxyType(models) for provider, models in aliases.items()})


def resolve_model_latest_alias(model: str, provider: str) -> str:
    aliases: Final = get_model_latest_aliases().get(provider)
    return aliases.get(model, model) if aliases is not None else model


def expand_model_latest_alias(model: str) -> tuple[str, ...]:
    provider, separator, name = model.partition("/")
    target: Final = resolve_model_latest_alias(name, provider) if separator else name
    return (model, f"{provider}/{target}") if separator and target != name else (model,)


def is_model_latest_alias(model: str, custom_llm_provider: str | None = None) -> bool:
    prefix, separator, name = model.partition("/")
    provider: Final = custom_llm_provider or prefix
    candidate: Final = name if separator and prefix == provider else model
    return bool(custom_llm_provider or separator) and resolve_model_latest_alias(candidate, provider) != candidate
