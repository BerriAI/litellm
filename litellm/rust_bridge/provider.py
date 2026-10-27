from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.rust_bridge.public_call import NativeCall, optional_str
from litellm.types.router import GenericLiteLLMParams

__all__ = ("Resolution", "Resolved", "Unresolved", "resolve")

_HANDLER_FIELDS: Final = frozenset({"api_base", "api_key", "custom_llm_provider"})


@dataclass(frozen=True, slots=True)
class Resolved:
    model: str
    provider: str


@dataclass(frozen=True, slots=True)
class Unresolved:
    pass


Resolution: TypeAlias = Resolved | Unresolved


def resolve(request: NativeCall) -> Resolution:
    resolved: Final = request.resolved
    model: Final = str(resolved["model"])
    custom_llm_provider: Final = optional_str(resolved.get("custom_llm_provider"))
    api_base: Final = optional_str(resolved.get("api_base"))
    api_key: Final = optional_str(resolved.get("api_key"))
    extra: Final = MappingProxyType(
        {
            name: value
            for name, value in request.kwargs.items()
            if name not in request.base and name not in _HANDLER_FIELDS
        }
    )
    litellm_params: Final = GenericLiteLLMParams(
        **extra,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
    )
    try:
        resolved_model, resolved_provider, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
            litellm_params=litellm_params,
        )
    except BadRequestError:
        return Unresolved()
    return Resolved(resolved_model, resolved_provider)
