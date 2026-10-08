"""Bind a public LiteLLM call to its legacy Python signature without running it."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import (  # noqa: TID251  # narrows caller-owned containers without copying them
    Final,
    TypeAlias,
    TypeVar,
    cast,
)

import litellm

_INFERENCE_CONTEXT: Final = frozenset(
    {
        "model",
        "messages",
        "input",
        "api_key",
        "api_base",
        "base_url",
        "custom_llm_provider",
        "extra_headers",
        "timeout",
        "request_timeout",
        "callbacks",
        "success_callback",
        "failure_callback",
        "metadata",
        "litellm_metadata",
        "litellm_call_id",
        "litellm_trace_id",
        "litellm_logging_obj",
        "litellm_credential_name",
        "proxy_server_request",
    }
)


Bind: TypeAlias = Callable[[tuple[object, ...], Mapping[str, object]], Mapping[str, object] | None]


def binder(python: Callable[..., object]) -> Bind:
    """Binds a public ``(*args, **kwargs)`` call onto ``python``'s named parameters, defaults
    applied, or ``None`` when the call does not fit that signature."""
    python_signature: Final = inspect.signature(python)

    def bind(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        try:
            bound: Final = python_signature.bind(*args, **kwargs)
        except TypeError:
            return None
        bound.apply_defaults()
        return bound.arguments

    return bind


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def optional_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast("Mapping[str, object]", value)  # cast-ok: the same caller-owned object is handed on unchanged


def optional_sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return None
    return cast("Sequence[object]", value)  # cast-ok: the same caller-owned object is handed on unchanged


def inference_decline_reason(parameters: tuple[str, ...], kwargs: Mapping[str, object]) -> str | None:
    if litellm.drop_params or litellm.modify_params:
        return "native inference does not implement the configured parameter rewrites"
    for name, value in kwargs.items():
        if value is None:
            continue
        if name in {"cache", "caching"}:
            continue
        if name not in parameters and name not in _INFERENCE_CONTEXT:
            return f"native inference does not implement {name}"
    return None


@dataclass(frozen=True, slots=True)
class NativeCall:
    args: tuple[object, ...]
    kwargs: Mapping[str, object]
    bound: Mapping[str, object]


def native_call(args: tuple[object, ...], kwargs: Mapping[str, object], fields: Mapping[str, object]) -> NativeCall:
    extra: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    named: Final = {name: value for name, value in fields.items() if name != "kwargs"}
    return NativeCall(args=args, kwargs=kwargs, bound=MappingProxyType({**named, **extra}))


NativeResultT: Final = TypeVar("NativeResultT")


def native_call_hook(
    hook: Callable[[NativeCall], NativeResultT],
    call: NativeCall,
    _args: tuple[object, ...],
    _kwargs: Mapping[str, object],
) -> NativeResultT:
    return hook(call)
