"""Bind a public LiteLLM call to its legacy Python signature without running it."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # narrows caller-owned containers without copying them

from litellm.types.utils import is_litellm_owned_kwarg

_CONTEXT_ARGUMENTS: Final = frozenset(
    {
        "model",
        "messages",
        "input",
        "document",
        "file",
        "audio",
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


def signature(legacy: Callable[..., object]) -> inspect.Signature:
    return inspect.signature(legacy)


def bind(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> Mapping[str, object] | None:
    try:
        bound: Final = legacy.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    return bound.arguments


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
    import litellm

    if litellm.cache is not None or litellm.drop_params or litellm.modify_params:
        return "native inference does not implement the configured cache or parameter rewrites"
    for name, value in kwargs.items():
        if value is None:
            continue
        if (
            name not in parameters
            and name not in _CONTEXT_ARGUMENTS
            and name != "extra_body"
            and is_litellm_owned_kwarg(name)
        ):
            return f"native inference does not implement {name}"
    return None


def provider_parameters(
    parameters: Mapping[str, object],
    kwargs: Mapping[str, object],
    arguments: Mapping[str, object],
    names: tuple[str, ...],
) -> Mapping[str, object]:
    supplied: Final = {**parameters, **kwargs, **arguments}
    return {
        name: value
        for name, value in supplied.items()
        if (name in names and value is not None)
        or name == "extra_body"
        or (name not in names and name not in _CONTEXT_ARGUMENTS and not is_litellm_owned_kwarg(name))
    }
