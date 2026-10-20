"""Detectors for Python-only Chat Completions behavior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.rust_bridge.detectors import (
    agentic_loop_hook,
    api_base,
    call_callbacks,
    mock_response,
    overridden,
    present,
    retries_fallbacks,
)
from litellm.rust_bridge.public_call import NativeCall, optional_sequence, optional_str

_PARAMS: Final = TypeAdapter(dict[str, object])


def chat_mock_response(request: NativeCall) -> bool:
    """Chat mocks can supply text, tool calls or a timeout without provider IO."""
    return (
        mock_response(request)
        or bool(request.resolved.get("mock_tool_calls"))
        or bool(request.resolved.get("mock_timeout"))
    )


def model_list(request: NativeCall) -> bool:
    """Python fans the request out across the supplied model deployments."""
    return present(request, "model_list")


def model_alias_map(request: NativeCall) -> bool:
    """Python rewrites this model through the configured alias map."""
    return bool(litellm.model_alias_map) and str(request.resolved["model"]) in litellm.model_alias_map


def chat_retries_fallbacks(request: NativeCall) -> bool:
    """Chat also reads global model fallbacks and context-window fallbacks."""
    return (
        retries_fallbacks(request)
        or bool(optional_sequence(litellm.model_fallbacks))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # litellm annotates it with a bare list
        or present(request, "context_window_fallback_dict")
    )


_CHAT_UNGATED: Final = frozenset(
    {
        "model",
        "messages",
        "custom_llm_provider",
        "api_version",
        "drop_params",
        "allowed_openai_params",
        "additional_drop_params",
    }
)


_NAMES: Final = TypeAdapter(list[str])


def _chat_non_default_params(request: NativeCall) -> Mapping[str, object]:
    from litellm.constants import DEFAULT_CHAT_COMPLETION_PARAM_VALUES

    defaults: Final = _PARAMS.validate_python(DEFAULT_CHAT_COMPLETION_PARAM_VALUES)
    return {
        name: value
        for name, value in request.resolved.items()
        if name in defaults and name not in _CHAT_UNGATED and value != defaults[name]
    }


def _chat_provider(request: NativeCall) -> tuple[str, str | None]:
    from litellm.exceptions import BadRequestError
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model: Final = str(request.resolved["model"])
    provider: Final = optional_str(request.resolved.get("custom_llm_provider"))
    try:
        resolved_model, resolved_provider, _, _ = get_llm_provider(model, provider, api_base=api_base(request))
    except BadRequestError:
        return model, provider
    return resolved_model, resolved_provider


def _unvalidated(params: Mapping[str, object]) -> frozenset[str]:
    from litellm.utils import PROVIDER_UNVALIDATED_PARAMS

    return PROVIDER_UNVALIDATED_PARAMS | (frozenset({"n"}) if params.get("n") == 1 else frozenset[str]())


def chat_model_param_gating(request: NativeCall) -> bool:
    """Python rejects unsupported chat params or maps them differently with drop_params."""
    from litellm.exceptions import UnsupportedParamsError
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,  # pyright: ignore[reportUnknownVariableType]  # returns a bare list
    )
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    params: Final = _chat_non_default_params(request)
    model, provider = _chat_provider(request)
    if not params or provider is None:
        return False
    supported: Final = frozenset(_NAMES.validate_python(get_supported_openai_params(model, provider) or []))
    if any(name not in supported for name in params.keys() - _unvalidated(params)):
        return True
    try:
        llm_provider: Final = LlmProviders(provider)
    except ValueError:
        return False
    config: Final = ProviderConfigManager.get_provider_chat_config(model, llm_provider)
    if config is None:
        return False
    try:
        strict: Final = _PARAMS.validate_python(
            config.map_openai_params(dict(params), {}, model, drop_params=False)  # pyright: ignore[reportUnknownMemberType]  # returns a bare dict
        )
    except UnsupportedParamsError:
        return True
    return strict != _PARAMS.validate_python(
        config.map_openai_params(dict(params), {}, model, drop_params=True)  # pyright: ignore[reportUnknownMemberType]  # returns a bare dict
    )


def chat_agentic_loop_hook(request: NativeCall) -> bool:
    """Chat callbacks can also override the chat-completion-specific agentic loop."""
    return agentic_loop_hook(request) or overridden(
        "async_should_run_chat_completion_agentic_loop", call_callbacks(request)
    )
