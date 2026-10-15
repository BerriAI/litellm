from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias, cast  # noqa: TID251  # tool lists stay caller-owned

import httpx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.rust_bridge.catalog import Feature
from litellm.rust_bridge.public_call import NativeCall, optional_mapping, optional_sequence, optional_str

if TYPE_CHECKING:
    from litellm.responses.mcp.litellm_proxy_mcp_handler import ToolParam

Detector: TypeAlias = Callable[[NativeCall], bool]
Detectors: TypeAlias = Mapping[Feature, Detector]


def detected(detectors: Detectors, request: NativeCall) -> frozenset[Feature]:
    return frozenset(feature for feature, detector in detectors.items() if detector(request))


def _present(request: NativeCall, name: str) -> bool:
    return request.resolved.get(name) is not None


def mock_response(request: NativeCall) -> bool:
    return _present(request, "mock_response")


def caller_client(request: NativeCall) -> bool:
    return _present(request, "client")


def streaming(request: NativeCall) -> bool:
    return request.resolved.get("stream") is True


def timeout_variants(request: NativeCall) -> bool:
    return (
        _present(request, "request_timeout")
        or (streaming(request) and _present(request, "stream_timeout"))
        or isinstance(request.resolved.get("timeout"), httpx.Timeout)
    )


def proxy_mcp_tools(request: NativeCall) -> bool:
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    tools: Final = optional_sequence(request.resolved.get("tools"))
    return tools is not None and LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(
        cast("Iterable[ToolParam]", tools)  # cast-ok: the predicate skips entries that are not MCP tool dicts
    )


def prompt_management(request: NativeCall) -> bool:
    return _present(request, "prompt_id")


def text_format(request: NativeCall) -> bool:
    return _present(request, "text_format")


def chat_completions_bridge(request: NativeCall) -> bool:
    from litellm.responses.main import (
        _normalize_openai_chat_completions_responses_model,  # pyright: ignore[reportPrivateUsage]  # the handler's own bridge prefix check
    )

    return (
        bool(request.resolved.get("use_chat_completions_api"))
        or _normalize_openai_chat_completions_responses_model(str(request.resolved["model"]))[1]
    )


def retries_fallbacks(request: NativeCall) -> bool:
    return any(_present(request, name) for name in ("num_retries", "max_retries", "fallbacks"))


def drop_modify_params(request: NativeCall) -> bool:
    return (
        bool(litellm.drop_params)
        or bool(litellm.modify_params)
        or normalize_drop_params(request.resolved.get("drop_params")) is True
        or bool(optional_sequence(request.resolved.get("additional_drop_params")))
    )


def deployment_response_id(request: NativeCall) -> bool:
    metadata: Final = optional_mapping(request.resolved.get("litellm_metadata")) or {}
    model_info: Final = optional_mapping(metadata.get("model_info")) or {}
    return model_info.get("id") is not None


def _encoded_item(item: Mapping[str, object]) -> bool:
    from litellm.responses.utils import ResponsesAPIRequestUtils

    item_id: Final = optional_str(item.get("id"))
    encrypted_content: Final = optional_str(item.get("encrypted_content"))
    return (item_id is not None and ResponsesAPIRequestUtils.decode_encrypted_item_id(item_id) is not None) or (
        encrypted_content is not None
        and ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(encrypted_content)[1] != encrypted_content
    )


def encoded_item_ids(request: NativeCall) -> bool:
    items: Final = optional_sequence(request.resolved.get("input")) or ()
    mappings: Final = (optional_mapping(item) for item in items)
    return any(_encoded_item(item) for item in mappings if item is not None)


def _custom_loggers(callbacks: Iterable[object]) -> tuple[CustomLogger, ...]:
    from litellm.litellm_core_utils.litellm_logging import get_custom_logger_compatible_class

    resolved: Final = (
        get_custom_logger_compatible_class(callback) if isinstance(callback, str) else callback
        for callback in callbacks
    )
    return tuple(callback for callback in resolved if isinstance(callback, CustomLogger))


def _overridden(method: str, callbacks: Iterable[object]) -> bool:
    base: Final = getattr(CustomLogger, method)
    return any(getattr(type(callback), method, base) is not base for callback in _custom_loggers(callbacks))


def _request_success_callbacks(request: NativeCall) -> tuple[object, ...]:
    from litellm.litellm_core_utils.litellm_logging import Logging

    supplied: Final = optional_sequence(request.resolved.get("success_callback")) or ()
    logging_obj: Final = request.resolved.get("litellm_logging_obj")
    dynamic: Final = (
        optional_sequence(logging_obj.dynamic_success_callbacks)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Logging annotates the list with a bare Callable
        if isinstance(logging_obj, Logging)
        else None
    )
    return (*supplied, *(dynamic or ()))


def pre_request_hook(_request: NativeCall) -> bool:
    return _overridden("async_pre_request_hook", litellm.callbacks)


def agentic_loop_hook(request: NativeCall) -> bool:
    return _overridden("async_should_run_agentic_loop", (*litellm.callbacks, *_request_success_callbacks(request)))


def websearch_interception(_request: NativeCall) -> bool:
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger

    return any(isinstance(callback, WebSearchInterceptionLogger) for callback in litellm.callbacks)


def response_cache(request: NativeCall) -> bool:
    cache: Final = optional_mapping(request.resolved.get("cache")) or {}
    return litellm.cache is not None and cache.get("no-cache") is not True


def ssl_security_level(_request: NativeCall) -> bool:
    return bool(os.getenv("SSL_SECURITY_LEVEL", litellm.ssl_security_level))


COMMON: Final[Detectors] = MappingProxyType(
    {
        Feature.MOCK_RESPONSE: mock_response,
        Feature.PROXY_MCP_TOOLS: proxy_mcp_tools,
        Feature.CALLER_CLIENT: caller_client,
        Feature.TIMEOUT_VARIANTS: timeout_variants,
        Feature.AGENTIC_LOOP_HOOK: agentic_loop_hook,
        Feature.SSL_SECURITY_LEVEL: ssl_security_level,
    }
)
