"""Shared feature detectors and helpers for native-route selection."""

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


def present(request: NativeCall, name: str) -> bool:
    return request.resolved.get(name) is not None


def mock_response(request: NativeCall) -> bool:
    """Python returns a supplied mock instead of calling the provider."""
    return present(request, "mock_response")


def caller_client(request: NativeCall) -> bool:
    """Python sends through the caller's HTTP client or shared session."""
    return present(request, "client") or present(request, "shared_session")


def extra_body(request: NativeCall) -> bool:
    """Python transforms and merges additional fields into the provider body."""
    return bool(request.resolved.get("extra_body"))


def streaming(request: NativeCall) -> bool:
    """The caller requested streaming, which some native routes do not support yet."""
    return request.resolved.get("stream") is True


def timeout_variants(request: NativeCall) -> bool:
    """Python handles request and stream overrides and httpx phase timeouts."""
    return (
        present(request, "request_timeout")
        or (streaming(request) and present(request, "stream_timeout"))
        or isinstance(request.resolved.get("timeout"), httpx.Timeout)
    )


def proxy_mcp_tools(request: NativeCall) -> bool:
    """Python expands proxy MCP tools and may execute their tool calls."""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    tools: Final = optional_sequence(request.resolved.get("tools"))
    return tools is not None and LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(
        cast("Iterable[ToolParam]", tools)  # cast-ok: the predicate skips entries that are not MCP tool dicts
    )


def prompt_management(request: NativeCall) -> bool:
    """Python resolves prompt-manager inputs or strips the litellm_agent prefix."""
    from litellm.integrations.litellm_agent.litellm_agent_model_resolver import LITELLM_AGENT_PREFIX
    from litellm.types.utils import DynamicPromptManagementParamLiteral

    return (
        present(request, "prompt_id")
        or str(request.resolved["model"]).startswith(LITELLM_AGENT_PREFIX)
        or any(request.resolved.get(param) for param in DynamicPromptManagementParamLiteral.list_all_params())
    )


def retries_fallbacks(request: NativeCall) -> bool:
    """Python runs the requested retry policy or fallback chain."""
    return bool(litellm.num_retries) or any(
        present(request, name) for name in ("num_retries", "max_retries", "fallbacks", "retry_policy")
    )


def api_base(request: NativeCall) -> str | None:
    return optional_str(request.resolved.get("api_base")) or optional_str(request.resolved.get("base_url"))


def cache_control_injection(request: NativeCall) -> bool:
    """Python injects Anthropic prompt-cache breakpoints into messages and system content."""
    from litellm.integrations.anthropic_cache_control_hook import configured_injection_points

    return (
        litellm.enable_anthropic_prompt_caching is True
        or request.resolved.get("enable_prompt_caching") is True
        or bool(configured_injection_points(request.resolved.get("cache_control_injection_points")))
    )


def drop_params(request: NativeCall) -> bool:
    """Python drops unsupported or explicitly excluded request parameters."""
    return (
        bool(litellm.drop_params)
        or normalize_drop_params(request.resolved.get("drop_params")) is True
        or bool(optional_sequence(request.resolved.get("additional_drop_params")))
    )


def modify_params(_request: NativeCall) -> bool:
    """Python rewrites unsupported request values when modify_params is enabled."""
    return bool(litellm.modify_params)


def _custom_loggers(callbacks: Iterable[object]) -> tuple[CustomLogger, ...]:
    from litellm.litellm_core_utils.litellm_logging import get_custom_logger_compatible_class

    resolved: Final = (
        get_custom_logger_compatible_class(callback) if isinstance(callback, str) else callback
        for callback in callbacks
    )
    return tuple(callback for callback in resolved if isinstance(callback, CustomLogger))


def overridden(method: str, callbacks: Iterable[object]) -> bool:
    base: Final[object] = getattr(CustomLogger, method)
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


def call_callbacks(request: NativeCall) -> tuple[object, ...]:
    return (*litellm.callbacks, *_request_success_callbacks(request))


def agentic_loop_hook(request: NativeCall) -> bool:
    """A callback can request follow-up tool calls through Python's agentic loop."""
    return overridden("async_should_run_agentic_loop", call_callbacks(request))


def response_cache(request: NativeCall) -> bool:
    """Python owns cache lookup and storage unless this call disables cache reads."""
    cache: Final = optional_mapping(request.resolved.get("cache")) or {}
    return litellm.cache is not None and cache.get("no-cache") is not True


_NATIVE_ECDH_CURVES: Final = frozenset({"x25519", "prime256v1", "secp256r1", "p-256", "secp384r1", "p-384"})


def tls_policy(_request: NativeCall) -> bool:
    """Python applies TLS settings or ECDH curves the native client cannot configure."""
    curve: Final = os.getenv("SSL_ECDH_CURVE", litellm.ssl_ecdh_curve)
    return bool(os.getenv("SSL_SECURITY_LEVEL", litellm.ssl_security_level)) or (
        bool(curve) and str(curve).strip().lower() not in _NATIVE_ECDH_CURVES
    )


COMMON: Final[Detectors] = MappingProxyType(
    {
        Feature.MOCK_RESPONSE: mock_response,
        Feature.PROXY_MCP_TOOLS: proxy_mcp_tools,
        Feature.CALLER_CLIENT: caller_client,
        Feature.TIMEOUT_VARIANTS: timeout_variants,
        Feature.AGENTIC_LOOP_HOOK: agentic_loop_hook,
        Feature.TLS_POLICY: tls_policy,
    }
)
