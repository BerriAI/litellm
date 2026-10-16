from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import accumulate, chain, repeat, takewhile
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias, cast  # noqa: TID251  # tool lists stay caller-owned

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.rust_bridge.catalog import Feature
from litellm.rust_bridge.public_call import NativeCall, optional_mapping, optional_sequence, optional_str

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputParam

    from litellm.responses.mcp.litellm_proxy_mcp_handler import ToolParam
    from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponsesAPIOptionalRequestParams

Detector: TypeAlias = Callable[[NativeCall], bool]
Detectors: TypeAlias = Mapping[Feature, Detector]


def detected(detectors: Detectors, request: NativeCall) -> frozenset[Feature]:
    return frozenset(feature for feature, detector in detectors.items() if detector(request))


def _present(request: NativeCall, name: str) -> bool:
    return request.resolved.get(name) is not None


def mock_response(request: NativeCall) -> bool:
    return _present(request, "mock_response")


def caller_client(request: NativeCall) -> bool:
    return _present(request, "client") or _present(request, "shared_session")


def extra_body(request: NativeCall) -> bool:
    return bool(request.resolved.get("extra_body"))


def provider_affinity(request: NativeCall) -> bool:
    return _present(request, "provider_affinity_header")


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
    from litellm.types.utils import DynamicPromptManagementParamLiteral

    return _present(request, "prompt_id") or any(
        request.resolved.get(param) for param in DynamicPromptManagementParamLiteral.list_all_params()
    )


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
    return bool(litellm.num_retries) or any(
        _present(request, name) for name in ("num_retries", "max_retries", "fallbacks", "retry_policy")
    )


def _openai_model(request: NativeCall) -> str:
    from litellm.exceptions import BadRequestError
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model: Final = str(request.resolved["model"])
    try:
        return get_llm_provider(model, optional_str(request.resolved.get("custom_llm_provider")))[0]
    except BadRequestError:
        return model


def _api_base(request: NativeCall) -> str | None:
    return optional_str(request.resolved.get("api_base")) or optional_str(request.resolved.get("base_url"))


def _model_info(request: NativeCall) -> Mapping[str, object] | None:
    return optional_mapping(request.resolved.get("model_info"))


def _tools(request: NativeCall) -> Sequence[object]:
    return optional_sequence(request.resolved.get("tools")) or ()


def managed_resource_ids(request: NativeCall) -> bool:
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
        update_responses_tools_with_model_file_ids,
    )

    model_id: Final = optional_str((_model_info(request) or {}).get("id"))
    mapping: Final = request.resolved.get("model_file_id_mapping")
    file_ids: Final = cast("dict[str, dict[str, str]] | None", mapping)  # cast-ok: handler passes it unvalidated
    input_value: Final = request.resolved.get("input")
    tools: Final = [dict(tool) for tool in map(optional_mapping, _tools(request)) if tool is not None]
    rewritten_tools: Final = update_responses_tools_with_model_file_ids(tools, model_id, file_ids) if tools else tools
    return update_responses_input_with_model_file_ids(input_value, model_id, file_ids) != input_value or (
        rewritten_tools != tools
    )


def input_rewrite(request: NativeCall) -> bool:
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    input_value: Final = request.resolved.get("input")
    if not isinstance(input_value, list):
        return False
    config: Final = OpenAIResponsesAPIConfig()
    replayable: Final = config._drop_bridge_minted_reasoning_items(  # pyright: ignore[reportPrivateUsage]  # use the handler's replay policy
        cast("ResponseInputParam", input_value),  # cast-ok: the handler accepts unvalidated input items
    )
    normalized: Final = config._validate_input_param(replayable)  # pyright: ignore[reportPrivateUsage]  # use the handler's normalization
    return config._drop_foreign_tool_call_item_ids(normalized) != input_value  # pyright: ignore[reportPrivateUsage]  # use the handler's id cleanup


def _children(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(cast("Mapping[object, object]", value).values())  # cast-ok: only values are walked
    if isinstance(value, list | tuple):
        return tuple(cast("Sequence[object]", value))  # cast-ok: only items are walked
    return ()


def _next_level(level: tuple[object, ...], _: None) -> tuple[object, ...]:
    return tuple(chain.from_iterable(map(_children, level)))


def _mentions(value: object, key: str) -> bool:
    levels: Final = takewhile(bool, accumulate(repeat(None), _next_level, initial=(value,)))
    return any(
        key in cast("Mapping[object, object]", node)  # cast-ok: node passed the Mapping guard
        for node in chain.from_iterable(levels)
        if isinstance(node, Mapping)
    )


def cache_control_marks(request: NativeCall) -> bool:
    return _mentions(request.resolved.get("input"), "cache_control") or _mentions(_tools(request), "cache_control")


def anthropic_tool_search(request: NativeCall) -> bool:
    from litellm.responses.litellm_completion_transformation.tool_search import is_anthropic_tool_search_chat_tool

    return any(is_anthropic_tool_search_chat_tool(tool) for tool in _tools(request))


def _undecoded_schema(tool: object) -> bool:
    mapping: Final = optional_mapping(tool)
    parameters: Final = None if mapping is None else mapping.get("parameters")
    return parameters is not None and not isinstance(parameters, Mapping)


def tool_schema_repair(request: NativeCall) -> bool:
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.types.router import GenericLiteLLMParams

    tools: Final = _tools(request)
    if any(_undecoded_schema(tool) for tool in tools):
        return True
    sanitized: Final = OpenAIResponsesAPIConfig()._sanitized_tool_schemas_for_openai(  # pyright: ignore[reportPrivateUsage]  # the handler's own schema rewrite
        model=_openai_model(request),
        tools=cast("Sequence[ALL_RESPONSES_API_TOOL_PARAMS]", tools),  # cast-ok: sanitizer skips non-dict tools
        litellm_params=GenericLiteLLMParams(model_info=_model_info(request)),
    )
    return bool(tools) and sanitized is not tools


_PARAMS: Final = TypeAdapter(dict[str, object])


def model_param_gating(request: NativeCall) -> bool:
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.rust_bridge.host.routes.responses import PARAMETERS

    params: Final = {name: request.resolved[name] for name in PARAMETERS if request.resolved.get(name) is not None}
    mapped: Final = _PARAMS.validate_python(
        OpenAIResponsesAPIConfig().map_openai_params(  # pyright: ignore[reportUnknownMemberType]  # returns a bare dict
            cast("ResponsesAPIOptionalRequestParams", params),  # cast-ok: the handler passes these kwargs
            _openai_model(request),
            drop_params=True,
        )
    )
    return mapped != params


def openai_workload_identity(request: NativeCall) -> bool:
    from litellm.llms.openai.workload_identity import resolve_openai_workload_identity_config

    api_key: Final = optional_str(request.resolved.get("api_key")) or litellm.api_key or litellm.openai_key
    return resolve_openai_workload_identity_config(api_key, _api_base(request), request.resolved) is not None


def anthropic_workload_identity(request: NativeCall) -> bool:
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    from litellm.secret_managers.main import get_secret_str

    def configured(param: str, env: str) -> bool:
        return _present(request, param) or get_secret_str(env) is not None

    return (
        AnthropicModelInfo.get_api_key(optional_str(request.resolved.get("api_key"))) is None
        and AnthropicModelInfo.get_auth_token() is None
        and configured("anthropic_federation_rule_id", "ANTHROPIC_FEDERATION_RULE_ID")
        and configured("anthropic_organization_id", "ANTHROPIC_ORGANIZATION_ID")
    )


def cache_control_injection(request: NativeCall) -> bool:
    from litellm.integrations.anthropic_cache_control_hook import configured_injection_points

    return (
        litellm.enable_anthropic_prompt_caching is True
        or request.resolved.get("enable_prompt_caching") is True
        or bool(configured_injection_points(request.resolved.get("cache_control_injection_points")))
    )


_THINKING_BLOCKS: Final = frozenset({"thinking", "redacted_thinking"})


def _replays_thinking(message: object) -> bool:
    content: Final = optional_sequence((optional_mapping(message) or {}).get("content")) or ()
    return any((optional_mapping(block) or {}).get("type") in _THINKING_BLOCKS for block in content)


def thinking_signature_retry(request: NativeCall) -> bool:
    return any(_replays_thinking(message) for message in optional_sequence(request.resolved.get("messages")) or ())


def stream_cost_injection(request: NativeCall) -> bool:
    return streaming(request) and litellm.include_cost_in_streaming_usage is True


def drop_params(request: NativeCall) -> bool:
    return (
        bool(litellm.drop_params)
        or normalize_drop_params(request.resolved.get("drop_params")) is True
        or bool(optional_sequence(request.resolved.get("additional_drop_params")))
    )


def modify_params(_request: NativeCall) -> bool:
    return bool(litellm.modify_params)


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


_NATIVE_ECDH_CURVES: Final = frozenset({"x25519", "prime256v1", "secp256r1", "p-256", "secp384r1", "p-384"})


def tls_policy(_request: NativeCall) -> bool:
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
