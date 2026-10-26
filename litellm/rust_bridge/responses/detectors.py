"""Detectors for Python-only Responses behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import accumulate, chain, repeat, takewhile
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # tool lists stay caller-owned

from pydantic import TypeAdapter

import litellm
from litellm.llms.opencode.common_utils import is_opencode_api_base
from litellm.rust_bridge.detectors import api_base, call_callbacks, overridden, present
from litellm.rust_bridge.public_call import NativeCall, optional_mapping, optional_sequence, optional_str

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputParam

    from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponsesAPIOptionalRequestParams


def client_headers(request: NativeCall) -> bool:
    """Python merges proxy-forwarded headers into extra_headers before sending."""
    return bool(optional_mapping(request.resolved.get("headers")))


def provider_affinity(request: NativeCall) -> bool:
    """Python validates and fills the affinity header from the session ID."""
    return present(request, "provider_affinity_header") or is_opencode_api_base(api_base(request))


def text_format(request: NativeCall) -> bool:
    """Python builds the output schema and parses the response into the supplied type."""
    return present(request, "text_format")


def chat_completions_bridge(request: NativeCall) -> bool:
    """Python serves this Responses request through Chat Completions."""
    from litellm.responses.main import (
        _normalize_openai_chat_completions_responses_model,  # pyright: ignore[reportPrivateUsage]  # the handler's own bridge prefix check
    )

    return (
        bool(request.resolved.get("use_chat_completions_api"))
        or _normalize_openai_chat_completions_responses_model(str(request.resolved["model"]))[1]
    )


def _openai_model(request: NativeCall) -> str:
    from litellm.exceptions import BadRequestError
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model: Final = str(request.resolved["model"])
    try:
        return get_llm_provider(model, optional_str(request.resolved.get("custom_llm_provider")))[0]
    except BadRequestError:
        return model


def _model_info(request: NativeCall) -> Mapping[str, object] | None:
    return optional_mapping(request.resolved.get("model_info"))


def _tools(request: NativeCall) -> Sequence[object]:
    return optional_sequence(request.resolved.get("tools")) or ()


def managed_resource_ids(request: NativeCall) -> bool:
    """Python replaces managed file IDs in Responses input and tools with provider IDs."""
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
    """Python normalizes replayed Responses items and removes unsupported IDs or namespaces."""
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    input_value: Final = request.resolved.get("input")
    if not isinstance(input_value, list):
        return False
    config: Final = OpenAIResponsesAPIConfig()
    replayable: Final = config._drop_bridge_minted_reasoning_items(  # pyright: ignore[reportPrivateUsage]  # use the handler's replay policy
        cast("ResponseInputParam", input_value),  # cast-ok: the handler accepts unvalidated input items
    )
    normalized: Final = config._validate_input_param(replayable)  # pyright: ignore[reportPrivateUsage]  # use the handler's normalization
    cleaned: Final = config._drop_foreign_tool_call_item_ids(normalized)  # pyright: ignore[reportPrivateUsage]  # use the handler's id cleanup
    return config.strip_custom_tool_call_namespace_from_responses_input(cleaned) != input_value


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
    """Python strips or translates cache_control marks in Responses input and tools."""
    return _mentions(request.resolved.get("input"), "cache_control") or _mentions(_tools(request), "cache_control")


def anthropic_tool_search(request: NativeCall) -> bool:
    """Python translates Anthropic tool-search definitions for the Responses API."""
    from litellm.responses.litellm_completion_transformation.tool_search import is_anthropic_tool_search_chat_tool

    return any(is_anthropic_tool_search_chat_tool(tool) for tool in _tools(request))


def _undecoded_schema(tool: object) -> bool:
    mapping: Final = optional_mapping(tool)
    parameters: Final = None if mapping is None else mapping.get("parameters")
    return parameters is not None and not isinstance(parameters, Mapping)


def tool_schema_repair(request: NativeCall) -> bool:
    """Python decodes or repairs tool schemas before sending them to OpenAI."""
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
    """Python clamps, drops or remaps Responses params according to the model's capabilities."""
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
    """Python mints an OpenAI federation token when no static API key is configured."""
    from litellm.llms.openai.workload_identity import resolve_openai_workload_identity_config

    api_key: Final = optional_str(request.resolved.get("api_key")) or litellm.api_key or litellm.openai_key
    return resolve_openai_workload_identity_config(api_key, api_base(request), request.resolved) is not None


def deployment_response_id(request: NativeCall) -> bool:
    """Python encodes the deployment ID into the returned Responses ID."""
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
    """Python restores provider IDs or encrypted content from previously wrapped items."""
    items: Final = optional_sequence(request.resolved.get("input")) or ()
    mappings: Final = (optional_mapping(item) for item in items)
    return any(_encoded_item(item) for item in mappings if item is not None)


def deployment_hook(request: NativeCall) -> bool:
    """The sync Responses handler runs deployment hooks that native currently skips."""
    return overridden("async_pre_call_deployment_hook", call_callbacks(request))
