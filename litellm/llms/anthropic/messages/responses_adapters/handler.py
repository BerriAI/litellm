"""
Handler for the Anthropic v1/messages -> OpenAI Responses API path.

Used when the target model is an OpenAI model.
"""

from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm._internal_context import resent_on
from litellm._logging import verbose_logger
from litellm.integrations.anthropic_cache_control_hook import (
    AnthropicCacheControlHook,
    routed_model_info,
    supports_openai_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.asyncify import (
    run_async_function,  # pyright: ignore[reportUnknownVariableType]  # untyped helper
)
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.prompt_cache_breakpoint_utils import (
    cache_control_as_prompt_cache_breakpoint,
    cache_control_as_responses_prompt_cache_breakpoint,
    prompt_cache_breakpoint_with_readable_mode,
    with_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.reasoning_effort_strength import REASONING_EFFORT_STRENGTH_RANK
from litellm.litellm_core_utils.reasoning_effort_utils import reasoning_effort_from_thinking_silently
from litellm.litellm_core_utils.tool_fields import drop_policy, tools_without_mcp_servers
from litellm.llms.base_llm.base_utils import optional_str
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.instructions_length import with_long_instructions_in_input
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import auto_reasoning_summary
from litellm.types.integrations.custom_logger import CONVERTED_STREAM_KEYS
from litellm.types.llms.anthropic import (
    AllAnthropicMessageValues,
    AllAnthropicToolsValues,
    AnthropicMcpServerTool,
    AnthropicMessagesRequest,
    AnthropicOutputConfig,
    AnthropicOutputSchema,
    ContextManagementResponse,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat
from litellm.utils import ProviderConfigManager

from ..adapters.handler import (
    context_management_explicitly_dropped,
    messages_without_rejected_reasoning,
    model_info_with_request_base_model,
    request_safeguards_evaluator,
    resent_without_rejected_reasoning,
)
from ..context_management import AnthropicContextManagementError, PolyfillResult, apply_context_management
from ..context_management.constants import COMPACT_EDIT_TYPE
from ..context_management.dispatcher import normalize_spec
from ..convertible_blocks import reject_malformed_text
from ..deferred_tools import native_tool_search_name, usable_deferred_tools_mode
from ..safeguards import with_safeguard_results
from ..utils import (
    KNOWN_REASONING_EFFORTS,
    REASONING_EFFORT_OFF,
    ClientEffort,
    ThinkingOffSwitch,
    litellm_logging_obj_from_kwargs,
    local_model_name,
    maps_bare_output_effort,
    model_excludes_effort,
    normalize_reasoning_effort_value,
    reasoning_off_level,
    thinking_off_switch,
)
from .streaming_iterator import AnthropicResponsesStreamWrapper
from .transformation import LiteLLMAnthropicToResponsesAPIAdapter

AnthropicRequestMessages: TypeAlias = list[AllAnthropicMessageValues] | list[dict[str, object]]
BlockRewrite: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]
_BLOCK_LIST: Final = TypeAdapter(list[dict[str, object]])
_SYSTEM_ITEMS: Final = TypeAdapter(list[object])
_SPEC: Final = TypeAdapter(dict[str, object])
_SplitEdits: TypeAlias = tuple[dict[str, object] | None, Sequence[Mapping[str, object]]]
AnthropicContextManagementSpec: TypeAlias = dict[str, object] | list[dict[str, object]] | None

_ADAPTER: Final = LiteLLMAnthropicToResponsesAPIAdapter()
_UNVERIFIED_ORGANIZATION_SUMMARY_REFUSAL: Final = "must be verified to generate reasoning summaries"
_WEB_SEARCH_FILTERS_REFUSAL: Final = "Parameter 'filters' not supported with model"


def _split_context_management(
    spec: AnthropicContextManagementSpec,
    kwargs: Mapping[str, object],
) -> _SplitEdits:
    """Return (the edits the Responses API runs natively, the edits the gateway polyfill runs).

    The Responses API compacts natively (``compact_20260112`` becomes its ``compaction``); any other
    edit, such as ``clear_tool_uses_20250919``, has no Responses API form and runs in the polyfill, as
    it does on the chat completions path. An OpenAI-style list spec is not Anthropic's shape and stays
    unsent, as it was before the polyfill ran here.
    """
    if not isinstance(spec, dict):
        return None, ()
    if not spec or context_management_explicitly_dropped(kwargs.get("additional_drop_params")):
        return spec, ()
    edits: Final = normalize_spec(spec) or []
    native: Final = [edit for edit in edits if edit.get("type") == COMPACT_EDIT_TYPE]
    return ({"edits": native} if native else None), tuple(
        edit for edit in edits if edit.get("type") != COMPACT_EDIT_TYPE
    )


def _system_blocks(system: object) -> str | Sequence[Mapping[str, object]] | None:
    if system is None or isinstance(system, str):
        return system
    try:
        items: Final = _SYSTEM_ITEMS.validate_python(system)
    except ValidationError:
        return None
    return tuple(block for block in map(as_str_mapping, items) if block is not None)


async def _polyfilled(
    *,
    model: str,
    messages: AnthropicRequestMessages,
    tools: Sequence[AllAnthropicToolsValues | Mapping[str, object]] | None,
    system: object,
    edits: Sequence[Mapping[str, object]],
    kwargs: Mapping[str, object],
) -> PolyfillResult | None:
    """Run the edits the Responses API can't run natively; raise like the chat completions path does."""
    if not edits:
        return None
    reject_malformed_text(messages, model, kwargs)
    drop_invalid_edits: Final = drops_unsupported_params(kwargs.get("drop_params"))
    blocks: Final = _system_blocks(system)
    try:
        return await apply_context_management(
            model=model,
            messages=_BLOCK_LIST.validate_python(messages),
            tools=_BLOCK_LIST.validate_python(tools) if tools else None,
            system=blocks if blocks is None or isinstance(blocks, str) else _BLOCK_LIST.validate_python(blocks),
            context_management_spec={"edits": [dict(edit) for edit in edits]},
            drop_invalid_edits=drop_invalid_edits,
        )
    except AnthropicContextManagementError:
        raise
    except Exception as e:
        verbose_logger.exception("context_management polyfill failed on the Responses API path: %s", e)
        raise AnthropicContextManagementError(
            status_code=500, message=f"context_management polyfill failed: {e}"
        ) from e


async def count_view_messages(
    *,
    model: str,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    system: object,
    context_management: CountContextManagement | None,
) -> Sequence[Mapping[str, object]]:
    """The messages /v1/messages sends the Responses API once the polyfill has run the edits it can't run natively."""
    if context_management is None or not isinstance(context_management.spec, dict):
        return messages
    _, edits = _split_context_management(_SPEC.validate_python(context_management.spec), {})
    result: Final = await _polyfilled(
        model=model,
        messages=_BLOCK_LIST.validate_python(messages),
        tools=tools,
        system=system,
        edits=edits,
        kwargs={"drop_params": context_management.drop_invalid_edits},
    )
    return messages if result is None else result.messages


async def count_messages_as_sent(
    *,
    model: str,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    system: object,
    messages_format: TokenCountMessagesFormat,
    context_management: CountContextManagement | None,
) -> Sequence[Mapping[str, object]] | AnthropicContextManagementError:
    match messages_format:
        case "openai":
            return messages
        case "anthropic":
            try:
                return await count_view_messages(
                    model=model,
                    messages=messages,
                    tools=tools,
                    system=system,
                    context_management=context_management,
                )
            except AnthropicContextManagementError as error:
                return error
        case _:
            assert_never(messages_format)


def _with_applied_edits(
    response: AnthropicMessagesResponse, polyfill_result: PolyfillResult | None
) -> AnthropicMessagesResponse:
    applied_edits: Final = polyfill_result.applied_edits_for_response() if polyfill_result else None
    if not applied_edits:
        return response
    return {**response, "context_management": ContextManagementResponse(applied_edits=list(applied_edits))}


def _record_bridged_response(responses_kwargs: Mapping[str, object], result: ResponsesAPIResponse) -> None:
    logging_obj: Final = litellm_logging_obj_from_kwargs(responses_kwargs)
    if logging_obj is not None:
        logging_obj.record_bridged_responses_api_response(result)


def _sent_responses_fields(request_params: Mapping[str, object], fields: Sequence[str]) -> Mapping[str, object]:
    extra_body: Final = as_str_mapping(request_params.get("extra_body")) or {}
    dropped: Final = drop_policy(request_params).additional_drop_params
    top_level: Final = {key: request_params.get(key) for key in fields if key not in dropped}
    return {key: extra_body.get(key, top_level.get(key)) for key in fields}


def preserves_system_instructions(request_params: Mapping[str, object]) -> bool:
    sent: Final = _sent_responses_fields(request_params, ("conversation", "truncation"))
    return bool(sent["conversation"]) or sent["truncation"] == "auto"


_PREVIOUS_RESPONSE_ID_UNSUPPORTED: Final = (
    "`previous_response_id` is not supported on the Messages API. Continue a stored response on the "
    "Responses API (`/v1/responses`), or send the earlier turns in `messages`."
)


def previous_response_id_error(request_params: Mapping[str, object]) -> str | None:
    sent: Final = _sent_responses_fields(request_params, ("previous_response_id",))
    return _PREVIOUS_RESPONSE_ID_UNSUPPORTED if sent["previous_response_id"] is not None else None


def _forwarded_kwargs(extra_kwargs: Mapping[str, object] | None) -> Mapping[str, object]:
    """The litellm-specific kwargs forwarded verbatim onto the Responses API request."""
    return extra_kwargs or {}


def _provider_returns_encrypted_reasoning(model: str, custom_llm_provider: object) -> bool:
    provider: Final = (
        custom_llm_provider if isinstance(custom_llm_provider, str) else litellm.get_llm_provider(model=model)[1]
    )
    provider_model: Final = local_model_name(model, provider)
    responses_config: Final = ProviderConfigManager.get_provider_responses_api_config(provider, provider_model)
    return responses_config is not None and "include" in responses_config.get_supported_openai_params(provider_model)


def _reads_prompt_cache_breakpoint(model: str, forwarded_kwargs: Mapping[str, object]) -> bool:
    provider: Final = forwarded_kwargs.get("custom_llm_provider")
    return AnthropicCacheControlHook.targets_openai_prompt_cache_breakpoint(
        model,
        provider if isinstance(provider, str) else None,
        forwarded_kwargs.get("api_base") or forwarded_kwargs.get("base_url"),
        forwarded_kwargs.get("prompt_cache_options"),
        routed_model_info(forwarded_kwargs.get("model_info")),
    )


def _model_takes_prompt_cache_breakpoint(model: str, forwarded_kwargs: Mapping[str, object]) -> bool:
    provider: Final = forwarded_kwargs.get("custom_llm_provider")
    provider_name: Final = provider if isinstance(provider, str) else None
    base_model: Final = forwarded_kwargs.get("base_model")
    model_info: Final = routed_model_info(forwarded_kwargs.get("model_info"))
    return supports_openai_prompt_cache_breakpoint(model, provider_name, model_info) or (
        isinstance(base_model, str)
        and bool(base_model)
        and supports_openai_prompt_cache_breakpoint(base_model, provider_name, model_info)
    )


def _without_prompt_cache_breakpoint(block: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in block.items() if key != "prompt_cache_breakpoint"}


def _cache_control_as_wire_safe_prompt_cache_breakpoint(
    block: Mapping[str, object], rename: BlockRewrite
) -> Mapping[str, object]:
    if "prompt_cache_breakpoint" not in block:
        return rename(block)
    wire_marker: Final = prompt_cache_breakpoint_with_readable_mode(block["prompt_cache_breakpoint"], drop_params=True)
    return rename(with_prompt_cache_breakpoint(_without_prompt_cache_breakpoint(block), wire_marker))


def _cache_control_rename(forwarded_kwargs: Mapping[str, object]) -> BlockRewrite:
    if forwarded_kwargs.get("custom_llm_provider") == "openai":
        return cache_control_as_responses_prompt_cache_breakpoint
    return cache_control_as_prompt_cache_breakpoint


def _prompt_cache_block_rewrite(model: str, forwarded_kwargs: Mapping[str, object]) -> BlockRewrite | None:
    """Client cache marks become breakpoints for a deployment that reads them, and a model that takes no breakpoints
    drops raw ones under drop_params."""
    drops_params: Final = drops_unsupported_params(forwarded_kwargs.get("drop_params"))
    if _reads_prompt_cache_breakpoint(model, forwarded_kwargs):
        rename: Final = _cache_control_rename(forwarded_kwargs)
        return partial(_cache_control_as_wire_safe_prompt_cache_breakpoint, rename=rename) if drops_params else rename
    if _model_takes_prompt_cache_breakpoint(model, forwarded_kwargs):
        return None
    if drops_params:
        return _without_prompt_cache_breakpoint
    return None


def _rewritten_block(block: Mapping[str, object], rewrite: BlockRewrite) -> Mapping[str, object]:
    rewritten: Final = rewrite(block)
    if rewritten.get("type") != "tool_result":
        return rewritten
    inner_blocks: Final = _rewritten_blocks(rewritten.get("content"), rewrite)
    return {**rewritten, "content": list(inner_blocks)} if inner_blocks else rewritten


def _rewritten_blocks(content: object, rewrite: BlockRewrite) -> Sequence[Mapping[str, object]]:
    """The content's blocks each rewritten, a tool_result's inner blocks included, empty when it is not a list of
    blocks."""
    try:
        return [_rewritten_block(block, rewrite) for block in _BLOCK_LIST.validate_python(content)]
    except ValidationError:
        return ()


def _content_blocks(content: object) -> Sequence[Mapping[str, object]]:
    if isinstance(content, str):
        return ({"type": "text", "text": content},)
    try:
        return _BLOCK_LIST.validate_python(content)
    except ValidationError:
        return ()


def _with_automatic_cache_mark(
    messages: AnthropicRequestMessages, cache_control: object
) -> Sequence[Mapping[str, object]]:
    """Anthropic's top-level ``cache_control`` caches the conversation up to its last block, so it becomes that
    block's own mark and maps like any client mark. An assistant prefill's blocks carry no breakpoint on the
    Responses API, so the mark goes on the last block of the turn before it."""
    marked: Final = next(
        (index for index in reversed(range(len(messages))) if messages[index].get("role") != "assistant"), None
    )
    if marked is None or cache_control is None:
        return messages
    blocks: Final = _content_blocks(messages[marked].get("content"))
    if not blocks or "cache_control" in blocks[-1]:
        return messages
    marked_content: Final = (*blocks[:-1], {**blocks[-1], "cache_control": cache_control})
    return (*messages[:marked], {**messages[marked], "content": marked_content}, *messages[marked + 1 :])


def _reasoning_for_target(
    reasoning: object,
    *,
    model: str,
    custom_llm_provider: str | None,
    deployment_model_info: object,
    drop_params: bool,
    off_switch: ThinkingOffSwitch | None,
) -> object:
    reasoning_params: Final = as_str_mapping(reasoning)
    effort: Final = reasoning_params.get("effort") if reasoning_params is not None else None
    if reasoning_params is None or not isinstance(effort, str):
        return reasoning
    if effort == REASONING_EFFORT_OFF and off_switch is not None:
        off_level: Final = reasoning_off_level(
            model,
            custom_llm_provider,
            deployment_model_info=deployment_model_info,
            drop_params=drop_params,
            between_tools=off_switch == "between_tools",
        )
        if off_level is ClientEffort.DROPPED:
            verbose_logger.warning(
                "Dropping reasoning.effort=none for %s, its model cost map entry does not set "
                "supports_none_reasoning_effort, because drop_params is set",
                model,
            )
            return off_level
        if off_level is None or _responses_api_refuses_reasoning_effort(model, custom_llm_provider):
            return None
        return {**reasoning_params, "effort": off_level}
    if drop_params and effort not in KNOWN_REASONING_EFFORTS:
        verbose_logger.warning(
            "Dropping reasoning.effort=%r for %s, no reasoning API defines it, because drop_params is set",
            effort,
            model,
        )
        return ClientEffort.DROPPED
    normalized: Final = normalize_reasoning_effort_value(
        effort,
        model=model,
        custom_llm_provider=custom_llm_provider,
        deployment_model_info=deployment_model_info,
    )
    if drop_params and model_excludes_effort(
        normalized, model=model, custom_llm_provider=custom_llm_provider, deployment_model_info=deployment_model_info
    ):
        verbose_logger.warning(
            "Dropping reasoning.effort=%r for %s, its model cost map entry does not list it, because drop_params is set",
            normalized,
            model,
        )
        return ClientEffort.DROPPED
    return reasoning if normalized == effort else {**reasoning_params, "effort": normalized}


def _effort_rank(reasoning: object) -> int | None:
    reasoning_params: Final = as_str_mapping(reasoning)
    effort: Final = reasoning_params.get("effort") if reasoning_params is not None else None
    return REASONING_EFFORT_STRENGTH_RANK.get(effort) if isinstance(effort, str) else None


def _falls_back_to_budget(requested_reasoning: object, thinking: object, *, drop_params: bool) -> bool:
    thinking_params: Final = as_str_mapping(thinking)
    if requested_reasoning is None or thinking_params is None or thinking_params.get("type") != "enabled":
        return False
    if requested_reasoning is ClientEffort.DROPPED:
        return True
    budget_effort: Final = reasoning_effort_from_thinking_silently(thinking, drop_params=drop_params)
    requested_rank: Final = _effort_rank(requested_reasoning)
    budget_rank: Final = REASONING_EFFORT_STRENGTH_RANK.get(budget_effort) if budget_effort else None
    return requested_rank is not None and budget_rank is not None and requested_rank > budget_rank


def _responses_api_refuses_reasoning_effort(model: str, custom_llm_provider: str | None) -> bool:
    return custom_llm_provider == LlmProviders.OPENAI.value and not OpenAIResponsesAPIConfig.supports_reasoning_param(
        model
    )


@dataclass(frozen=True, slots=True)
class _RefusedField:
    refusal: str
    reason: str
    sent: Callable[[Mapping[str, object]], bool]
    without: Callable[[Mapping[str, object]], Mapping[str, object]]

    def refused(self, error: Exception) -> bool:
        return isinstance(error, litellm.BadRequestError) and self.refusal in error.message


def _sends_summary(responses_kwargs: Mapping[str, object]) -> bool:
    reasoning: Final = as_str_mapping(responses_kwargs.get("reasoning"))
    return reasoning is not None and "summary" in reasoning


def _without_summary(responses_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    reasoning: Final = as_str_mapping(responses_kwargs.get("reasoning")) or {}
    return {**responses_kwargs, "reasoning": {key: value for key, value in reasoning.items() if key != "summary"}}


def _filtered_web_search(tool: object) -> Mapping[str, object] | None:
    mapping: Final = as_str_mapping(tool)
    return mapping if mapping is not None and mapping.get("type") == "web_search" and "filters" in mapping else None


def _unfiltered(tool: object) -> object:
    search: Final = _filtered_web_search(tool)
    return tool if search is None else {key: value for key, value in search.items() if key != "filters"}


def _sends_web_search_filters(responses_kwargs: Mapping[str, object]) -> bool:
    return any(_filtered_web_search(tool) is not None for tool in as_object_list(responses_kwargs.get("tools")) or ())


def _without_web_search_filters(responses_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    tools: Final = as_object_list(responses_kwargs.get("tools")) or ()
    return {**responses_kwargs, "tools": [_unfiltered(tool) for tool in tools]}


_REFUSED_FIELDS: Final = (
    _RefusedField(
        refusal=_UNVERIFIED_ORGANIZATION_SUMMARY_REFUSAL,
        reason="reasoning.summary, its OpenAI organization cannot get reasoning summaries until it is verified",
        sent=_sends_summary,
        without=_without_summary,
    ),
    _RefusedField(
        refusal=_WEB_SEARCH_FILTERS_REFUSAL,
        reason="web search domain filters, OpenAI does not take them on this model",
        sent=_sends_web_search_filters,
        without=_without_web_search_filters,
    ),
)


def _droppable_fields(responses_kwargs: Mapping[str, object], *, drop_params: bool) -> tuple[_RefusedField, ...]:
    return tuple(field for field in _REFUSED_FIELDS if drop_params and field.sent(responses_kwargs))


def _resent_without_refused_fields(
    responses_kwargs: Mapping[str, object], *, drop_params: bool
) -> AbstractContextManager[None]:
    droppable: Final = _droppable_fields(responses_kwargs, drop_params=drop_params)
    return resent_on(lambda error: any(field.refused(error) for field in droppable)) if droppable else nullcontext()


def _retry_without_refused_field(
    responses_kwargs: Mapping[str, object], error: litellm.BadRequestError, *, model: str, drop_params: bool
) -> Mapping[str, object] | None:
    droppable: Final = _droppable_fields(responses_kwargs, drop_params=drop_params)
    refused: Final = next((field for field in droppable if field.refused(error)), None)
    if refused is None:
        return None
    verbose_logger.warning("Retrying %s without %s, because drop_params is set", model, refused.reason)
    return refused.without(responses_kwargs)


def _maps_reasoning(
    thinking: Mapping[str, object] | None, model: str, provider: str | None, forwarded_kwargs: Mapping[str, object]
) -> bool:
    return bool(thinking) or (
        forwarded_kwargs.get("reasoning_effort") is None
        and maps_bare_output_effort(model, provider, deployment_model_info=forwarded_kwargs.get("model_info"))
    )


def _build_responses_kwargs(
    *,
    max_tokens: int,
    messages: AnthropicRequestMessages,
    model: str,
    context_management: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    output_config: AnthropicOutputConfig | None = None,
    stop_sequences: list[str] | None = None,
    stream: bool | None = False,
    system: str | None = None,
    temperature: float | None = None,
    thinking: dict[str, object] | None = None,
    tool_choice: dict[str, object] | None = None,
    tools: list[AllAnthropicToolsValues | dict[str, object]] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    output_format: AnthropicOutputSchema | None = None,
    speed: str | None = None,
    mcp_servers: Sequence[AnthropicMcpServerTool] | None = None,
    container: str | Mapping[str, object] | None = None,
    extra_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Build the kwargs dict to pass directly to litellm.responses() / litellm.aresponses().
    """
    from litellm.responses.bridge_check import drop_refused_forwarded_reasoning

    forwarded_kwargs: Final = drop_refused_forwarded_reasoning(_forwarded_kwargs(extra_kwargs), model=model)
    target_provider: Final = forwarded_kwargs.get("custom_llm_provider")
    unsupported_param_error: Final = previous_response_id_error(forwarded_kwargs)
    if unsupported_param_error is not None:
        raise litellm.UnsupportedParamsError(
            message=unsupported_param_error,
            model=model,
            llm_provider=target_provider if isinstance(target_provider, str) else None,
        )
    reject_malformed_text(messages, model, forwarded_kwargs)
    explicitly_dropped: Final = drop_policy(forwarded_kwargs).additional_drop_params
    explicitly_drops_mcp_servers: Final = "mcp_servers" in explicitly_dropped
    sent_mcp_servers: Final = None if explicitly_drops_mcp_servers else mcp_servers
    request_tools: Final = (
        tools_without_mcp_servers(
            mcp_servers,
            tools or (),
            model=model,
            route=target_provider if isinstance(target_provider, str) else "this route",
            drop_params=True,
        )
        if explicitly_drops_mcp_servers
        else tools or ()
    )
    block_rewrite: Final = _prompt_cache_block_rewrite(model, forwarded_kwargs)
    top_level_cache_control: Final = (
        None if "cache_control" in explicitly_dropped else forwarded_kwargs.get("cache_control")
    )
    request_messages: Final[AnthropicRequestMessages] = (
        [
            {
                **message,
                "content": list(_rewritten_blocks(message.get("content"), block_rewrite)) or message.get("content"),
            }
            for message in _with_automatic_cache_mark(messages, top_level_cache_control)
        ]
        if block_rewrite
        else messages
    )
    request_system: Final = (list(_rewritten_blocks(system, block_rewrite)) or system) if block_rewrite else system

    # Build a typed AnthropicMessagesRequest for the adapter
    request_data: Final[AnthropicMessagesRequest] = {
        "model": model,
        "messages": request_messages,
        "max_tokens": max_tokens,
    }
    if context_management:
        request_data["context_management"] = context_management
    if output_config:
        request_data["output_config"] = output_config
    if metadata:
        request_data["metadata"] = metadata
    if request_system:
        request_data["system"] = request_system
    if temperature is not None:
        request_data["temperature"] = temperature
    if thinking:
        request_data["thinking"] = thinking
    if tool_choice and (request_tools or not tools):
        request_data["tool_choice"] = tool_choice
    if request_tools:
        request_data["tools"] = (
            [dict(tool) for tool in _rewritten_blocks(request_tools, block_rewrite)] or list(request_tools)
            if block_rewrite
            else list(request_tools)
        )
    if top_p is not None:
        request_data["top_p"] = top_p
    if top_k is not None and "top_k" not in explicitly_dropped:
        request_data["top_k"] = top_k
    inference_geo: Final = forwarded_kwargs.get("inference_geo")
    if isinstance(inference_geo, str) and "inference_geo" not in explicitly_dropped:
        request_data["inference_geo"] = inference_geo
    if output_format:
        request_data["output_format"] = output_format
    if speed is not None and "speed" not in explicitly_dropped:
        request_data["speed"] = speed
    if sent_mcp_servers:
        request_data["mcp_servers"] = list(sent_mcp_servers)
    anthropic_request: Final[AnthropicMessagesRequest] = (
        {**request_data, "container": container if isinstance(container, str) else dict(container)}
        if container is not None and "container" not in explicitly_dropped
        else AnthropicMessagesRequest(**request_data)
    )
    drop_params: Final = drops_unsupported_params(forwarded_kwargs.get("drop_params"))
    auto_summary_level: Final = auto_reasoning_summary(model, forwarded_kwargs.get("custom_llm_provider"))
    responses_kwargs: Final = _ADAPTER.translate_request(
        anthropic_request,
        include_encrypted_reasoning=_provider_returns_encrypted_reasoning(
            model, forwarded_kwargs.get("custom_llm_provider")
        ),
        auto_summary_level=auto_summary_level,
        drop_params=drop_params,
        preserve_system_instructions=preserves_system_instructions(forwarded_kwargs),
        deferred_tools=usable_deferred_tools_mode(
            forwarded_kwargs.get("model_info"), optional_str(forwarded_kwargs.get("custom_llm_provider"))
        ),
    )

    caller_extra_body: Final = as_str_mapping(forwarded_kwargs.get("extra_body")) or {}
    caller_instructions: Final = caller_extra_body.get("instructions", forwarded_kwargs.get("instructions"))
    system_instructions: Final = responses_kwargs.get("instructions")
    if isinstance(system_instructions, str) and isinstance(caller_instructions, str):
        combined_instructions: Final = (
            f"{system_instructions}\n{caller_instructions}" if caller_instructions else system_instructions
        )
        responses_kwargs["instructions"] = combined_instructions
        if "instructions" in caller_extra_body:
            responses_kwargs["extra_body"] = {**caller_extra_body, "instructions": combined_instructions}

    provider_hint: Final = forwarded_kwargs.get("custom_llm_provider")
    provider: Final = provider_hint if isinstance(provider_hint, str) else None
    bridged_reasoning: Final[object] = responses_kwargs.pop("reasoning", None)
    for_target: Final = partial(
        _reasoning_for_target,
        model=model,
        custom_llm_provider=provider,
        deployment_model_info=model_info_with_request_base_model(forwarded_kwargs),
        drop_params=drop_params,
        off_switch=thinking_off_switch(anthropic_request),
    )
    requested_reasoning: Final = (
        for_target(bridged_reasoning) if _maps_reasoning(thinking, model, provider, forwarded_kwargs) else None
    )
    budget_reasoning: Final = (
        _ADAPTER.translate_thinking_to_reasoning(
            thinking, None, auto_summary_level, model=model, drop_params=drop_params
        )
        if _falls_back_to_budget(requested_reasoning, thinking, drop_params=drop_params)
        else None
    )
    target_reasoning: Final = (
        for_target(budget_reasoning)
        if budget_reasoning is not None and budget_reasoning != bridged_reasoning
        else requested_reasoning
    )
    client_effort_dropped: Final = target_reasoning is ClientEffort.DROPPED
    if target_reasoning is not None and not client_effort_dropped:
        responses_kwargs["reasoning"] = target_reasoning

    if stream:
        responses_kwargs["stream"] = True

    # Forward litellm-specific kwargs (api_key, api_base, logging obj, etc.)
    excluded: Final = (
        CONVERTED_STREAM_KEYS
        | {"anthropic_messages", "mcp_servers", "cache_control", "inference_geo"}
        | (frozenset({"reasoning_effort"}) if client_effort_dropped else frozenset())
    )
    for key, value in forwarded_kwargs.items():
        if key == "litellm_logging_obj" and value is not None:
            from litellm.litellm_core_utils.litellm_logging import (
                Logging as LiteLLMLoggingObject,
            )
            from litellm.types.utils import CallTypes

            if isinstance(value, LiteLLMLoggingObject):
                # Keep call_type as anthropic_messages so spend_logs are billed
                # against /v1/messages; the success handler translates the
                # Responses API result back to a ModelResponse for the row.
                setattr(value, "call_type", CallTypes.anthropic_messages.value)
            responses_kwargs[key] = value
        elif key not in excluded and key not in responses_kwargs and value is not None:
            responses_kwargs[key] = value

    explicit_prompt_cache_key: Final = forwarded_kwargs.get("prompt_cache_key")
    if explicit_prompt_cache_key is not None:
        responses_kwargs["prompt_cache_key"] = explicit_prompt_cache_key

    explicit_service_tier: Final = forwarded_kwargs.get("service_tier")
    if explicit_service_tier not in (None, "auto"):
        responses_kwargs["service_tier"] = explicit_service_tier

    deployment_include: Final = forwarded_kwargs.get("include")
    bridge_include: Final = responses_kwargs.get("include")
    if isinstance(deployment_include, list) and isinstance(bridge_include, list):
        responses_kwargs["include"] = [
            *bridge_include,
            *(item for item in deployment_include if item not in bridge_include),
        ]

    return dict(with_long_instructions_in_input(responses_kwargs))


_ResponsesKwargsFor: TypeAlias = Callable[[AnthropicRequestMessages], Mapping[str, object]]
_ResponsesResult: TypeAlias = ResponsesAPIResponse | BaseResponsesAPIStreamingIterator
_SyncResponsesResult: TypeAlias = _ResponsesResult | Coroutine[object, object, _ResponsesResult]


def _is_encrypted_reasoning_item(item: object) -> bool:
    mapping: Final = as_str_mapping(item)
    return mapping is not None and mapping.get("type") == "reasoning" and bool(mapping.get("encrypted_content"))


def _sends_encrypted_reasoning(responses_kwargs: Mapping[str, object]) -> bool:
    return any(map(_is_encrypted_reasoning_item, as_object_list(responses_kwargs.get("input")) or ()))


def _kwarg_text(responses_kwargs: Mapping[str, object], key: str) -> str:
    value: Final = responses_kwargs.get(key)
    return value if isinstance(value, str) else ""


async def _aresponses_dropping_refused_fields(
    responses_kwargs: Mapping[str, object], *, drop_params: bool
) -> _ResponsesResult:
    try:
        with _resent_without_refused_fields(responses_kwargs, drop_params=drop_params):
            return await litellm.aresponses(**responses_kwargs)
    except litellm.BadRequestError as error:
        retry_kwargs: Final = _retry_without_refused_field(
            responses_kwargs, error, model=_kwarg_text(responses_kwargs, "model"), drop_params=drop_params
        )
        if retry_kwargs is None:
            raise
        return await _aresponses_dropping_refused_fields(retry_kwargs, drop_params=drop_params)


def _responses_dropping_refused_fields(
    responses_kwargs: Mapping[str, object], *, drop_params: bool
) -> _SyncResponsesResult:
    try:
        with _resent_without_refused_fields(responses_kwargs, drop_params=drop_params):
            return litellm.responses(**responses_kwargs)
    except litellm.BadRequestError as error:
        retry_kwargs: Final = _retry_without_refused_field(
            responses_kwargs, error, model=_kwarg_text(responses_kwargs, "model"), drop_params=drop_params
        )
        if retry_kwargs is None:
            raise
        return _responses_dropping_refused_fields(retry_kwargs, drop_params=drop_params)


def _messages_without_rejected_reasoning(
    messages: AnthropicRequestMessages,
    error: litellm.BadRequestError,
    responses_kwargs: Mapping[str, object],
    sent_encrypted_reasoning: bool,
) -> AnthropicRequestMessages | None:
    return messages_without_rejected_reasoning(
        messages,
        error,
        sent_encrypted_reasoning=sent_encrypted_reasoning,
        provider=_kwarg_text(responses_kwargs, "custom_llm_provider"),
        model=_kwarg_text(responses_kwargs, "model"),
    )


async def _aresponses_without_rejected_reasoning(
    messages: AnthropicRequestMessages, responses_kwargs_for: _ResponsesKwargsFor, *, drop_params: bool
) -> tuple[Mapping[str, object], _ResponsesResult]:
    responses_kwargs: Final = responses_kwargs_for(messages)
    sent_encrypted_reasoning: Final = _sends_encrypted_reasoning(responses_kwargs)
    try:
        with resent_without_rejected_reasoning(sent_encrypted_reasoning):
            return responses_kwargs, await _aresponses_dropping_refused_fields(
                responses_kwargs, drop_params=drop_params
            )
    except litellm.BadRequestError as error:
        replayed_messages: Final = _messages_without_rejected_reasoning(
            messages, error, responses_kwargs, sent_encrypted_reasoning
        )
        if replayed_messages is None:
            raise
        replay_kwargs: Final = responses_kwargs_for(replayed_messages)
        return replay_kwargs, await _aresponses_dropping_refused_fields(replay_kwargs, drop_params=drop_params)


def _responses_without_rejected_reasoning(
    messages: AnthropicRequestMessages, responses_kwargs_for: _ResponsesKwargsFor, *, drop_params: bool
) -> tuple[Mapping[str, object], _SyncResponsesResult]:
    responses_kwargs: Final = responses_kwargs_for(messages)
    sent_encrypted_reasoning: Final = _sends_encrypted_reasoning(responses_kwargs)
    try:
        with resent_without_rejected_reasoning(sent_encrypted_reasoning):
            return responses_kwargs, _responses_dropping_refused_fields(responses_kwargs, drop_params=drop_params)
    except litellm.BadRequestError as error:
        replayed_messages: Final = _messages_without_rejected_reasoning(
            messages, error, responses_kwargs, sent_encrypted_reasoning
        )
        if replayed_messages is None:
            raise
        replay_kwargs: Final = responses_kwargs_for(replayed_messages)
        return replay_kwargs, _responses_dropping_refused_fields(replay_kwargs, drop_params=drop_params)


class LiteLLMMessagesToResponsesAPIHandler:
    """
    Handles Anthropic /v1/messages requests for OpenAI / Azure models by
    calling litellm.responses() / litellm.aresponses() directly and translating
    the response back to Anthropic format.
    """

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: AnthropicRequestMessages,
        model: str,
        context_management: AnthropicContextManagementSpec = None,
        metadata: dict[str, object] | None = None,
        output_config: AnthropicOutputConfig | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool | None = False,
        system: str | None = None,
        temperature: float | None = None,
        thinking: dict[str, object] | None = None,
        tool_choice: dict[str, object] | None = None,
        tools: list[AllAnthropicToolsValues | dict[str, object]] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        output_format: AnthropicOutputSchema | None = None,
        speed: str | None = None,
        mcp_servers: Sequence[AnthropicMcpServerTool] | None = None,
        container: str | Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> AnthropicMessagesResponse | AsyncIterator[bytes]:
        native_context_management, polyfilled_edits = _split_context_management(context_management, kwargs)
        polyfill_result: Final = await _polyfilled(
            model=model, messages=messages, tools=tools, system=system, edits=polyfilled_edits, kwargs=kwargs
        )

        def responses_kwargs_for(sent_messages: AnthropicRequestMessages) -> Mapping[str, object]:
            return _build_responses_kwargs(
                max_tokens=max_tokens,
                messages=sent_messages,
                model=model,
                context_management=native_context_management,
                metadata=metadata,
                output_config=output_config,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                output_format=output_format,
                speed=speed,
                mcp_servers=mcp_servers,
                container=container,
                extra_kwargs=kwargs,
            )

        safeguards_evaluator: Final = request_safeguards_evaluator(kwargs, messages)
        tool_search_name: Final = native_tool_search_name(
            kwargs.get("model_info"), optional_str(kwargs.get("custom_llm_provider")), tools
        )
        responses_kwargs, result = await _aresponses_without_rejected_reasoning(
            polyfill_result.messages if polyfill_result is not None else messages,
            responses_kwargs_for,
            drop_params=drops_unsupported_params(kwargs.get("drop_params")),
        )

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result,
                model=local_model_name(model, kwargs.get("custom_llm_provider")),
                litellm_logging_obj=litellm_logging_obj_from_kwargs(responses_kwargs),
                safeguards_evaluator=safeguards_evaluator,
                applied_edits=polyfill_result.applied_edits_for_response() if polyfill_result else None,
                tool_search_name=tool_search_name,
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        served: Final = await with_safeguard_results(
            _with_applied_edits(_ADAPTER.translate_response(result, tool_search_name), polyfill_result),
            safeguards_evaluator,
        )
        _record_bridged_response(responses_kwargs, result)
        return served

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: AnthropicRequestMessages,
        model: str,
        context_management: AnthropicContextManagementSpec = None,
        metadata: dict[str, object] | None = None,
        output_config: AnthropicOutputConfig | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool | None = False,
        system: str | None = None,
        temperature: float | None = None,
        thinking: dict[str, object] | None = None,
        tool_choice: dict[str, object] | None = None,
        tools: list[AllAnthropicToolsValues | dict[str, object]] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        output_format: AnthropicOutputSchema | None = None,
        speed: str | None = None,
        mcp_servers: Sequence[AnthropicMcpServerTool] | None = None,
        container: str | Mapping[str, object] | None = None,
        _is_async: bool = False,
        **kwargs: object,
    ) -> (
        AnthropicMessagesResponse
        | Iterator[bytes]
        | Coroutine[None, None, AnthropicMessagesResponse | AsyncIterator[bytes]]
    ):
        if _is_async:
            return LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                context_management=context_management,
                metadata=metadata,
                output_config=output_config,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                output_format=output_format,
                speed=speed,
                mcp_servers=mcp_servers,
                container=container,
                **kwargs,
            )

        # Sync path
        native_context_management, polyfilled_edits = _split_context_management(context_management, kwargs)
        ran: Final[object] = (  # pyright: ignore[reportUnknownVariableType]  # run_async_function is untyped
            run_async_function(
                _polyfilled,
                model=model,
                messages=messages,
                tools=tools,
                system=system,
                edits=polyfilled_edits,
                kwargs=kwargs,
            )
            if polyfilled_edits
            else None
        )
        polyfill_result: Final = ran if isinstance(ran, PolyfillResult) else None

        def responses_kwargs_for(sent_messages: AnthropicRequestMessages) -> Mapping[str, object]:
            return _build_responses_kwargs(
                max_tokens=max_tokens,
                messages=sent_messages,
                model=model,
                context_management=native_context_management,
                metadata=metadata,
                output_config=output_config,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                output_format=output_format,
                speed=speed,
                mcp_servers=mcp_servers,
                container=container,
                extra_kwargs=kwargs,
            )

        tool_search_name: Final = native_tool_search_name(
            kwargs.get("model_info"), optional_str(kwargs.get("custom_llm_provider")), tools
        )
        responses_kwargs, result = _responses_without_rejected_reasoning(
            polyfill_result.messages if polyfill_result is not None else messages,
            responses_kwargs_for,
            drop_params=drops_unsupported_params(kwargs.get("drop_params")),
        )

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result,
                model=local_model_name(model, kwargs.get("custom_llm_provider")),
                litellm_logging_obj=litellm_logging_obj_from_kwargs(responses_kwargs),
                applied_edits=polyfill_result.applied_edits_for_response() if polyfill_result else None,
                tool_search_name=tool_search_name,
            )
            return wrapper.anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _with_applied_edits(_ADAPTER.translate_response(result, tool_search_name), polyfill_result)
