"""
Handler for the Anthropic v1/messages -> OpenAI Responses API path.

Used when the target model is an OpenAI or Azure model.
"""

from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
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
    prompt_cache_breakpoint_with_readable_mode,
    with_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.tool_fields import drop_policy, drops_mcp_servers, tools_without_mcp_servers
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import auto_reasoning_summary
from litellm.types.integrations.custom_logger import CONVERTED_STREAM_KEYS
from litellm.types.llms.anthropic import (
    AllAnthropicMessageValues,
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
    AnthropicOutputConfig,
    AnthropicOutputSchema,
    ContextManagementResponse,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import CountContextManagement, TokenCountMessagesFormat
from litellm.utils import ProviderConfigManager

from ..adapters.handler import (
    context_management_explicitly_dropped,
    messages_without_rejected_reasoning,
    request_safeguards_evaluator,
    resent_without_rejected_reasoning,
)
from ..context_management import AnthropicContextManagementError, PolyfillResult, apply_context_management
from ..context_management.constants import COMPACT_EDIT_TYPE
from ..context_management.dispatcher import normalize_spec
from ..convertible_blocks import reject_malformed_tool_result_text
from ..safeguards import with_safeguard_results
from ..utils import litellm_logging_obj_from_kwargs, local_model_name
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
    reject_malformed_tool_result_text(messages, model, kwargs)
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


def _forwarded_kwargs(extra_kwargs: Mapping[str, object] | None) -> Mapping[str, object]:
    """The litellm-specific kwargs forwarded verbatim onto the Responses API request."""
    return extra_kwargs or {}


def _requests_background(forwarded_kwargs: Mapping[str, object]) -> bool:
    extra_body: Final = as_str_mapping(forwarded_kwargs.get("extra_body")) or {}
    return any(
        value is not None and value is not False
        for value in (forwarded_kwargs.get("background"), extra_body.get("background"))
    )


def _without_background(extra_body: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in extra_body.items() if key != "background"}


def _reject_background_mode(model: str, forwarded_kwargs: Mapping[str, object]) -> None:
    if not _requests_background(forwarded_kwargs) or drop_policy(forwarded_kwargs).drops("background"):
        return
    provider: Final = forwarded_kwargs.get("custom_llm_provider")
    raise litellm.UnsupportedParamsError(
        model=model,
        llm_provider=provider if isinstance(provider, str) else None,
        message=(
            f"/v1/messages does not support parameters: ['background'], for model={model}. "
            "A background run answers with a response id the Anthropic Messages API has no way to retrieve. "
            "To drop it, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
        ),
    )


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


def _cache_control_as_wire_safe_prompt_cache_breakpoint(block: Mapping[str, object]) -> Mapping[str, object]:
    if "prompt_cache_breakpoint" not in block:
        return cache_control_as_prompt_cache_breakpoint(block)
    wire_marker: Final = prompt_cache_breakpoint_with_readable_mode(block["prompt_cache_breakpoint"], drop_params=True)
    return cache_control_as_prompt_cache_breakpoint(
        with_prompt_cache_breakpoint(_without_prompt_cache_breakpoint(block), wire_marker)
    )


def _prompt_cache_block_rewrite(model: str, forwarded_kwargs: Mapping[str, object]) -> BlockRewrite | None:
    """Client cache marks become breakpoints for a deployment that reads them, and a model that takes no breakpoints
    drops raw ones under drop_params."""
    drops_params: Final = drops_unsupported_params(forwarded_kwargs.get("drop_params"))
    if _reads_prompt_cache_breakpoint(model, forwarded_kwargs):
        return (
            _cache_control_as_wire_safe_prompt_cache_breakpoint
            if drops_params
            else cache_control_as_prompt_cache_breakpoint
        )
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
    extra_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Build the kwargs dict to pass directly to litellm.responses() / litellm.aresponses().
    """
    forwarded_kwargs: Final = _forwarded_kwargs(extra_kwargs)
    reject_malformed_tool_result_text(messages, model, forwarded_kwargs)
    _reject_background_mode(model, forwarded_kwargs)
    target_provider: Final = forwarded_kwargs.get("custom_llm_provider")
    request_tools: Final = tools_without_mcp_servers(
        forwarded_kwargs.get("mcp_servers"),
        tools or (),
        model=model,
        route=target_provider if isinstance(target_provider, str) else "this route",
        drop_params=drops_mcp_servers(forwarded_kwargs),
    )
    block_rewrite: Final = _prompt_cache_block_rewrite(model, forwarded_kwargs)
    request_messages: Final[AnthropicRequestMessages] = (
        [
            {
                **message,
                "content": list(_rewritten_blocks(message.get("content"), block_rewrite)) or message.get("content"),
            }
            for message in messages
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
        request_data["tools"] = list(request_tools)
    if top_p is not None:
        request_data["top_p"] = top_p
    if output_format:
        request_data["output_format"] = output_format

    anthropic_request: Final = AnthropicMessagesRequest(**request_data)
    responses_kwargs: Final = _ADAPTER.translate_request(
        anthropic_request,
        include_encrypted_reasoning=_provider_returns_encrypted_reasoning(
            model, forwarded_kwargs.get("custom_llm_provider")
        ),
        auto_summary_level=auto_reasoning_summary(model, forwarded_kwargs.get("custom_llm_provider")),
        drop_params=drops_unsupported_params(forwarded_kwargs.get("drop_params")),
    )

    # Normalize reasoning effort based on model capabilities
    # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
    reasoning: Final = responses_kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        effort: Final[object] = reasoning.get("effort")
        if isinstance(effort, str):
            from litellm.llms.anthropic.messages.utils import (
                normalize_reasoning_effort_value,
            )

            provider_hint: Final = forwarded_kwargs.get("custom_llm_provider")
            normalized: Final = normalize_reasoning_effort_value(
                effort,
                model=model,
                custom_llm_provider=provider_hint if isinstance(provider_hint, str) else None,
                deployment_model_info=forwarded_kwargs.get("model_info"),
            )
            if normalized != effort:
                responses_kwargs["reasoning"] = {**reasoning, "effort": normalized}

    if stream:
        responses_kwargs["stream"] = True

    # Forward litellm-specific kwargs (api_key, api_base, logging obj, etc.)
    excluded: Final = CONVERTED_STREAM_KEYS | {"anthropic_messages", "background", "mcp_servers"}
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
        elif key == "extra_body" and (extra_body := as_str_mapping(value)) is not None:
            responses_kwargs[key] = _without_background(extra_body)
        elif key not in excluded and key not in responses_kwargs and value is not None:
            responses_kwargs[key] = value

    explicit_prompt_cache_key: Final = forwarded_kwargs.get("prompt_cache_key")
    if explicit_prompt_cache_key is not None:
        responses_kwargs["prompt_cache_key"] = explicit_prompt_cache_key

    deployment_include: Final = forwarded_kwargs.get("include")
    bridge_include: Final = responses_kwargs.get("include")
    if isinstance(deployment_include, list) and isinstance(bridge_include, list):
        responses_kwargs["include"] = [
            *bridge_include,
            *(item for item in deployment_include if item not in bridge_include),
        ]

    return responses_kwargs


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
    messages: AnthropicRequestMessages, responses_kwargs_for: _ResponsesKwargsFor
) -> tuple[Mapping[str, object], _ResponsesResult]:
    responses_kwargs: Final = responses_kwargs_for(messages)
    sent_encrypted_reasoning: Final = _sends_encrypted_reasoning(responses_kwargs)
    try:
        with resent_without_rejected_reasoning(sent_encrypted_reasoning):
            return responses_kwargs, await litellm.aresponses(**responses_kwargs)
    except litellm.BadRequestError as error:
        replayed_messages: Final = _messages_without_rejected_reasoning(
            messages, error, responses_kwargs, sent_encrypted_reasoning
        )
        if replayed_messages is None:
            raise
        replay_kwargs: Final = responses_kwargs_for(replayed_messages)
        return replay_kwargs, await litellm.aresponses(**replay_kwargs)


def _responses_without_rejected_reasoning(
    messages: AnthropicRequestMessages, responses_kwargs_for: _ResponsesKwargsFor
) -> tuple[Mapping[str, object], _SyncResponsesResult]:
    responses_kwargs: Final = responses_kwargs_for(messages)
    sent_encrypted_reasoning: Final = _sends_encrypted_reasoning(responses_kwargs)
    try:
        with resent_without_rejected_reasoning(sent_encrypted_reasoning):
            return responses_kwargs, litellm.responses(**responses_kwargs)
    except litellm.BadRequestError as error:
        replayed_messages: Final = _messages_without_rejected_reasoning(
            messages, error, responses_kwargs, sent_encrypted_reasoning
        )
        if replayed_messages is None:
            raise
        replay_kwargs: Final = responses_kwargs_for(replayed_messages)
        return replay_kwargs, litellm.responses(**replay_kwargs)


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
                extra_kwargs=kwargs,
            )

        safeguards_evaluator: Final = request_safeguards_evaluator(kwargs, messages)
        responses_kwargs, result = await _aresponses_without_rejected_reasoning(
            polyfill_result.messages if polyfill_result is not None else messages, responses_kwargs_for
        )

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result,
                model=local_model_name(model, kwargs.get("custom_llm_provider")),
                litellm_logging_obj=litellm_logging_obj_from_kwargs(responses_kwargs),
                safeguards_evaluator=safeguards_evaluator,
                applied_edits=polyfill_result.applied_edits_for_response() if polyfill_result else None,
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        served: Final = await with_safeguard_results(
            _with_applied_edits(_ADAPTER.translate_response(result), polyfill_result), safeguards_evaluator
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
                extra_kwargs=kwargs,
            )

        responses_kwargs, result = _responses_without_rejected_reasoning(
            polyfill_result.messages if polyfill_result is not None else messages, responses_kwargs_for
        )

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result,
                model=local_model_name(model, kwargs.get("custom_llm_provider")),
                litellm_logging_obj=litellm_logging_obj_from_kwargs(responses_kwargs),
                applied_edits=polyfill_result.applied_edits_for_response() if polyfill_result else None,
            )
            return wrapper.anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _with_applied_edits(_ADAPTER.translate_response(result), polyfill_result)
