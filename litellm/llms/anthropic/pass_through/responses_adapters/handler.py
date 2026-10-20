"""
Handler for the Anthropic v1/messages -> OpenAI Responses API path.

Used when the target model is an OpenAI or Azure model.
"""

from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from typing import Final, NoReturn, TypeAlias

import httpx
from aiohttp import ClientSession
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.constants import request_timeout
from litellm.integrations.anthropic_cache_control_hook import (
    AnthropicCacheControlHook,
    supports_openai_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.common_utils import cache_control_as_prompt_cache_breakpoint
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.native.responses import execute_responses
from litellm.responses.main import (
    apply_managed_file_id_mapping,
    base_llm_http_handler,
    requested_reasoning,
    resolve_model_provider_for_responses,
    resolve_responses_api_provider_config,
    with_provider_affinity_header,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils, auto_reasoning_summary
from litellm.types.integrations.custom_logger import CONVERTED_STREAM_KEYS
from litellm.types.llms.anthropic import (
    AllAnthropicMessageValues,
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
    AnthropicOutputConfig,
    AnthropicOutputSchema,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CallTypes
from litellm.utils import ProviderConfigManager

from ..adapters.handler import request_safeguards_evaluator
from ..convertible_blocks import reject_malformed_tool_result_text
from ..safeguards import with_safeguard_results
from ..utils import litellm_logging_obj_from_kwargs, local_model_name
from .streaming_iterator import AnthropicResponsesStreamWrapper
from .transformation import LiteLLMAnthropicToResponsesAPIAdapter

AnthropicRequestMessages: TypeAlias = list[AllAnthropicMessageValues] | list[dict[str, object]]
BlockRewrite: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]
_BLOCK_LIST: Final = TypeAdapter(list[dict[str, object]])

_ADAPTER: Final = LiteLLMAnthropicToResponsesAPIAdapter()
_STRING_KEYED_DICT: Final = TypeAdapter(dict[str, object])
_CLIENT_HEADERS: Final = TypeAdapter(dict[str, str])
_STRINGS: Final = TypeAdapter(tuple[str, ...])


_NEVER_FORWARDED: Final = CONVERTED_STREAM_KEYS | {"anthropic_messages"}


def _record_bridged_response(responses_kwargs: Mapping[str, object], result: ResponsesAPIResponse) -> None:
    logging_obj: Final = litellm_logging_obj_from_kwargs(responses_kwargs)
    if logging_obj is not None:
        logging_obj.record_bridged_responses_api_response(result)


def _forwarded_kwargs(extra_kwargs: Mapping[str, object] | None) -> Mapping[str, object]:
    """The litellm-specific kwargs (credentials, routing, metadata) that carry over to the Responses API call."""
    return {
        key: value for key, value in (extra_kwargs or {}).items() if key not in _NEVER_FORWARDED and value is not None
    }


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
    )


def _model_takes_prompt_cache_breakpoint(model: str, forwarded_kwargs: Mapping[str, object]) -> bool:
    provider: Final = forwarded_kwargs.get("custom_llm_provider")
    provider_name: Final = provider if isinstance(provider, str) else None
    base_model: Final = forwarded_kwargs.get("base_model")
    return supports_openai_prompt_cache_breakpoint(model, provider_name) or (
        isinstance(base_model, str)
        and bool(base_model)
        and supports_openai_prompt_cache_breakpoint(base_model, provider_name)
    )


def _without_prompt_cache_breakpoint(block: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in block.items() if key != "prompt_cache_breakpoint"}


def _prompt_cache_block_rewrite(model: str, forwarded_kwargs: Mapping[str, object]) -> BlockRewrite | None:
    """Client cache marks become breakpoints for a deployment that reads them, and a model that takes no breakpoints
    drops raw ones under drop_params."""
    if _reads_prompt_cache_breakpoint(model, forwarded_kwargs):
        return cache_control_as_prompt_cache_breakpoint
    if _model_takes_prompt_cache_breakpoint(model, forwarded_kwargs):
        return None
    if drops_unsupported_params(forwarded_kwargs.get("drop_params")):
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


def _build_responses_request(
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
    """The Responses API request the Messages request translates to, without the forwarded litellm kwargs."""
    forwarded_kwargs: Final = _forwarded_kwargs(extra_kwargs)
    reject_malformed_tool_result_text(messages, model, forwarded_kwargs)
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
    if tool_choice:
        request_data["tool_choice"] = tool_choice
    if tools:
        request_data["tools"] = tools
    if top_p is not None:
        request_data["top_p"] = top_p
    if output_format:
        request_data["output_format"] = output_format

    anthropic_request: Final = AnthropicMessagesRequest(**request_data)
    responses_request: Final = _ADAPTER.translate_request(
        anthropic_request,
        include_encrypted_reasoning=_provider_returns_encrypted_reasoning(
            model, forwarded_kwargs.get("custom_llm_provider")
        ),
        auto_summary_level=auto_reasoning_summary(model, forwarded_kwargs.get("custom_llm_provider")),
    )

    # Normalize reasoning effort based on model capabilities
    # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
    reasoning: Final = responses_request.get("reasoning")
    if isinstance(reasoning, dict):
        effort: Final[object] = reasoning.get("effort")
        if isinstance(effort, str):
            from litellm.llms.anthropic.pass_through.utils import (
                normalize_reasoning_effort_value,
            )

            provider_hint: Final = forwarded_kwargs.get("custom_llm_provider")
            normalized: Final = normalize_reasoning_effort_value(
                effort,
                model=model,
                custom_llm_provider=provider_hint if isinstance(provider_hint, str) else None,
            )
            if normalized != effort:
                responses_request["reasoning"] = {**reasoning, "effort": normalized}

    if stream:
        responses_request["stream"] = True

    explicit_prompt_cache_key: Final = forwarded_kwargs.get("prompt_cache_key")
    if explicit_prompt_cache_key is not None:
        responses_request["prompt_cache_key"] = explicit_prompt_cache_key

    deployment_include: Final = forwarded_kwargs.get("include")
    bridge_include: Final = responses_request.get("include")
    if isinstance(deployment_include, list) and isinstance(bridge_include, list):
        responses_request["include"] = [
            *bridge_include,
            *(item for item in deployment_include if item not in bridge_include),
        ]

    return responses_request


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_keyed_dict(value: object) -> dict[str, object] | None:
    try:
        return _STRING_KEYED_DICT.validate_python(value) if value is not None else None
    except ValidationError:
        return None


def _client_headers(value: object) -> Mapping[str, str] | None:
    try:
        return _CLIENT_HEADERS.validate_python(value) if isinstance(value, dict) else None
    except ValidationError:
        return None


def _shared_session(value: object) -> ClientSession | None:
    return value if isinstance(value, ClientSession) else None


def _http_client(value: object) -> HTTPHandler | AsyncHTTPHandler | None:
    return value if isinstance(value, (HTTPHandler, AsyncHTTPHandler)) else None


def _timeout(value: object) -> float | httpx.Timeout:
    return value if isinstance(value, (float, int, httpx.Timeout)) and value else request_timeout


def _allowed_openai_params(value: object) -> Sequence[str] | None:
    try:
        return _STRINGS.validate_python(value) if value is not None else None
    except ValidationError:
        return None


def _native_responses_call(
    request: Mapping[str, object],
    model: str,
    forwarded_kwargs: Mapping[str, object],
    logging_obj: LiteLLMLoggingObj,
    is_async: bool,
) -> (
    ResponsesAPIResponse
    | BaseResponsesAPIStreamingIterator
    | Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
):
    """Resolve the deployment and hand the translated request to the native Responses executor."""
    litellm_params: Final = GenericLiteLLMParams(**forwarded_kwargs)
    provider_model, custom_llm_provider = resolve_model_provider_for_responses(
        model=model,
        custom_llm_provider=_optional_str(forwarded_kwargs.get("custom_llm_provider")),
        litellm_params=litellm_params,
        local_vars={},
    )
    if custom_llm_provider is None:
        raise ValueError(f"No provider resolved for {model}")
    provider_config: Final = resolve_responses_api_provider_config(
        provider_model, custom_llm_provider, forwarded_kwargs.get("model_info"), litellm_params.api_base
    )
    if provider_config is None:
        raise ValueError(f"{custom_llm_provider} has no Responses API config for {provider_model}")
    request_input, request_tools = apply_managed_file_id_mapping(
        input=request["input"],
        tools=request.get("tools"),
        kwargs=dict(forwarded_kwargs),
        local_vars={},
    )
    params: Final = {
        **forwarded_kwargs,
        **request,
        "input": request_input,
        "tools": request_tools,
        "custom_llm_provider": custom_llm_provider,
    }
    reasoning: Final = requested_reasoning(params, provider_model, custom_llm_provider)
    return execute_responses(
        http_handler=base_llm_http_handler,
        model=provider_model,
        input=request_input,
        provider_config=provider_config,
        requested_params=ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            {key: value for key, value in {**params, "reasoning": reasoning}.items() if key != "reasoning_effort"}
        ),
        allowed_openai_params=_allowed_openai_params(forwarded_kwargs.get("allowed_openai_params")),
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        user=_optional_str(request.get("user")),
        extra_headers=with_provider_affinity_header(
            ResponsesAPIRequestUtils.merge_client_forwarded_headers(
                extra_headers=_string_keyed_dict(forwarded_kwargs.get("extra_headers")),
                client_headers=_client_headers(forwarded_kwargs.get("headers")),
            ),
            litellm_params,
            _string_keyed_dict(request.get("metadata")),
            forwarded_kwargs,
            provider_model,
            custom_llm_provider,
        ),
        extra_body=_string_keyed_dict(forwarded_kwargs.get("extra_body")),
        timeout=_timeout(forwarded_kwargs.get("timeout")),
        is_async=is_async,
        stream=request.get("stream") is True,
        client=_http_client(forwarded_kwargs.get("client")),
        shared_session=_shared_session(forwarded_kwargs.get("shared_session")),
        litellm_metadata=_string_keyed_dict(forwarded_kwargs.get("litellm_metadata")) or {},
        request_kwargs=dict(forwarded_kwargs),
    )


def _native_logging_obj(
    logging_obj: LiteLLMLoggingObj | None, forwarded_kwargs: Mapping[str, object], request: Mapping[str, object]
) -> LiteLLMLoggingObj | None:
    """The logging object when the native executor can serve the call, None when only `aresponses` can.

    The executor skips the prompt management hooks `aresponses` runs, so a call that needs them re-enters it.
    """
    if logging_obj is None:
        return None
    params: Final = {**forwarded_kwargs, **request}
    if logging_obj.should_run_prompt_management_hooks(
        non_default_params=params, prompt_id=_optional_str(params.get("prompt_id"))
    ):
        return None
    return logging_obj


def _reentrant_kwargs(
    forwarded_kwargs: Mapping[str, object], request: Mapping[str, object], logging_obj: LiteLLMLoggingObj | None
) -> dict[str, object]:
    if logging_obj is not None:
        setattr(logging_obj, "call_type", CallTypes.anthropic_messages.value)
    return {**forwarded_kwargs, **request}


def _raise_mapped(error: Exception, model: str, forwarded_kwargs: Mapping[str, object]) -> NoReturn:
    raise litellm.exception_type(
        model=model,
        custom_llm_provider=_optional_str(forwarded_kwargs.get("custom_llm_provider")),
        original_exception=error,
        extra_kwargs=dict(forwarded_kwargs),
    ) from error


def _native_responses(
    request: Mapping[str, object],
    model: str,
    forwarded_kwargs: Mapping[str, object],
    logging_obj: LiteLLMLoggingObj,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    try:
        response: Final = _native_responses_call(request, model, forwarded_kwargs, logging_obj, is_async=False)
    except Exception as e:  # noqa: BLE001  # every provider failure maps to the litellm exception aresponses would raise
        _raise_mapped(e, model, forwarded_kwargs)
    if isinstance(response, Coroutine):
        raise TypeError("The sync Responses executor returned a coroutine")
    return response


async def _native_aresponses(
    request: Mapping[str, object],
    model: str,
    forwarded_kwargs: Mapping[str, object],
    logging_obj: LiteLLMLoggingObj,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    try:
        response: Final = _native_responses_call(request, model, forwarded_kwargs, logging_obj, is_async=True)
        return await response if isinstance(response, Coroutine) else response
    except Exception as e:  # noqa: BLE001  # every provider failure maps to the litellm exception aresponses would raise
        _raise_mapped(e, model, forwarded_kwargs)


class LiteLLMMessagesToResponsesAPIHandler:
    """
    Handles Anthropic /v1/messages requests for OpenAI / Azure models by
    calling the native Responses executor and translating the response back
    to Anthropic format.
    """

    @staticmethod
    async def async_anthropic_messages_handler(
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
        **kwargs: object,
    ) -> AnthropicMessagesResponse | AsyncIterator[bytes]:
        request: Final = _build_responses_request(
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
            extra_kwargs=kwargs,
        )
        forwarded_kwargs: Final = _forwarded_kwargs(kwargs)
        logging_obj: Final = litellm_logging_obj_from_kwargs(kwargs)

        safeguards_evaluator: Final = request_safeguards_evaluator(kwargs, messages)
        native_logging_obj: Final = _native_logging_obj(logging_obj, forwarded_kwargs, request)
        result: Final = (
            await litellm.aresponses(**_reentrant_kwargs(forwarded_kwargs, request, logging_obj))
            if native_logging_obj is None
            else await _native_aresponses(request, model, forwarded_kwargs, native_logging_obj)
        )

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result,
                model=local_model_name(model, kwargs.get("custom_llm_provider")),
                litellm_logging_obj=logging_obj,
                safeguards_evaluator=safeguards_evaluator,
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        served: Final = await with_safeguard_results(_ADAPTER.translate_response(result), safeguards_evaluator)
        _record_bridged_response(kwargs, result)
        return served

    @staticmethod
    def anthropic_messages_handler(
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
        _is_async: bool = False,
        **kwargs: object,
    ) -> (
        AnthropicMessagesResponse
        | AsyncIterator[bytes]
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

        request: Final = _build_responses_request(
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
            extra_kwargs=kwargs,
        )
        forwarded_kwargs: Final = _forwarded_kwargs(kwargs)
        logging_obj: Final = litellm_logging_obj_from_kwargs(kwargs)
        native_logging_obj: Final = _native_logging_obj(logging_obj, forwarded_kwargs, request)
        result: Final = (
            litellm.responses(**_reentrant_kwargs(forwarded_kwargs, request, logging_obj))
            if native_logging_obj is None
            else _native_responses(request, model, forwarded_kwargs, native_logging_obj)
        )

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result,
                model=local_model_name(model, kwargs.get("custom_llm_provider")),
                litellm_logging_obj=logging_obj,
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _ADAPTER.translate_response(result)
