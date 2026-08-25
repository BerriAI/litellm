"""
Handler for the Anthropic v1/messages -> OpenAI Responses API path.

Used when the target model is an OpenAI or Azure model.
"""

from collections.abc import AsyncIterator, Coroutine, Mapping
from typing import Any, Final

import litellm
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
    AnthropicOutputConfig,
    AnthropicOutputSchema,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse

from ..utils import local_model_name
from .streaming_iterator import AnthropicResponsesStreamWrapper
from .transformation import LiteLLMAnthropicToResponsesAPIAdapter

_ADAPTER: Final = LiteLLMAnthropicToResponsesAPIAdapter()


def _forwarded_kwargs(extra_kwargs: Mapping[str, object] | None) -> Mapping[str, object]:
    """The litellm-specific kwargs forwarded verbatim onto the Responses API request."""
    return extra_kwargs or {}


def _build_responses_kwargs(
    *,
    max_tokens: int,
    messages: list[dict],
    model: str,
    context_management: dict | None = None,
    metadata: dict | None = None,
    output_config: AnthropicOutputConfig | None = None,
    stop_sequences: list[str] | None = None,
    stream: bool | None = False,
    system: str | None = None,
    temperature: float | None = None,
    thinking: dict | None = None,
    tool_choice: dict | None = None,
    tools: list[AllAnthropicToolsValues | dict] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    output_format: AnthropicOutputSchema | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the kwargs dict to pass directly to litellm.responses() / litellm.aresponses().
    """
    # Build a typed AnthropicMessagesRequest for the adapter
    request_data: Final[AnthropicMessagesRequest] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if context_management:
        request_data["context_management"] = context_management
    if output_config:
        request_data["output_config"] = output_config
    if metadata:
        request_data["metadata"] = metadata
    if system:
        request_data["system"] = system
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
    responses_kwargs: Final = _ADAPTER.translate_request(anthropic_request)

    # Normalize reasoning effort based on model capabilities
    # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
    reasoning: Final = responses_kwargs.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        from litellm.llms.anthropic.experimental_pass_through.utils import (
            normalize_reasoning_effort_value,
        )

        effort: Final = reasoning["effort"]
        normalized: Final = normalize_reasoning_effort_value(
            effort,
            model=model,
            custom_llm_provider=(extra_kwargs or {}).get("custom_llm_provider"),
        )
        if normalized != effort:
            responses_kwargs["reasoning"] = {**reasoning, "effort": normalized}

    if stream:
        responses_kwargs["stream"] = True

    # Forward litellm-specific kwargs (api_key, api_base, logging obj, etc.)
    excluded: Final = {"anthropic_messages"}
    forwarded_kwargs: Final = _forwarded_kwargs(extra_kwargs)
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

    return responses_kwargs


class LiteLLMMessagesToResponsesAPIHandler:
    """
    Handles Anthropic /v1/messages requests for OpenAI / Azure models by
    calling litellm.responses() / litellm.aresponses() directly and translating
    the response back to Anthropic format.
    """

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: list[dict],
        model: str,
        context_management: dict | None = None,
        metadata: dict | None = None,
        output_config: AnthropicOutputConfig | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool | None = False,
        system: str | None = None,
        temperature: float | None = None,
        thinking: dict | None = None,
        tool_choice: dict | None = None,
        tools: list[AllAnthropicToolsValues | dict] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        output_format: AnthropicOutputSchema | None = None,
        **kwargs,
    ) -> AnthropicMessagesResponse | AsyncIterator[bytes]:
        responses_kwargs: Final = _build_responses_kwargs(
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

        result: Final = await litellm.aresponses(**responses_kwargs)

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=local_model_name(model, kwargs.get("custom_llm_provider"))
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _ADAPTER.translate_response(result)

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: list[dict],
        model: str,
        context_management: dict | None = None,
        metadata: dict | None = None,
        output_config: AnthropicOutputConfig | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool | None = False,
        system: str | None = None,
        temperature: float | None = None,
        thinking: dict | None = None,
        tool_choice: dict | None = None,
        tools: list[AllAnthropicToolsValues | dict] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        output_format: AnthropicOutputSchema | None = None,
        _is_async: bool = False,
        **kwargs,
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

        # Sync path
        responses_kwargs: Final = _build_responses_kwargs(
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

        result: Final = litellm.responses(**responses_kwargs)

        if stream:
            wrapper: Final = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=local_model_name(model, kwargs.get("custom_llm_provider"))
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        return _ADAPTER.translate_response(result)
