from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    pass

########################################################
# init adapter
ANTHROPIC_ADAPTER = AnthropicAdapter()
########################################################


class LiteLLMMessagesToCompletionTransformationHandler:
    @staticmethod
    def _prepare_completion_kwargs(
        *,
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare kwargs for litellm.completion/acompletion"""
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObject,
        )

        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if metadata:
            request_data["metadata"] = metadata
        if stop_sequences:
            request_data["stop_sequences"] = stop_sequences
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
        if top_k is not None:
            request_data["top_k"] = top_k
        if top_p is not None:
            request_data["top_p"] = top_p

        openai_request = ANTHROPIC_ADAPTER.translate_completion_input_params(
            request_data
        )

        if openai_request is None:
            raise ValueError("Failed to translate request to OpenAI format")

        completion_kwargs: Dict[str, Any] = dict(openai_request)

        if stream:
            completion_kwargs["stream"] = stream

        excluded_keys = {"anthropic_messages"}
        extra_kwargs = extra_kwargs or {}
        for key, value in extra_kwargs.items():
            if (
                key == "litellm_logging_obj"
                and value is not None
                and isinstance(value, LiteLLMLoggingObject)
            ):
                from litellm.types.utils import CallTypes

                setattr(value, "call_type", CallTypes.completion.value)
            if (
                key not in excluded_keys
                and key not in completion_kwargs
                and value is not None
            ):
                completion_kwargs[key] = value

        return completion_kwargs

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        """Handle non-Anthropic models asynchronously using the adapter"""

        completion_kwargs = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                metadata=metadata,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                extra_kwargs=kwargs,
            )
        )

        try:
            completion_response = await litellm.acompletion(**completion_kwargs)

            if stream:
                transformed_stream = (
                    ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                        completion_response
                    )
                )
                if transformed_stream is not None:
                    return transformed_stream
                raise ValueError("Failed to transform streaming response")
            else:
                anthropic_response = (
                    ANTHROPIC_ADAPTER.translate_completion_output_params(
                        cast(ModelResponse, completion_response)
                    )
                )
                if anthropic_response is not None:
                    return anthropic_response
                raise ValueError("Failed to transform response to Anthropic format")
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Error calling litellm.acompletion for non-Anthropic model: {str(e)}"
            )

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        _is_async: bool = False,
        **kwargs,
    ) -> Union[
        AnthropicMessagesResponse,
        AsyncIterator[Any],
        Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
    ]:
        """Handle non-Anthropic models using the adapter."""
        if _is_async is True:
            return LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                metadata=metadata,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )

        completion_kwargs = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                metadata=metadata,
                stop_sequences=stop_sequences,
                stream=stream,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                extra_kwargs=kwargs,
            )
        )

        try:
            completion_response = litellm.completion(**completion_kwargs)

            if stream:
                transformed_stream = (
                    ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                        completion_response
                    )
                )
                if transformed_stream is not None:
                    return transformed_stream
                raise ValueError("Failed to transform streaming response")
            else:
                anthropic_response = (
                    ANTHROPIC_ADAPTER.translate_completion_output_params(
                        cast(ModelResponse, completion_response)
                    )
                )
                if anthropic_response is not None:
                    return anthropic_response
                raise ValueError("Failed to transform response to Anthropic format")
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Error calling litellm.completion for non-Anthropic model: {str(e)}"
            )
