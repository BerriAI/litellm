from typing import AsyncIterator, Dict, List, Optional, Union, cast

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.utils import ModelResponse

########################################################
# init adapter
ANTHROPIC_ADAPTER = AnthropicAdapter()
########################################################


class LiteLLMCompletionTransformationHandler:
    @staticmethod
    async def anthropic_messages_handler(
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
        """Handle non-Anthropic models using the adapter"""

        # Prepare request data for adapter
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Add optional parameters
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

        # Use adapter to translate to OpenAI format
        openai_request = ANTHROPIC_ADAPTER.translate_completion_input_params(
            request_data
        )

        if openai_request is None:
            raise ValueError("Failed to translate request to OpenAI format")

        # Prepare completion kwargs
        completion_kwargs = dict(openai_request)

        # Add stream parameter
        if stream:
            completion_kwargs["stream"] = stream

        # Add specific kwargs we want to pass through
        excluded_keys = {"litellm_logging_obj", "anthropic_messages"}
        for key, value in kwargs.items():
            if (
                key not in excluded_keys
                and key not in completion_kwargs
                and value is not None
            ):
                completion_kwargs[key] = value

        try:
            # Call litellm.acompletion
            completion_response = await litellm.acompletion(**completion_kwargs)  # type: ignore

            if stream:
                # Transform streaming response using adapter
                transformed_stream = (
                    ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                        completion_response
                    )
                )
                if transformed_stream is not None:
                    return transformed_stream
                else:
                    raise ValueError("Failed to transform streaming response")
            else:
                # Transform response back to Anthropic format using adapter
                anthropic_response = (
                    ANTHROPIC_ADAPTER.translate_completion_output_params(
                        cast(ModelResponse, completion_response)
                    )
                )
                if anthropic_response is not None:
                    return anthropic_response
                else:
                    raise ValueError("Failed to transform response to Anthropic format")
        except Exception as e:
            raise ValueError(
                f"Error calling litellm.acompletion for non-Anthropic model: {str(e)}"
            )
