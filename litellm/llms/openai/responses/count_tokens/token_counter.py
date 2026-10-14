"""
OpenAI Token Counter implementation using the Responses API /input_tokens endpoint.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.llms.base_llm.base_utils import AnthropicCountRequestFormat, BaseTokenCounter
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.count_tokens.handler import (
    OpenAICountTokensHandler,
)
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
)
from litellm.types.llms.anthropic import (
    AllAnthropicPassThroughMessageValues,
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
    AnthropicSystemParam,
)
from litellm.types.utils import LlmProviders, TokenCountResponse

# Global handler instance - reuse across all token counting requests
openai_count_tokens_handler: Final = OpenAICountTokensHandler()
_ANTHROPIC_MESSAGES_ADAPTER: Final = TypeAdapter[tuple[AllAnthropicPassThroughMessageValues, ...]](
    tuple[AllAnthropicPassThroughMessageValues, ...]
)
_ANTHROPIC_TOOLS_ADAPTER: Final = TypeAdapter[tuple[AllAnthropicToolsValues, ...]](tuple[AllAnthropicToolsValues, ...])
_OPENAI_CHAT_REQUEST_ADAPTER: Final = TypeAdapter[Mapping[str, object]](Mapping[str, object])
_OPENAI_ITEMS_ADAPTER: Final = TypeAdapter[tuple[Mapping[str, object], ...]](tuple[Mapping[str, object], ...])
_OPENAI_CONVERTED_MESSAGES_ADAPTER: Final = TypeAdapter[tuple[tuple[Mapping[str, object], ...], str | None]](
    tuple[tuple[Mapping[str, object], ...], str | None]
)


def _materialize_anthropic_messages(
    messages: tuple[AllAnthropicPassThroughMessageValues, ...],
) -> tuple[AllAnthropicPassThroughMessageValues, ...]:
    return tuple(
        message
        if isinstance(message["content"], (str, list))
        else cast(  # cast-ok: validation established content item types before iterator materialization
            AllAnthropicPassThroughMessageValues,
            {**message, "content": list(message["content"])},
        )
        for message in messages
    )


def _convert_anthropic_count_request(
    model: str,
    messages: Sequence[Mapping[str, object]] | None,
    tools: Sequence[Mapping[str, object]] | None,
    system: AnthropicSystemParam,
) -> tuple[tuple[Mapping[str, object], ...], str | None, tuple[Mapping[str, object], ...] | None] | None:
    if messages is None:
        return None
    try:
        typed_messages: Final = _ANTHROPIC_MESSAGES_ADAPTER.validate_python(messages)
        validated_messages: Final = _materialize_anthropic_messages(typed_messages)
        validated_tools: Final = _ANTHROPIC_TOOLS_ADAPTER.validate_python(tools) if tools is not None else None
        anthropic_request: Final = AnthropicMessagesRequest(
            model=model,
            messages=[dict(message) for message in validated_messages],
            tools=list(validated_tools) if validated_tools is not None else None,
            system=system,
        )
        if litellm.use_chat_completions_url_for_anthropic_messages:
            from litellm.llms.anthropic.pass_through.adapters.transformation import (
                LiteLLMAnthropicMessagesAdapter,
            )

            openai_request_value, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
                anthropic_request,
                custom_llm_provider=LlmProviders.OPENAI.value,
            )
            openai_request: Final = _OPENAI_CHAT_REQUEST_ADAPTER.validate_python(openai_request_value)
            openai_messages_value: Final = openai_request.get("messages")
            converted_messages: Final = _OPENAI_CONVERTED_MESSAGES_ADAPTER.validate_python(
                OpenAICountTokensConfig.messages_to_responses_input(  # pyright: ignore[reportUnknownMemberType]  # output is validated immediately
                    [dict(item) for item in _OPENAI_ITEMS_ADAPTER.validate_python(openai_messages_value)]
                )
            )
            openai_tools_value: Final = openai_request.get("tools")
            converted_tools: Final = (
                _OPENAI_ITEMS_ADAPTER.validate_python(openai_tools_value) if openai_tools_value is not None else None
            )
            return converted_messages[0], converted_messages[1], converted_tools

        responses_request: Final = LiteLLMAnthropicToResponsesAPIAdapter().translate_request(
            anthropic_request,
            include_encrypted_reasoning=False,
        )
        response_input_items: Final = _OPENAI_ITEMS_ADAPTER.validate_python(responses_request.get("input", []))
        instructions_value: Final = responses_request.get("instructions")
        response_instructions: Final = instructions_value if isinstance(instructions_value, str) else None
        tools_value: Final = responses_request.get("tools")
        response_tools: Final = _OPENAI_ITEMS_ADAPTER.validate_python(tools_value) if tools_value is not None else None
        return response_input_items, response_instructions, response_tools
    except ValidationError:
        return None


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter implementation for OpenAI provider using the Responses API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.OPENAI.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, object]] | None,
        contents: list[dict[str, object]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, object]] | None = None,
        system: AnthropicSystemParam = None,
        request_format: AnthropicCountRequestFormat = "chat",
    ) -> TokenCountResponse | None:
        """
        Count tokens using OpenAI's Responses API /input_tokens endpoint.
        """
        if not messages:
            return None

        deployment = deployment or {}
        litellm_params: Final = deployment.get("litellm_params", {})

        # Get OpenAI API key from deployment config or environment
        api_key = litellm_params.get("api_key")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            verbose_logger.warning("No OpenAI API key found for token counting")
            return None

        api_base: Final = litellm_params.get("api_base")

        # Convert chat messages to Responses API input format
        anthropic_count_request: Final = (
            _convert_anthropic_count_request(model_to_use, messages, tools, system)
            if request_format == "anthropic"
            else None
        )
        resolved_count_request: Final = (
            anthropic_count_request
            if anthropic_count_request is not None
            else _OPENAI_CONVERTED_MESSAGES_ADAPTER.validate_python(
                OpenAICountTokensConfig.messages_to_responses_input(  # pyright: ignore[reportUnknownMemberType]  # output is validated immediately
                    messages
                )
            )
        )
        input_items: Final = [dict(item) for item in resolved_count_request[0]]
        instructions_from_request: Final = resolved_count_request[1]
        instructions: Final = (
            instructions_from_request
            if instructions_from_request is not None
            else (system if isinstance(system, str) else str(system))
            if system is not None
            else None
        )
        resolved_tools: Final = (
            [dict(tool) for tool in anthropic_count_request[2]]
            if anthropic_count_request is not None and anthropic_count_request[2] is not None
            else tools
        )

        # If no input items were produced (e.g., system-only messages), fall back to local counting
        if not input_items:
            return None

        try:
            result: Final = await openai_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                input=input_items if input_items is not None else [],
                api_key=api_key,
                api_base=api_base,
                tools=resolved_tools,
                instructions=instructions,
            )

            if result is not None:
                return TokenCountResponse.model_validate(
                    {
                        "total_tokens": result.get("input_tokens", 0),
                        "request_model": request_model,
                        "model_used": model_to_use,
                        "tokenizer_type": "openai_api",
                        "original_response": result,
                    }
                )
        except OpenAIError as e:
            verbose_logger.warning("OpenAI CountTokens API error: status=%s, message=%s", e.status_code, e.message)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="openai_api",
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        except Exception as e:
            verbose_logger.warning("Error calling OpenAI CountTokens API: %s", e)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="openai_api",
                error=True,
                error_message=str(e),
                status_code=500,
            )

        return None
