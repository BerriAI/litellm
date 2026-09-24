from dataclasses import dataclass
from typing import Any, Final

from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,  # pyright: ignore[reportPrivateUsage]  # shared helper already used by gemini/chat, context_caching, and vertex_and_google_ai_studio_gemini
    _transform_system_message,  # pyright: ignore[reportPrivateUsage]  # same helper the Gemini chat transformation uses to split system prompts
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.vertex_ai import ContentType, SystemInstructions, Tools


@dataclass(frozen=True, slots=True)
class GeminiCountTokensPayload:
    contents: list[ContentType]
    system_instruction: SystemInstructions | None
    tools: list[Tools] | None


def build_count_tokens_payload(
    model: str,
    messages: list[dict[str, Any]],
    system: object | None,
    tools: list[dict[str, Any]] | None,
) -> GeminiCountTokensPayload:
    """Translate an Anthropic Messages token-count request into the Gemini countTokens shape."""
    anthropic_request: Final = AnthropicMessagesRequest(
        model=model,
        messages=messages,
        **({"system": system} if isinstance(system, (str, list)) else {}),
        **({"tools": tools} if tools else {}),
    )
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_request, custom_llm_provider="gemini"
    )
    system_instruction, remaining_messages = _transform_system_message(
        supports_system_message=True,
        messages=list(openai_request["messages"]),
    )
    contents: Final = _gemini_convert_messages_with_history(
        messages=remaining_messages, model=model, custom_llm_provider="gemini"
    )
    openai_tools: Final = openai_request.get("tools")
    gemini_tools: Final = (
        VertexGeminiConfig()._map_function(  # pyright: ignore[reportPrivateUsage]  # same tool mapper the Gemini chat path uses
            value=[dict(tool) for tool in openai_tools],
            optional_params={},
        )
        if openai_tools
        else None
    )
    return GeminiCountTokensPayload(contents=contents, system_instruction=system_instruction, tools=gemini_tools)
