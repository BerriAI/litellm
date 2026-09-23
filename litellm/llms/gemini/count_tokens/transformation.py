"""Translate an Anthropic /v1/messages/count_tokens request into a Gemini
countTokens payload (contents + systemInstruction + tools)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,  # pyright: ignore[reportPrivateUsage]  # shared helper already used by gemini/chat, context_caching, and vertex_and_google_ai_studio_gemini
    _transform_system_message,
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
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload:
    anthropic_request: Final[AnthropicMessagesRequest] = cast(  # cast-ok: adapter reads only the keys supplied
        AnthropicMessagesRequest,
        {  # mutable-ok: transient request dict for the anthropic adapter
            "model": model,
            "messages": list(messages),  # mutable-ok: adapter contract takes a list of messages
            **({"system": system} if system else {}),  # mutable-ok: transient request dict for the anthropic adapter
            **({"tools": list(tools)} if tools else {}),  # mutable-ok: transient request dict for the anthropic adapter
        },
    )
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_request, custom_llm_provider="gemini"
    )
    system_instruction, remaining_messages = _transform_system_message(
        supports_system_message=True,
        messages=list(openai_request["messages"]),  # mutable-ok: helper pops the leading system message
    )
    contents: Final = _gemini_convert_messages_with_history(
        messages=remaining_messages,
        model=model,
        custom_llm_provider="gemini",
    )
    openai_tools: Final = openai_request.get("tools")
    gemini_tools: Final = (
        VertexGeminiConfig()._map_function(
            value=[dict(tool) for tool in openai_tools],  # mutable-ok: _map_function takes plain tool dicts
            optional_params={},  # mutable-ok: _map_function signature takes a dict
        )
        if openai_tools
        else None
    )
    return GeminiCountTokensPayload(
        contents=contents,
        system_instruction=system_instruction,
        tools=gemini_tools,
    )
