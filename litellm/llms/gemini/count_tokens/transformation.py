from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

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


class AnthropicCountTokensInput(TypedDict):
    """The Anthropic-shaped fields of a token-count request before validation."""

    model: ReadOnly[str]
    messages: ReadOnly[Sequence[Mapping[str, object]]]
    system: ReadOnly[object]
    tools: ReadOnly[Sequence[Mapping[str, object]] | None]


@dataclass(frozen=True, slots=True)
class GeminiCountTokensPayload:
    contents: Sequence[ContentType]
    system_instruction: SystemInstructions | None
    tools: Sequence[Tools] | None


@dataclass(frozen=True, slots=True)
class InvalidAnthropicRequest:
    message: str


_ANTHROPIC_REQUEST: Final = TypeAdapter(AnthropicMessagesRequest)


def _validated_request(raw: AnthropicCountTokensInput) -> AnthropicMessagesRequest | InvalidAnthropicRequest:
    try:
        _ANTHROPIC_REQUEST.validate_python(raw)
    except ValidationError as e:
        return InvalidAnthropicRequest(message=str(e))
    return cast(AnthropicMessagesRequest, raw)  # cast-ok: validated above; pydantic returns lazy Iterable validators


def build_count_tokens_payload(raw: AnthropicCountTokensInput) -> GeminiCountTokensPayload | InvalidAnthropicRequest:
    """Translate an Anthropic Messages token-count request into the Gemini countTokens shape."""
    request: Final = _validated_request(raw)
    if isinstance(request, InvalidAnthropicRequest):
        return request
    try:
        openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            request, custom_llm_provider="gemini"
        )
    except (KeyError, TypeError, ValueError) as e:
        return InvalidAnthropicRequest(message=f"Invalid Anthropic Messages request: {e!r}")
    system_instruction, remaining_messages = _transform_system_message(
        supports_system_message=True, messages=openai_request["messages"]
    )
    contents: Final = _gemini_convert_messages_with_history(
        messages=remaining_messages, model=raw["model"], custom_llm_provider="gemini"
    )
    openai_tools: Final = openai_request.get("tools")
    gemini_tools: Final = (
        VertexGeminiConfig()._map_function(  # pyright: ignore[reportPrivateUsage]  # same tool mapper the Gemini chat path uses
            value=[dict(tool) for tool in openai_tools],  # mutable-ok: _map_function only accepts list[dict]
            optional_params={},  # mutable-ok: _map_function writes retrieval config into the dict it is given
        )
        if openai_tools
        else None
    )
    return GeminiCountTokensPayload(contents=contents, system_instruction=system_instruction, tools=gemini_tools)
