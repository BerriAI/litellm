import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, Literal

from integration._support.client import object_value
from integration.translation.chat_completions.cross_model.contracts import (
    CacheCreationDetails,
    Contract,
    NonemptyString,
    TokenCount,
)
from pydantic import Field, JsonValue, TypeAdapter

JSON_ARGUMENTS: Final = TypeAdapter(dict[str, JsonValue])


class TextBlock(Contract):
    type: Literal["text"]
    text: str
    citations: list[dict[str, JsonValue]] | None = None


class ThinkingBlock(Contract):
    type: Literal["thinking"]
    thinking: str
    signature: NonemptyString


class RedactedThinkingBlock(Contract):
    type: Literal["redacted_thinking"]
    data: NonemptyString


class ToolUseBlock(Contract):
    type: Literal["tool_use"]
    id: NonemptyString
    name: NonemptyString
    input: dict[str, JsonValue]
    caller: dict[str, JsonValue] | None = None


MessageBlock = Annotated[TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolUseBlock, Field(discriminator="type")]


class MessageOutputDetails(Contract):
    thinking_tokens: TokenCount | None = None


class MessageServerToolUse(Contract):
    web_search_requests: TokenCount | None = None
    web_fetch_requests: TokenCount | None = None


class MessageUsage(Contract):
    input_tokens: TokenCount
    output_tokens: TokenCount
    cache_creation_input_tokens: TokenCount | None = None
    cache_read_input_tokens: TokenCount | None = None
    cache_creation: CacheCreationDetails | None = None
    output_tokens_details: MessageOutputDetails | None = None
    server_tool_use: MessageServerToolUse | None = None
    service_tier: str | None = None
    inference_geo: str | None = None


class MessagesResponse(Contract):
    id: NonemptyString
    type: Literal["message"]
    role: Literal["assistant"]
    model: NonemptyString
    content: Annotated[list[MessageBlock], Field(min_length=1)]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "pause_turn", "refusal"]
    stop_sequence: str | None
    usage: MessageUsage
    container: dict[str, JsonValue] | None = None
    diagnostics: dict[str, JsonValue] | None = None
    stop_details: dict[str, JsonValue] | None = None


class OutputText(Contract):
    type: Literal["output_text"]
    text: str
    annotations: list[dict[str, JsonValue]]
    logprobs: list[dict[str, JsonValue]] | None = None


class OutputMessage(Contract):
    type: Literal["message"]
    id: NonemptyString
    status: Literal["completed", "incomplete", "in_progress"]
    role: Literal["assistant"]
    content: Annotated[list[OutputText], Field(min_length=1)]
    phase: str | None = None


class ReasoningSummary(Contract):
    type: Literal["summary_text"]
    text: str


class OutputReasoning(Contract):
    type: Literal["reasoning"]
    id: NonemptyString
    summary: list[ReasoningSummary] | None = None
    content: list[OutputText] | None = None
    encrypted_content: str | None = None


class OutputFunctionCall(Contract):
    type: Literal["function_call"]
    id: NonemptyString
    call_id: NonemptyString
    name: NonemptyString
    arguments: str
    status: Literal["completed", "incomplete", "in_progress"] | None = None
    namespace: str | None = None


OutputItem = Annotated[OutputMessage | OutputReasoning | OutputFunctionCall, Field(discriminator="type")]


class InputTokenDetails(Contract):
    cached_tokens: TokenCount | None = None
    cache_write_tokens: TokenCount | None = None
    text_tokens: TokenCount | None = None
    audio_tokens: TokenCount | None = None
    image_tokens: TokenCount | None = None
    video_tokens: TokenCount | None = None


class OutputTokenDetails(Contract):
    reasoning_tokens: TokenCount | None = None
    text_tokens: TokenCount | None = None
    audio_tokens: TokenCount | None = None


class ResponsesUsage(Contract):
    input_tokens: TokenCount
    output_tokens: TokenCount
    total_tokens: TokenCount
    input_tokens_details: InputTokenDetails | None = None
    output_tokens_details: OutputTokenDetails | None = None


class ResponsesResponse(Contract):
    id: NonemptyString
    object: Literal["response"]
    created_at: Annotated[int | float, Field(ge=0)]
    status: Literal["completed", "incomplete", "failed", "in_progress", "cancelled", "queued"]
    error: dict[str, JsonValue] | None
    incomplete_details: dict[str, JsonValue] | None
    instructions: str | None
    metadata: dict[str, str]
    model: NonemptyString
    output: Annotated[list[OutputItem], Field(min_length=1)]
    parallel_tool_calls: bool
    tool_choice: str | dict[str, JsonValue]
    tools: list[dict[str, JsonValue]]
    usage: ResponsesUsage
    max_output_tokens: TokenCount | None = None
    previous_response_id: str | None = None
    reasoning: dict[str, JsonValue] | None = None
    store: bool | None = None
    temperature: float | None = None
    text: dict[str, JsonValue] | None = None
    top_p: float | None = None
    truncation: str | None = None
    user: str | None = None
    background: bool | None = None
    completed_at: Annotated[int | float, Field(ge=0)] | None = None
    frequency_penalty: float | None = None
    max_tool_calls: TokenCount | None = None
    presence_penalty: float | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None
    service_tier: str | None = None
    top_logprobs: TokenCount | None = None
    provider_specific_fields: dict[str, JsonValue] | None = None
    access_programs: dict[str, JsonValue] | None = None
    billing: dict[str, JsonValue] | None = None
    moderation: dict[str, JsonValue] | None = None
    tool_usage: dict[str, JsonValue] | None = None


MESSAGES_OPTIONAL: Final = MappingProxyType(
    {
        "": frozenset(("container", "diagnostics", "stop_details")),
        "content[]": frozenset(("caller", "citations")),
        "usage": frozenset(
            (
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "cache_creation",
                "output_tokens_details",
                "server_tool_use",
                "service_tier",
                "inference_geo",
                "speed",
            )
        ),
    }
)
RESPONSES_OPTIONAL: Final = MappingProxyType(
    {
        "": frozenset(
            (
                "parallel_tool_calls",
                "tool_choice",
                "tools",
                "max_output_tokens",
                "previous_response_id",
                "reasoning",
                "store",
                "temperature",
                "text",
                "top_p",
                "truncation",
                "user",
                "background",
                "completed_at",
                "frequency_penalty",
                "max_tool_calls",
                "presence_penalty",
                "prompt_cache_key",
                "prompt_cache_retention",
                "safety_identifier",
                "service_tier",
                "top_logprobs",
                "provider_specific_fields",
                "access_programs",
                "billing",
                "moderation",
                "tool_usage",
            )
        ),
        "output[]": frozenset(("namespace", "status", "phase")),
        "output[].content[]": frozenset(("logprobs",)),
        "usage": frozenset(("input_tokens_details", "output_tokens_details", "cost")),
    }
)


def messages_structure(body: Mapping[str, JsonValue]) -> JsonValue:
    response: Final = MessagesResponse.model_validate(body)
    tools: Final = tuple(block for block in response.content if isinstance(block, ToolUseBlock))
    assert len({tool.id for tool in tools}) == len(tools), body
    assert response.stop_reason == ("tool_use" if tools else "end_turn"), body
    return _structure(dict(body), "", MESSAGES_OPTIONAL, "content", "tool_use")


def responses_structure(body: Mapping[str, JsonValue]) -> JsonValue:
    response: Final = ResponsesResponse.model_validate(body)
    assert response.status == "completed", body
    assert response.error is None and response.incomplete_details is None, body
    assert response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens, body
    tools: Final = tuple(item for item in response.output if isinstance(item, OutputFunctionCall))
    assert len({tool.call_id for tool in tools}) == len(tools), body
    assert len({item.id for item in response.output}) == len(response.output), body
    for tool in tools:
        JSON_ARGUMENTS.validate_json(tool.arguments)
    return _structure(dict(body), "", RESPONSES_OPTIONAL, "output", "function_call")


def _structure(
    value: JsonValue,
    path: str,
    optional: Mapping[str, frozenset[str]],
    items_path: str,
    tool_type: str,
) -> JsonValue:
    if isinstance(value, dict):
        excluded: Final = optional.get(path, frozenset[str]())
        return {
            key: _structure(item, f"{path}.{key}" if path else key, optional, items_path, tool_type)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        tools: Final = tuple(
            item for item in value if path == items_path and object_value(item).get("type") == tool_type
        )
        compared: Final = tools or tuple(value)
        shapes: Final = tuple(_structure(item, f"{path}[]", optional, items_path, tool_type) for item in compared)
        ordered: Final = tuple(sorted({json.dumps(shape, sort_keys=True) for shape in shapes}))
        return [TypeAdapter(JsonValue).validate_json(shape) for shape in ordered]
    if path == "created_at":
        return "number"
    return type(value).__name__
