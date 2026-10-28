import json
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, Literal

from integration._support.client import JSON_OBJECT, list_value, object_value
from integration.translation.case import StreamingTranslationTestCase
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_message import Annotation, ChatCompletionAudio
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

NonemptyString = Annotated[str, Field(min_length=1)]
TokenCount = Annotated[int, Field(ge=0)]
JSON_ARGUMENTS: Final = TypeAdapter(dict[str, JsonValue])


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="allow")


class Function(Contract):
    name: NonemptyString
    arguments: str


class ToolCall(Contract):
    id: NonemptyString
    type: Literal["function"]
    function: Function


class Message(Contract):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None
    annotations: list[Annotation] | None = None
    audio: ChatCompletionAudio | None = None
    function_call: Function | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, JsonValue]] | None = None
    reasoning_items: list[dict[str, JsonValue]] | None = None


class Choice(Contract):
    index: Annotated[int, Field(ge=0, le=0)]
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]
    message: Message
    logprobs: ChoiceLogprobs | None = None


class CacheCreationDetails(Contract):
    ephemeral_5m_input_tokens: TokenCount | None = None
    ephemeral_1h_input_tokens: TokenCount | None = None


class PromptDetails(Contract):
    cached_tokens: TokenCount | None = None
    cache_write_tokens: TokenCount | None = None
    cache_creation_tokens: TokenCount | None = None
    audio_tokens: TokenCount | None = None
    text_tokens: TokenCount | None = None
    image_tokens: TokenCount | None = None
    cache_creation_token_details: CacheCreationDetails | None = None


class CompletionDetails(Contract):
    reasoning_tokens: TokenCount | None = None
    audio_tokens: TokenCount | None = None
    text_tokens: TokenCount | None = None
    accepted_prediction_tokens: TokenCount | None = None
    rejected_prediction_tokens: TokenCount | None = None


class Usage(Contract):
    prompt_tokens: TokenCount
    completion_tokens: TokenCount
    total_tokens: TokenCount
    prompt_tokens_details: PromptDetails | None = None
    completion_tokens_details: CompletionDetails | None = None


class Completion(Contract):
    id: NonemptyString
    object: Literal["chat.completion"]
    created: TokenCount
    model: NonemptyString
    choices: Annotated[list[Choice], Field(min_length=1, max_length=1)]
    usage: Usage
    system_fingerprint: str | None = None
    service_tier: str | None = None


OPTIONAL_FIELDS: Final = MappingProxyType(
    {
        "": frozenset(("system_fingerprint", "service_tier")),
        "choices[]": frozenset(("logprobs",)),
        "choices[].message": frozenset(("refusal", "annotations", "audio", "function_call")),
        "usage": frozenset(("prompt_tokens_details", "completion_tokens_details")),
    }
)
PROVIDER_EXTENSIONS: Final = MappingProxyType(
    {
        "": frozenset(
            (
                "billing",
                "tool_usage",
                "content_filters",
                "access_programs",
                "frequency_penalty",
                "presence_penalty",
                "vertex_ai_grounding_metadata",
                "vertex_ai_url_context_metadata",
                "vertex_ai_safety_results",
                "vertex_ai_citation_metadata",
            )
        ),
        "choices[]": frozenset(("provider_specific_fields",)),
        "choices[].message": frozenset(
            ("provider_specific_fields", "reasoning_content", "thinking_blocks", "reasoning_items", "images")
        ),
        "choices[].message.tool_calls[]": frozenset(("index", "caller", "provider_specific_fields")),
        "usage": frozenset(
            ("cache_creation_input_tokens", "cache_read_input_tokens", "inference_geo", "service_tier", "speed")
        ),
    }
)


def response_structure(body: Mapping[str, JsonValue]) -> JsonValue:
    assert_completion(body)
    return _structure(dict(body), "")


def _structure(value: JsonValue, path: str) -> JsonValue:
    if path == "choices[].message.content":
        return "str | null"
    if isinstance(value, dict):
        empty_tools: Final[frozenset[str]] = (
            frozenset(("tool_calls",))
            if path == "choices[].message" and value.get("tool_calls") in (None, [])
            else frozenset[str]()
        )
        excluded: Final = (
            OPTIONAL_FIELDS.get(path, frozenset[str]()) | PROVIDER_EXTENSIONS.get(path, frozenset[str]()) | empty_tools
        )
        return {
            key: _structure(item, f"{path}.{key}" if path else key)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [_structure(item, f"{path}[]") for item in value]
    return type(value).__name__


class FunctionDelta(Contract):
    name: str | None = None
    arguments: str | None = None


class ToolDelta(Contract):
    index: TokenCount
    id: NonemptyString | None = None
    type: Literal["function"] | None = None
    function: FunctionDelta | None = None


def assembled_tools(deltas: tuple[dict[str, JsonValue], ...]) -> tuple[ToolCall, ...]:
    groups: Final = tuple(delta.get("tool_calls") or [] for delta in deltas)
    lists: Final = TypeAdapter(tuple[list[ToolDelta], ...]).validate_python(groups)
    calls: Final = tuple(chain.from_iterable(lists))
    assert {call.index for call in calls} == {0, 1}
    return tuple(_assembled_tool(tuple(call for call in calls if call.index == index)) for index in (0, 1))


def _assembled_tool(parts: tuple[ToolDelta, ...]) -> ToolCall:
    functions: Final = tuple(part.function for part in parts if part.function is not None)
    assert any(part.type == "function" for part in parts)
    return ToolCall(
        id="".join(part.id or "" for part in parts),
        type="function",
        function=Function(
            name="".join(function.name or "" for function in functions),
            arguments="".join(function.arguments or "" for function in functions),
        ),
    )


def assembled_message(deltas: tuple[dict[str, JsonValue], ...]) -> Message:
    return Message(
        role="assistant",
        content="".join(_text(delta, "content") for delta in deltas) or None,
        tool_calls=list(assembled_tools(deltas)),
        reasoning_content="".join(_text(delta, "reasoning_content") for delta in deltas) or None,
        thinking_blocks=_assembled_thinking(_blocks(deltas, "thinking_blocks")),
        reasoning_items=_blocks(deltas, "reasoning_items"),
    )


def _text(delta: Mapping[str, JsonValue], key: str) -> str:
    value: Final = delta.get(key)
    assert value is None or isinstance(value, str), (key, value)
    return value or ""


def _blocks(deltas: tuple[dict[str, JsonValue], ...], key: str) -> list[dict[str, JsonValue]]:
    blocks: Final = tuple(list_value(delta[key]) for delta in deltas if delta.get(key) is not None)
    return [object_value(block) for block in chain.from_iterable(blocks)]


def _assembled_thinking(blocks: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    indexes: Final = tuple(dict.fromkeys(str(block.get("index", 0)) for block in blocks))
    return [
        _thinking_block(tuple(block for block in blocks if str(block.get("index", 0)) == index)) for index in indexes
    ]


def _thinking_block(parts: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    kind: Final = parts[0]["type"]
    assert kind in ("thinking", "redacted_thinking") and all(part["type"] == kind for part in parts), parts
    fields: Final = ("thinking", "signature") if kind == "thinking" else ("data",)
    return {"type": kind, **{key: "".join(_text(part, key) for part in parts) for key in fields}}


def message_values(message: Message) -> Mapping[str, JsonValue]:
    return {
        "role": message.role,
        "content": message.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": JSON_ARGUMENTS.validate_json(call.function.arguments),
                },
            }
            for call in message.tool_calls or []
        ],
        "reasoning_content": message.reasoning_content or None,
        "thinking_blocks": [_block_values(block) for block in message.thinking_blocks or []],
        "reasoning_items": [_reasoning_item_values(item) for item in message.reasoning_items or []],
    }


def _block_values(block: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in block.items() if key != "index"}


def _reasoning_item_values(item: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in item.items() if key != "content" or value not in (None, [])}


def assert_stream_preserved(
    message: Message, usage: Mapping[str, JsonValue], case: StreamingTranslationTestCase
) -> None:
    assert case.expected_litellm_message is not None and case.expected_litellm_usage is not None, case.id
    expected: Final = Message.model_validate(case.expected_litellm_message)
    assert message_values(message) == message_values(expected), (
        case.id,
        message_values(message),
        message_values(expected),
    )
    tokens: Final = Usage.model_validate(usage)
    assert tokens.prompt_tokens > 0 and tokens.completion_tokens > 0, case.id
    assert tokens.total_tokens == tokens.prompt_tokens + tokens.completion_tokens, case.id
    assert _usage_values(usage) == _usage_values(case.expected_litellm_usage), (
        case.id,
        _usage_values(usage),
        _usage_values(case.expected_litellm_usage),
    )


def _usage_values(usage: Mapping[str, JsonValue], path: str = "usage") -> Mapping[str, JsonValue]:
    return {
        key: dict(_usage_values(value, f"{path}.{key}")) if isinstance(value, dict) else value
        for key, value in usage.items()
        if value is not None
        and key not in PROVIDER_EXTENSIONS.get(path, frozenset[str]())
        and (path != "usage" or key != "cost")
    }


def assert_completion(body: Mapping[str, JsonValue]) -> Completion:
    completion: Final = Completion.model_validate(body)
    assert completion.usage.prompt_tokens > 0 and completion.usage.completion_tokens > 0
    assert completion.usage.total_tokens == completion.usage.prompt_tokens + completion.usage.completion_tokens
    return completion


def tool_shape(body: Mapping[str, JsonValue]) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    completion: Final = assert_completion(body)
    choice: Final = completion.choices[0]
    assert choice.finish_reason == "tool_calls", body
    calls: Final = choice.message.tool_calls
    assert calls is not None and len(calls) == 2, body
    return tool_calls_shape(tuple(calls))


def tool_calls_shape(calls: tuple[ToolCall, ...]) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    assert len({call.id for call in calls}) == 2, calls
    assert {call.function.name for call in calls} == {"search_code", "run_tests"}, calls
    shapes: Final = tuple(sorted(_tool_shape(call) for call in calls))
    assert shapes == (("run_tests", (("path", "str"),)), ("search_code", (("query", "str"),))), calls
    return shapes


def _tool_shape(call: ToolCall) -> tuple[str, tuple[tuple[str, str], ...]]:
    arguments: Final = JSON_ARGUMENTS.validate_json(call.function.arguments)
    assert all(isinstance(value, str) and value for value in arguments.values()), call
    return call.function.name, tuple(sorted((key, type(value).__name__) for key, value in arguments.items()))


def assert_final_text(body: Mapping[str, JsonValue]) -> None:
    completion: Final = assert_completion(body)
    choice: Final = completion.choices[0]
    assert choice.finish_reason == "stop", body
    assert isinstance(choice.message.content, str) and choice.message.content.strip(), body
    assert not choice.message.tool_calls, body


TOOL_RESULTS: Final[Mapping[str, JsonValue]] = {
    "search_code": {
        "matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = retries_remaining(3, attempts)"}]
    },
    "run_tests": {"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - assert 4 == 2"},
}
FOLLOWUP: Final[dict[str, JsonValue]] = {
    "role": "user",
    "content": "Stop after these diagnostics. Do not call more tools or change files. "
    "Summarize the bug, the caller found and the failed test in two short sentences.",
}


def tool_results(message: Message) -> list[JsonValue]:
    assert message.tool_calls is not None
    return [
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(TOOL_RESULTS[call.function.name])}
        for call in message.tool_calls
    ]


def followup_request(request: Mapping[str, JsonValue], response: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    choice: Final = object_value(list_value(response["choices"])[0])
    message: Final = Message.model_validate(choice["message"])
    assistant: Final = JSON_OBJECT.validate_python(message.model_dump(mode="json", exclude_none=True))
    return {
        **request,
        "messages": [*list_value(request["messages"]), assistant, *tool_results(message), FOLLOWUP],
    }
