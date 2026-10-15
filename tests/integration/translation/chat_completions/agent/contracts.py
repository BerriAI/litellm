import json
from collections.abc import Mapping
from itertools import chain
from typing import Annotated, Final, Literal

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
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, JsonValue]] | None = None
    reasoning_items: list[dict[str, JsonValue]] | None = None


class Choice(Contract):
    index: Literal[0]
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]
    message: Message


class PromptDetails(Contract):
    cached_tokens: TokenCount | None = None
    cache_write_tokens: TokenCount | None = None
    cache_creation_tokens: TokenCount | None = None


class CompletionDetails(Contract):
    reasoning_tokens: TokenCount | None = None


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
