import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import InstanceOf, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_object_list
from litellm.types.utils import ChatCompletionMessageToolCall, Function

_FIELDS: Final = TypeAdapter(InstanceOf[dict[str, object]])


def _fields(
    value: object,
) -> dict[str, object] | None:  # mutable-ok: rewrites must update the original provider container
    try:
        return _FIELDS.validate_python(value)
    except ValidationError:
        return None


def _items(value: object) -> tuple[object, ...]:
    return tuple(as_object_list(value) or ())


@dataclass(frozen=True, slots=True)
class RawToolCallGroup:
    format: Literal["openai", "legacy", "gemini", "bedrock"]
    message: dict[str, object]  # mutable-ok: rewrite only denied calls in the original message
    finish: dict[str, object]  # mutable-ok: synchronize the original finish reason after removing calls
    blocks: tuple[object, ...]
    calls: tuple[ChatCompletionMessageToolCall | None, ...]

    def rewrite(self, errors: Mapping[str, str]) -> None:
        messages: Final = tuple(errors[call.id] for call in self.calls if call is not None and call.id in errors)
        if not messages:
            return
        kept: Final = [block for block, call in zip(self.blocks, self.calls) if call is None or call.id not in errors]
        text: Final = "\n".join(messages)
        match self.format:
            case "openai" | "legacy":
                key: Final = "tool_calls" if self.format == "openai" else "function_call"
                self.message[key] = (kept or None) if self.format == "openai" else None
                content: Final = self.message.get("content")
                parts: Final = as_object_list(content)
                self.message["content"] = (
                    [*parts, {"type": "text", "text": text}]
                    if parts is not None
                    else f"{content}\n\n{text}"
                    if content
                    else text
                )
                if not self.message.get("tool_calls") and not self.message.get("function_call"):
                    self.finish["finish_reason"] = "stop"
            case "gemini":
                self.message["parts"] = [*kept, {"text": text}]
            case "bedrock":
                self.message["content"] = [*kept, {"text": text}]
                if not any((fields := _fields(block)) is not None and "toolUse" in fields for block in kept):
                    self.finish["stopReason"] = "end_turn"


def _call(block: object, format: str, identifier: str) -> ChatCompletionMessageToolCall | None:
    fields: Final = _fields(block)
    if fields is None:
        return None
    function_key: Final = {"openai": "function", "gemini": "functionCall", "bedrock": "toolUse"}.get(format)
    function: Final = fields if function_key is None else _fields(fields.get(function_key))
    if function is None:
        return None
    name: Final = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments_key: Final = "args" if format == "gemini" else "input" if format == "bedrock" else "arguments"
    arguments: Final = function.get(arguments_key)
    serialized: Final = (
        arguments if isinstance(arguments, str) else json.dumps(arguments) if arguments is not None else ""
    )
    return ChatCompletionMessageToolCall(
        id=identifier, type="function", function=Function(name=name, arguments=serialized)
    )


def _group(
    format: Literal["openai", "legacy", "gemini", "bedrock"],
    message: dict[str, object],  # mutable-ok: the group retains the provider message for in-place rewrites
    finish: dict[str, object],  # mutable-ok: the group retains the provider finish reason for in-place rewrites
    blocks: tuple[object, ...],
    index: int,
) -> RawToolCallGroup:
    # Positions keep missing or duplicate provider IDs from removing an allowed call.
    calls: Final = tuple(
        _call(block, format, f"raw_{format}_{index}_{position}") for position, block in enumerate(blocks)
    )
    return RawToolCallGroup(format, message, finish, blocks, calls)


def _choice_groups(index: int, choice: object) -> Iterator[RawToolCallGroup]:
    choice_fields: Final = _fields(choice)
    message: Final = _fields(choice_fields.get("message")) if choice_fields is not None else None
    if message is not None and choice_fields is not None:
        yield _group("openai", message, choice_fields, _items(message.get("tool_calls")), index)
        yield _group("legacy", message, choice_fields, (message.get("function_call"),), index)


def _candidate_groups(index: int, candidate: object) -> Iterator[RawToolCallGroup]:
    candidate_fields: Final = _fields(candidate)
    content: Final = _fields(candidate_fields.get("content")) if candidate_fields is not None else None
    if content is not None and candidate_fields is not None:
        yield _group("gemini", content, candidate_fields, _items(content.get("parts")), index)


def raw_tool_call_groups(response: object) -> Iterator[RawToolCallGroup]:
    fields: Final = _fields(response)
    if fields is None:
        return
    for index, choice in enumerate(_items(fields.get("choices"))):
        yield from _choice_groups(index, choice)
    for index, candidate in enumerate(_items(fields.get("candidates"))):
        yield from _candidate_groups(index, candidate)
    output: Final = _fields(fields.get("output"))
    bedrock_message: Final = _fields(output.get("message")) if output is not None else None
    if bedrock_message is not None:
        yield _group("bedrock", bedrock_message, fields, _items(bedrock_message.get("content")), 0)
