from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.types.utils import TokenCountMessagesFormat

_CONVERTIBLE_BLOCK_TYPES: Final = MappingProxyType(
    {
        "user": frozenset({"text", "image", "document", "tool_result"}),
        "assistant": frozenset({"text", "tool_use", "thinking", "redacted_thinking"}),
    }
)
_SOURCED_BLOCK_TYPES: Final = frozenset({"image", "document"})
_TOOL_RESULT_BLOCK_TYPES: Final = frozenset({"text", "image", "document", "tool_reference"})


_JSON_MESSAGES: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])


def _json_messages(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, JsonValue], ...]:
    try:
        return _JSON_MESSAGES.validate_python(messages)
    except ValidationError:
        return ()


def _source_label(block_type: str, source: JsonValue) -> str | None:
    if not isinstance(source, dict):
        return f"{block_type} without a source"
    match source.get("type"):
        case "base64" if source.get("data"):
            return None
        case "url" if source.get("url"):
            return None
        case "base64" | "url" as source_type:
            return f"{block_type} with an empty {source_type} source"
        case str() as source_type:
            return f"{block_type} with a {source_type} source"
        case _:
            return f"{block_type} with an untyped source"


def _block_label(block: JsonValue, convertible: frozenset[str]) -> str | None:
    if not isinstance(block, dict):
        return None
    block_type: Final = block.get("type")
    if not isinstance(block_type, str):
        return "a block without a type"
    if block_type not in convertible:
        return block_type
    return _source_label(block_type, block.get("source")) if block_type in _SOURCED_BLOCK_TYPES else None


def _tool_result_labels(block: JsonValue) -> Iterator[str]:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return iter(())
    inner: Final = block.get("content")
    if not isinstance(inner, list):
        return iter(())
    return (
        f"{label} inside a tool_result"
        for item in inner
        if (label := _block_label(item, _TOOL_RESULT_BLOCK_TYPES)) is not None
    )


def _content_labels(block: JsonValue, convertible: frozenset[str]) -> Iterator[str]:
    label: Final = _block_label(block, convertible)
    return chain(() if label is None else (label,), _tool_result_labels(block))


def _message_labels(message: Mapping[str, JsonValue]) -> Iterator[str]:
    role: Final = message.get("role")
    convertible: Final = _CONVERTIBLE_BLOCK_TYPES.get(role) if isinstance(role, str) else None
    content: Final = message.get("content")
    if convertible is None or not isinstance(content, list):
        return iter(())
    return chain.from_iterable(_content_labels(block, convertible) for block in content)


def unconvertible_anthropic_blocks(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(sorted(set(chain.from_iterable(_message_labels(message) for message in _json_messages(messages)))))


def unconvertible_blocks_error(
    messages: Sequence[Mapping[str, object]] | None, messages_format: TokenCountMessagesFormat
) -> str | None:
    match messages_format:
        case "openai":
            return None
        case "anthropic":
            labels: Final = unconvertible_anthropic_blocks(messages or ())
            return (
                "count_tokens can't convert these content blocks for this model, so the count would leave them out: "
                + ", ".join(labels)
                if labels
                else None
            )
        case _:
            assert_never(messages_format)
