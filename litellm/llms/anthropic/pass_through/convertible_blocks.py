import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.factory import sanitize_anthropic_tool_use_id
from litellm.llms.anthropic.pass_through.messages.utils import normalize_native_messages
from litellm.types.utils import TokenCountMessagesFormat

_CONVERTIBLE_BLOCK_TYPES: Final = MappingProxyType(
    {
        "user": frozenset({"text", "image", "document", "tool_result"}),
        "assistant": frozenset({"text", "tool_use", "thinking", "redacted_thinking"}),
    }
)
_SOURCED_BLOCK_TYPES: Final = frozenset({"image", "document"})
_TOOL_RESULT_BLOCK_TYPES: Final = frozenset({"text", "image", "document"})


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
    normalized: Final = _json_messages(normalize_native_messages(messages))
    return tuple(sorted(set(chain.from_iterable(_message_labels(message) for message in normalized))))


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


_OBJECTS: Final = TypeAdapter(tuple[object, ...])


def _list_items(value: object) -> tuple[object, ...]:
    return _OBJECTS.validate_python(value) if isinstance(value, list) else ()


def _has_non_string_text(block: object) -> bool:
    match block:
        case {"type": "text", "text": str()}:
            return False
        case {"type": "text", "text": object()}:
            return True
        case _:
            return False


def tool_result_text_part(block: object) -> str | None:
    match block:
        case {"type": "text", "text": str(text)}:
            return text
        case {"type": "text", "text": object()}:
            return None
        case {"type": "text"}:
            return ""
        case _:
            return None


def tool_result_text_parts(tool_result: Mapping[str, object]) -> tuple[str, ...]:
    parts: Final = map(tool_result_text_part, _list_items(tool_result.get("content")))
    return tuple(part for part in parts if part is not None)


def _user_content_blocks(message: Mapping[str, object]) -> tuple[object, ...]:
    return _list_items(message.get("content")) if message.get("role") == "user" else ()


def _non_string_text_errors(label: str, content: object) -> Iterator[str]:
    return (
        f"{label} content.{index}.text: Input should be a valid string"
        for index, text_block in enumerate(_list_items(content))
        if _has_non_string_text(text_block)
    )


def _tool_result_text_errors(block: object) -> Iterator[str]:
    match block:
        case {"type": "tool_result", "tool_use_id": object() as tool_use_id, "content": object() as content}:
            return _non_string_text_errors(f"tool_result {tool_use_id}", content)
        case {"type": "tool_result", "content": object() as content}:
            return _non_string_text_errors("tool_result", content)
        case _:
            return iter(())


def malformed_tool_result_text_error(messages: Iterable[Mapping[str, object]]) -> str | None:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, messages))
    return next(chain.from_iterable(map(_tool_result_text_errors, blocks)), None)


@dataclass(frozen=True, slots=True)
class _KeptPart:
    block: object
    scanned_text: str


_KeptParts: TypeAlias = Mapping[int, _KeptPart]


def _scannable_text(text: object) -> str:
    return json.dumps(text, default=str)


def _scannable_part(part: object) -> object:
    match as_str_mapping(part):
        case {"type": "text", "text": str()}:
            return part
        case {"type": "text", "text": text} as text_block:
            return {**text_block, "text": _scannable_text(text)}
        case _:
            return part


def _scannable_block(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            return {
                **tool_result,
                "content": [_scannable_part(part) for part in _list_items(tool_result.get("content"))],
            }
        case _:
            return block


def _scannable_message(message: object) -> object:
    match as_str_mapping(message):
        case {"role": "user", "content": list()} as user_message:
            return {
                **user_message,
                "content": [_scannable_block(block) for block in _user_content_blocks(user_message)],
            }
        case _:
            return message


def _scannable_messages(messages: object) -> object:
    items: Final = _list_items(messages)
    return [_scannable_message(message) for message in items] if items else messages


def scannable_tool_result_request(request: object) -> object:
    match as_str_mapping(request):
        case {"messages": messages} as request_fields:
            return {**request_fields, "messages": _scannable_messages(messages)}
        case None:
            return request
        case request_fields:
            return {**request_fields}


def _kept_part(block: object) -> _KeptPart | None:
    match as_str_mapping(block):
        case {"type": "text", "text": str()}:
            return None
        case {"type": "text", "text": text}:
            return _KeptPart(block=block, scanned_text=_scannable_text(text))
        case _:
            return None


def _non_string_text_parts(content: object) -> _KeptParts:
    return MappingProxyType(
        {index: part for index, block in enumerate(_list_items(content)) if (part := _kept_part(block)) is not None}
    )


def _tool_result_non_string_texts(block: object) -> tuple[tuple[str, _KeptParts], ...]:
    match block:
        case {"type": "tool_result", "tool_use_id": str(tool_use_id), "content": object() as content}:
            parts: Final = _non_string_text_parts(content)
            return ((sanitize_anthropic_tool_use_id(tool_use_id), parts),) if parts else ()
        case _:
            return ()


def _content_blocks(content: object) -> tuple[object, ...]:
    match content:
        case str() if content:
            return ({"type": "text", "text": content},)
        case _:
            return _list_items(content)


def _text_of(block: object) -> str | None:
    match as_str_mapping(block):
        case {"type": "text", "text": str(text)}:
            return text
        case _:
            return None


def _restored_part(block: object, kept: _KeptPart | None) -> object:
    return kept.block if kept is not None and _text_of(block) == kept.scanned_text else block


def _merged_content(blocks: tuple[object, ...], kept: _KeptParts) -> tuple[object, ...]:
    return tuple(_restored_part(block, kept.get(index)) for index, block in enumerate(blocks))


def _restored_block(block: object, kept: Mapping[str, _KeptParts]) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result if (
            sanitize_anthropic_tool_use_id(tool_use_id) in kept
        ):
            parts: Final = kept[sanitize_anthropic_tool_use_id(tool_use_id)]
            return {**tool_result, "content": list(_merged_content(_content_blocks(tool_result.get("content")), parts))}
        case _:
            return block


def _restored_message(message: object, kept: Mapping[str, _KeptParts]) -> object:
    match as_str_mapping(message):
        case {"role": "user", "content": list()} as user_message:
            blocks: Final = _list_items(user_message.get("content"))
            return {**user_message, "content": [_restored_block(block, kept) for block in blocks]}
        case _:
            return message


def keep_non_string_tool_result_texts(rewritten: Sequence[object], original: object) -> Sequence[object]:
    original_messages: Final = filter(None, map(as_str_mapping, _list_items(original)))
    blocks: Final = chain.from_iterable(map(_user_content_blocks, original_messages))
    kept: Final = MappingProxyType(dict(chain.from_iterable(map(_tool_result_non_string_texts, blocks))))
    if not kept:
        return rewritten
    return [_restored_message(message, kept) for message in rewritten]


def _without_non_string_text(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            parts: Final = _list_items(tool_result.get("content"))
            return {**tool_result, "content": [part for part in parts if not _has_non_string_text(part)]}
        case _:
            return block


def _without_non_string_texts(
    message: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    match message:
        case {"role": "user", "content": list()}:
            return {**message, "content": [_without_non_string_text(block) for block in _user_content_blocks(message)]}
        case _:
            return {**message}


def drop_non_string_tool_result_texts(
    messages: Sequence[dict[str, object]],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    drop_params: object,
) -> Sequence[dict[str, object]]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    if not drops_unsupported_params(drop_params) or malformed_tool_result_text_error(messages) is None:
        return messages
    return [_without_non_string_texts(message) for message in messages]


def deployment_drops_params(deployment: Mapping[str, object] | None) -> bool:
    match deployment:
        case {"litellm_params": {"drop_params": object() as drop_params}}:
            return drops_unsupported_params(drop_params)
        case _:
            return drops_unsupported_params(None)


def reject_malformed_tool_result_text(
    messages: Iterable[Mapping[str, object]], model: str, forwarded_kwargs: Mapping[str, object]
) -> None:
    if drops_unsupported_params(forwarded_kwargs.get("drop_params")):
        return
    error: Final = malformed_tool_result_text_error(messages)
    if error is None:
        return
    provider: Final = forwarded_kwargs.get("custom_llm_provider")
    raise litellm.BadRequestError(
        message=error, model=model, llm_provider=provider if isinstance(provider, str) else ""
    )
