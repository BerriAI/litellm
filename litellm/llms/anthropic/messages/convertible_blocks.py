import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from itertools import chain, groupby, islice
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.factory import sanitize_anthropic_tool_use_id
from litellm.llms.anthropic.messages.native.utils import normalize_native_messages
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


_ScanPart: TypeAlias = Callable[[object], object]
_Carried: TypeAlias = Callable[[object], bool]


def _scannable_text(text: object) -> str:
    return json.dumps(text, default=str, ensure_ascii=False)


def _is_non_string_text(block: object) -> bool:
    match as_str_mapping(block):
        case {"type": "text", "text": str()}:
            return False
        case {"type": "text", "text": _}:
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


def _user_content_blocks(message: object) -> tuple[object, ...]:
    match as_str_mapping(message):
        case {"role": "user", "content": content}:
            return _list_items(content)
        case _:
            return ()


def _with_user_blocks(message: object, transform: Callable[[object], object]) -> object:
    match as_str_mapping(message):
        case {"role": "user", "content": list()} as user_message:
            return {**user_message, "content": [transform(block) for block in _user_content_blocks(user_message)]}
        case _:
            return message


def _non_string_text_errors(label: str, content: object) -> Iterator[str]:
    return (
        f"{label} content.{index}.text: Input should be a valid string"
        for index, text_block in enumerate(_list_items(content))
        if _is_non_string_text(text_block)
    )


def _tool_result_text_errors(block: object) -> Iterator[str]:
    match block:
        case {"type": "tool_result", "tool_use_id": object() as tool_use_id, "content": object() as content}:
            return _non_string_text_errors(f"tool_result {tool_use_id}", content)
        case {"type": "tool_result", "content": object() as content}:
            return _non_string_text_errors("tool_result", content)
        case _:
            return iter(())


def malformed_tool_result_text_error(messages: Iterable[object]) -> str | None:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, messages))
    return next(chain.from_iterable(map(_tool_result_text_errors, blocks)), None)


def malformed_tool_result_count_error(
    messages: Iterable[Mapping[str, object]] | None,
    messages_format: TokenCountMessagesFormat,
    deployment: Mapping[str, object] | None,
) -> str | None:
    if messages_format != "anthropic" or deployment_drops_params(deployment):
        return None
    return malformed_tool_result_text_error(messages or ())


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


def _scannable_messages(messages: object) -> object:
    items: Final = _list_items(messages)
    return [_with_user_blocks(message, _scannable_block) for message in items] if items else messages


def scannable_tool_result_request(request: object) -> object:
    match as_str_mapping(request):
        case {"messages": messages} as request_fields:
            return {**request_fields, "messages": _scannable_messages(messages)}
        case None:
            return request
        case request_fields:
            return {**request_fields}


def _content_blocks(content: object) -> tuple[object, ...]:
    match content:
        case str() if content:
            return ({"type": "text", "text": content},)
        case _:
            return _list_items(content)


def _rewritten_part(original: object, rewritten: object) -> object:
    match (as_str_mapping(original), as_str_mapping(rewritten)):
        case ({"type": "text"} as text_block, {"type": "text", "text": str(text)}):
            unchanged: Final = text == tool_result_text_part(_scannable_part(original))
            return original if unchanged else {**text_block, "text": text}
        case ({"type": str(block_type), "source": source}, {"source": rewritten_source}) if (
            block_type in _SOURCED_BLOCK_TYPES and source == rewritten_source
        ):
            return original
        case _:
            return rewritten


def _carried_tool_result_part(scan_part: _ScanPart, part: object) -> bool:
    return scan_part(_scannable_part(part)) is not None


def _merged_part(original: object, rewritten: Iterator[object], carried: _Carried) -> tuple[object, ...]:
    if not carried(original):
        return (original,)
    return tuple(_rewritten_part(original, part) for part in islice(rewritten, 1))


def _merged_parts(originals: Iterable[object], rewritten: Iterator[object], carried: _Carried) -> tuple[object, ...]:
    return tuple(chain.from_iterable(_merged_part(part, rewritten, carried) for part in originals))


def _merged_content(original: object, rewritten: object, carried: _Carried) -> object:
    original_parts: Final = _list_items(original)
    if not isinstance(original, list):
        return rewritten
    remaining: Final = iter(_content_blocks(rewritten))
    merged: Final = _merged_parts(original_parts, remaining, carried)
    return [*merged, *remaining]


def _restored_tool_result(
    rewritten: Mapping[str, object], original: Mapping[str, object], carried: _Carried
) -> Mapping[str, object]:
    return {
        **original,
        **rewritten,
        "content": _merged_content(original.get("content"), rewritten.get("content"), carried),
    }


_ToolResultEntry: TypeAlias = tuple[str, Mapping[str, object]]


def _tool_result_entry(block: object) -> tuple[_ToolResultEntry, ...]:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result:
            return ((sanitize_anthropic_tool_use_id(tool_use_id), tool_result),)
        case _:
            return ()


def _entry_tool_use_id(entry: _ToolResultEntry) -> str:
    return entry[0]


def _entry_tool_result(entry: _ToolResultEntry) -> Mapping[str, object]:
    return entry[1]


def _restored_block(
    block: object, originals: Mapping[str, Iterator[Mapping[str, object]]], carried: _Carried
) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result if (
            sanitize_anthropic_tool_use_id(tool_use_id) in originals
        ):
            original: Final = next(originals[sanitize_anthropic_tool_use_id(tool_use_id)], tool_result)
            return _restored_tool_result(tool_result, original, carried)
        case _:
            return block


def _restored_tool_results(rewritten: Sequence[object], original: object, scan_part: _ScanPart) -> Sequence[object]:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, _list_items(original)))
    entries: Final = tuple(sorted(chain.from_iterable(map(_tool_result_entry, blocks)), key=_entry_tool_use_id))
    if not entries:
        return rewritten
    originals_in_order: Final = MappingProxyType(
        {
            tool_use_id: iter(tuple(map(_entry_tool_result, group)))
            for tool_use_id, group in groupby(entries, key=_entry_tool_use_id)
        }
    )
    restore: Final = partial(
        _restored_block, originals=originals_in_order, carried=partial(_carried_tool_result_part, scan_part)
    )
    return [_with_user_blocks(message, restore) for message in rewritten]


_Turn: TypeAlias = tuple[Mapping[str, object], ...]
_Run: TypeAlias = tuple[str | None, tuple[object, ...]]


def _turn_role(message: object) -> str | None:
    match as_str_mapping(message):
        case {"role": str(role)} if role in _CONVERTIBLE_BLOCK_TYPES:
            return role
        case _:
            return None


def _runs(messages: Iterable[object]) -> tuple[_Run, ...]:
    return tuple((role, tuple(group)) for role, group in groupby(messages, key=_turn_role))


def _turn_roles(runs: Iterable[_Run]) -> tuple[str, ...]:
    return tuple(role for role, _ in runs if role is not None)


def _turn(messages: Iterable[object]) -> _Turn:
    return tuple(filter(None, map(as_str_mapping, messages)))


def _carried_turn_block(role: str, scan_part: _ScanPart, block: object) -> bool:
    match as_str_mapping(block):
        case {"type": str(block_type)} if block_type not in _CONVERTIBLE_BLOCK_TYPES[role]:
            return False
        case {"type": str(block_type)} if block_type in _SOURCED_BLOCK_TYPES:
            return scan_part(block) is not None
        case _:
            return True


def _turn_blocks(turn: _Turn) -> Iterator[object]:
    return chain.from_iterable(_content_blocks(message.get("content")) for message in turn)


def _merged_turn(original: _Turn, rewritten: _Turn, carried: _Carried) -> _Turn:
    remaining: Final = iter(tuple(_turn_blocks(rewritten)))
    contents: Final = tuple(
        _merged_parts(_content_blocks(message.get("content")), remaining, carried) for message in original
    )
    completed: Final = (*contents[:-1], (*contents[-1], *remaining))
    return tuple(
        {**message, "content": list(content)} for message, content in zip(original, completed, strict=True) if content
    )


def _restored_run(run: _Run, original_turns: Iterator[_Turn], scan_part: _ScanPart) -> tuple[object, ...]:
    match run:
        case (str(role), messages):
            return _merged_turn(next(original_turns), _turn(messages), partial(_carried_turn_block, role, scan_part))
        case (_, messages):
            return messages


def _restored_turns(rewritten: Sequence[object], original: object, scan_part: _ScanPart) -> Sequence[object]:
    rewritten_runs: Final = _runs(rewritten)
    original_runs: Final = _runs(_list_items(original))
    if _turn_roles(rewritten_runs) != _turn_roles(original_runs):
        return rewritten
    original_turns: Final = iter(tuple(_turn(messages) for role, messages in original_runs if role is not None))
    return list(chain.from_iterable(_restored_run(run, original_turns, scan_part) for run in rewritten_runs))


def restore_client_blocks(rewritten: Sequence[object], original: object, scan_part: _ScanPart) -> Sequence[object]:
    return _restored_turns(_restored_tool_results(rewritten, original, scan_part), original, scan_part)


def _without_non_string_text(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            parts: Final = _list_items(tool_result.get("content"))
            return {**tool_result, "content": [part for part in parts if not _is_non_string_text(part)]}
        case _:
            return block


def _without_non_string_texts(
    message: dict[str, object],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
) -> dict[str, object]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    match message:
        case {"role": "user", "content": list()}:
            return {**message, "content": [_without_non_string_text(block) for block in _user_content_blocks(message)]}
        case _:
            return message


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
