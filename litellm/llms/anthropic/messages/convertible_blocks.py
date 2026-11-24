import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain, dropwhile, groupby, starmap
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias, TypeVar

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.common_utils import is_unrendered_block_without_text
from litellm.litellm_core_utils.prompt_templates.factory import sanitize_anthropic_tool_use_id
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
    if is_unrendered_block_without_text(block):
        return None
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
    from litellm.llms.anthropic.messages.native.utils import normalize_native_messages

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


@dataclass(frozen=True, slots=True)
class _KeptPart:
    block: object
    scanned_text: str


_KeptParts: TypeAlias = Mapping[int, _KeptPart]
_NO_KEPT_PARTS: Final[_KeptParts] = MappingProxyType({})


def _scannable_text(text: object) -> str:
    return json.dumps(text, default=str, ensure_ascii=False)


def _non_str_text(block: object) -> _KeptPart | None:
    match as_str_mapping(block):
        case {"type": "text", "text": str()}:
            return None
        case {"type": "text", "text": text}:
            return _KeptPart(block=block, scanned_text=_scannable_text(text))
        case _:
            return None


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
        if _non_str_text(text_block) is not None
    )


def _tool_result_text_errors(block: object) -> Iterator[str]:
    match block:
        case {"type": "tool_result", "tool_use_id": object() as tool_use_id, "content": object() as content}:
            return _non_string_text_errors(f"tool_result {tool_use_id}", content)
        case {"type": "tool_result", "content": object() as content}:
            return _non_string_text_errors("tool_result", content)
        case _:
            return iter(())


_BlockTextErrors: TypeAlias = Callable[[int, int, object], Iterator[str]]


def _user_text_errors(message_index: int, block_index: int, block: object) -> Iterator[str]:
    if _non_str_text(block) is None:
        return iter(())
    return iter((f"messages.{message_index}.content.{block_index}.text: Input should be a valid string",))


def _user_block_text_errors(message_index: int, block_index: int, block: object) -> Iterator[str]:
    return chain(_user_text_errors(message_index, block_index, block), _tool_result_text_errors(block))


def _message_text_errors(block_errors: _BlockTextErrors, message_index: int, message: object) -> Iterator[str]:
    return chain.from_iterable(
        block_errors(message_index, block_index, block)
        for block_index, block in enumerate(_user_content_blocks(message))
    )


def _first_text_error(block_errors: _BlockTextErrors, messages: Iterable[object]) -> str | None:
    return next(chain.from_iterable(starmap(partial(_message_text_errors, block_errors), enumerate(messages))), None)


def malformed_text_error(messages: Iterable[object]) -> str | None:
    return _first_text_error(_user_block_text_errors, messages)


def non_string_user_text_error(messages: Iterable[object]) -> str | None:
    return _first_text_error(_user_text_errors, messages)


def malformed_text_count_error(
    messages: Iterable[Mapping[str, object]] | None,
    messages_format: TokenCountMessagesFormat,
    deployment: Mapping[str, object] | None,
) -> str | None:
    if messages_format != "anthropic" or deployment_drops_params(deployment):
        return None
    return malformed_text_error(messages or ())


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


def _non_string_text_parts(content: object) -> _KeptParts:
    return MappingProxyType(
        {index: part for index, block in enumerate(_list_items(content)) if (part := _non_str_text(block)) is not None}
    )


def _kept_parts_of(tool_result: Mapping[str, object]) -> _KeptParts:
    return _non_string_text_parts(tool_result.get("content"))


def _content_blocks(content: object) -> tuple[object, ...]:
    match content:
        case str() if content:
            return ({"type": "text", "text": content},)
        case _:
            return _list_items(content)


def _restored_part(block: object, kept: _KeptPart | None) -> object:
    return kept.block if kept is not None and tool_result_text_part(block) == kept.scanned_text else block


def _with_kept_parts(tool_result: Mapping[str, object], parts: _KeptParts) -> object:
    if not parts:
        return tool_result
    blocks: Final = _content_blocks(tool_result.get("content"))
    return {**tool_result, "content": [_restored_part(part, parts.get(index)) for index, part in enumerate(blocks)]}


_V = TypeVar("_V")  # rebind-ok: a TypeVar cannot be Final
_ToolResultValues: TypeAlias = tuple[tuple[str, _V], ...]


def _entry_tool_use_id(entry: tuple[str, object]) -> str:
    return entry[0]


def _entry_value(entry: tuple[str, _V]) -> _V:
    return entry[1]


def _tool_result_value(block: object, value_of: Callable[[Mapping[str, object]], _V]) -> tuple[tuple[str, _V], ...]:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result:
            return ((sanitize_anthropic_tool_use_id(tool_use_id), value_of(tool_result)),)
        case _:
            return ()


def _values_by_tool_use_id(original: object, value_of: Callable[[Mapping[str, object]], _V]) -> _ToolResultValues[_V]:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, _list_items(original)))
    values: Final = chain.from_iterable(_tool_result_value(block, value_of) for block in blocks)
    return tuple(sorted(values, key=_entry_tool_use_id))


def _restored_tool_result(
    block: object,
    queues: Mapping[str, Iterator[_V]],
    restore: Callable[[Mapping[str, object], _V], object],
    missing: _V,
) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result if (
            sanitized := sanitize_anthropic_tool_use_id(tool_use_id)
        ) in queues:
            return restore(tool_result, next(queues[sanitized], missing))
        case _:
            return block


def _with_tool_result_values(
    rewritten: Sequence[object],
    values: _ToolResultValues[_V],
    restore: Callable[[Mapping[str, object], _V], object],
    missing: _V,
) -> Sequence[object]:
    queues: Final = MappingProxyType(
        {
            tool_use_id: iter(tuple(map(_entry_value, group)))
            for tool_use_id, group in groupby(values, key=_entry_tool_use_id)
        }
    )
    restore_block: Final = partial(_restored_tool_result, queues=queues, restore=restore, missing=missing)
    return [_with_user_blocks(message, restore_block) for message in rewritten]


def keep_non_string_tool_result_texts(rewritten: Sequence[object], original: object) -> Sequence[object]:
    values: Final = _values_by_tool_use_id(original, _kept_parts_of)
    if not any(parts for _, parts in values):
        return rewritten
    return _with_tool_result_values(rewritten, values, _with_kept_parts, _NO_KEPT_PARTS)


def _url_source_url(block: object) -> str | None:
    match as_str_mapping(block):
        case {"type": "image" | "document", "source": {"type": "url", "url": str(url)}}:
            return url
        case _:
            return None


def _is_document(block: object) -> bool:
    match as_str_mapping(block):
        case {"type": "document"}:
            return True
        case _:
            return False


@dataclass(frozen=True, slots=True, order=True)
class _UrlBlockPlace:
    in_tool_result: bool
    url: str


def _url_sourced_blocks(block: object, in_tool_result: bool = False) -> tuple[tuple[_UrlBlockPlace, object], ...]:
    url: Final = _url_source_url(block)
    if url is not None:
        return ((_UrlBlockPlace(in_tool_result, url), block),)
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list() as content}:
            in_this_tool_result: Final = partial(_url_sourced_blocks, in_tool_result=True)
            return tuple(chain.from_iterable(map(in_this_tool_result, _list_items(content))))
        case _:
            return ()


def _entry_place(entry: tuple[_UrlBlockPlace, object]) -> _UrlBlockPlace:
    return entry[0]


def _entry_block(entry: tuple[_UrlBlockPlace, object]) -> object:
    return entry[1]


def _restored_url_document(
    block: object, originals: Mapping[_UrlBlockPlace, Iterator[object]], in_tool_result: bool = False
) -> object:
    url: Final = _url_source_url(block)
    if url is not None:
        remaining: Final = originals.get(_UrlBlockPlace(in_tool_result, url))
        original: Final = None if remaining is None else next(remaining, None)
        return original if _is_document(original) else block
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list() as content} as tool_result:
            restore: Final = partial(_restored_url_document, originals=originals, in_tool_result=True)
            return {**tool_result, "content": list(map(restore, _list_items(content)))}
        case _:
            return block


def keep_url_documents(rewritten: Sequence[object], original: object) -> Sequence[object]:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, _list_items(original)))
    by_place: Final = tuple(sorted(chain.from_iterable(map(_url_sourced_blocks, blocks)), key=_entry_place))
    if not any(_is_document(block) for _, block in by_place):
        return rewritten
    originals_in_order: Final = MappingProxyType(
        {place: iter(tuple(map(_entry_block, group))) for place, group in groupby(by_place, key=_entry_place)}
    )
    restore: Final = partial(_restored_url_document, originals=originals_in_order)
    return [_with_user_blocks(message, restore) for message in rewritten]


def _is_error_of(tool_result: Mapping[str, object]) -> bool | None:
    match tool_result:
        case {"is_error": bool(is_error)}:
            return is_error
        case _:
            return None


def _with_is_error(tool_result: Mapping[str, object], is_error: bool | None) -> object:
    return tool_result if is_error is None else {**tool_result, "is_error": is_error}


def keep_tool_result_errors(rewritten: Sequence[object], original: object) -> Sequence[object]:
    values: Final = _values_by_tool_use_id(original, _is_error_of)
    if not any(is_error is not None for _, is_error in values):
        return rewritten
    return _with_tool_result_values(rewritten, values, _with_is_error, None)


_NO_EXTRAS: Final[Mapping[str, object]] = MappingProxyType({})


def _is_system_row(message: object) -> bool:
    match as_str_mapping(message):
        case {"role": "system"}:
            return True
        case _:
            return False


def is_non_empty_text_block(block: object) -> bool:
    match as_str_mapping(block):
        case {"type": "text", "text": str(text)}:
            return bool(text)
        case _:
            return False


def _has_text(content: object) -> bool:
    match content:
        case str():
            return bool(content)
        case _:
            return any(map(is_non_empty_text_block, _list_items(content)))


def _system_row_extras(message: object) -> Mapping[str, object] | None:
    match as_str_mapping(message):
        case {"role": "system", "content": content} as row if _has_text(content):
            return MappingProxyType({key: value for key, value in row.items() if key not in ("role", "content")})
        case _:
            return None


def _with_system_extras(message: object, extras: Iterator[Mapping[str, object]]) -> object:
    match as_str_mapping(message):
        case {"role": "system"} as row:
            return {**row, **next(extras, _NO_EXTRAS)}
        case _:
            return message


def keep_system_message_fields(rewritten: Sequence[object], original: object) -> Sequence[object]:
    in_sequence: Final = dropwhile(_is_system_row, _list_items(original))
    extras: Final = tuple(extra for message in in_sequence if (extra := _system_row_extras(message)) is not None)
    if not any(extras) or sum(map(_is_system_row, rewritten)) != len(extras):
        return rewritten
    remaining: Final = iter(extras)
    return [_with_system_extras(message, remaining) for message in rewritten]


_Restorer: TypeAlias = Callable[[Sequence[object], object], Sequence[object]]
_RESTORERS: Final[tuple[_Restorer, ...]] = (
    keep_non_string_tool_result_texts,
    keep_url_documents,
    keep_tool_result_errors,
    keep_system_message_fields,
)


def keep_original_fields(rewritten: Sequence[object], original: object) -> Sequence[object]:
    return reduce(lambda kept, restore: restore(kept, original), _RESTORERS, rewritten)


def _without_non_string_text(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            parts: Final = _list_items(tool_result.get("content"))
            return {**tool_result, "content": [part for part in parts if _non_str_text(part) is None]}
        case _:
            return block


def _without_non_string_texts(
    message: dict[str, object],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
) -> dict[str, object] | None:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    match message:
        case {"role": "user", "content": list()}:
            blocks: Final = _user_content_blocks(message)
            kept: Final = [_without_non_string_text(block) for block in blocks if _non_str_text(block) is None]
            return {**message, "content": kept} if kept or not blocks else None
        case _:
            return message


def drop_non_string_texts(
    messages: Sequence[dict[str, object]],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    drop_params: object,
) -> Sequence[dict[str, object]]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    if not drops_unsupported_params(drop_params) or malformed_text_error(messages) is None:
        return messages
    return [kept for message in messages if (kept := _without_non_string_texts(message)) is not None]


def emptied_messages_error(sent: Sequence[object], kept: Sequence[object]) -> str | None:
    if kept or malformed_text_error(sent) is None:
        return None
    return "messages: no message has content left after dropping empty or non-string text blocks"


def deployment_drops_params(deployment: Mapping[str, object] | None) -> bool:
    match deployment:
        case {"litellm_params": {"drop_params": object() as drop_params}}:
            return drops_unsupported_params(drop_params)
        case _:
            return drops_unsupported_params(None)


def reject_malformed_text(
    messages: Iterable[Mapping[str, object]], model: str, forwarded_kwargs: Mapping[str, object]
) -> None:
    _reject_text_error(
        malformed_text_error,
        messages,
        forwarded_kwargs.get("drop_params"),
        model,
        forwarded_kwargs.get("custom_llm_provider"),
    )


def reject_non_string_user_text(messages: Iterable[object], drop_params: object, model: str, provider: object) -> None:
    _reject_text_error(non_string_user_text_error, messages, drop_params, model, provider)


def _reject_text_error(
    first_error: Callable[[Iterable[object]], str | None],
    messages: Iterable[object],
    drop_params: object,
    model: str,
    provider: object,
) -> None:
    if drops_unsupported_params(drop_params):
        return
    error: Final = first_error(messages)
    if error is None:
        return
    _raise_bad_request(error, model, provider)


def reject_emptied_messages(sent: Sequence[object], kept: Sequence[object], model: str, provider: str | None) -> None:
    error: Final = emptied_messages_error(sent, kept)
    if error is None:
        return
    _raise_bad_request(error, model, provider)


def _raise_bad_request(error: str, model: str, provider: object) -> NoReturn:
    raise litellm.BadRequestError(message=error, model=model, llm_provider=_provider_name(model, provider))


def _provider_name(model: str, provider: object) -> str:
    if isinstance(provider, str) and provider:
        return provider
    try:
        return litellm.get_llm_provider(model=model)[1]
    except litellm.BadRequestError:
        return ""
