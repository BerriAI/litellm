import base64
import json
import posixpath
import re
from bisect import bisect_left
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import chain, groupby
from operator import attrgetter
from types import MappingProxyType
from typing import Final, NamedTuple
from urllib.parse import unquote, urlparse

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping

_TOOL_NAME_MAX_LENGTH: Final = 100
_DEFAULT_TOOL_RESULT_NAME: Final = "tool_result"
_NO_TOOL_INPUT: Final[Mapping[str, object]] = MappingProxyType({})

_DATA_URL_RE: Final = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<params>(?:;[^;,]+)*?)(?P<b64>;base64)?,", re.IGNORECASE)
_URLSAFE_TO_STANDARD_B64: Final = str.maketrans("-_", "+/")
_OBJECT_LIST: Final = TypeAdapter(list[object])


def str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _list_items(value: object) -> Sequence[object]:
    try:
        return _OBJECT_LIST.validate_python(value, strict=True)
    except ValidationError:
        return ()


class FilePart(NamedTuple):
    name: str | None
    data: bytes | None
    mime_hint: str | None
    inline: bool
    oversize: bool
    message_index: int
    undecodable: bool = False


def _split_data_url(value: str) -> tuple[str | None, str | None]:
    match: Final = _DATA_URL_RE.match(value)
    if not match:
        if value.lower().startswith("data:"):
            return None, None
        return None, value
    mime: Final = match.group("mime") or None
    if not match.group("b64"):
        return mime, None
    return mime, value[match.end() :]


def _b64decode_or_none(payload: str) -> bytes | None:
    try:
        return base64.b64decode(payload, validate=True)
    except ValueError:
        return None


def _decoded_base64(cleaned: str) -> bytes | None:
    strict: Final = _b64decode_or_none(cleaned)
    if strict is not None or ("-" not in cleaned and "_" not in cleaned):
        return strict
    return _b64decode_or_none(cleaned.translate(_URLSAFE_TO_STANDARD_B64))


def _inline_part(name: str | None, value: str, size_limit: int | None, message_index: int) -> FilePart | None:
    mime_hint, payload = _split_data_url(value)
    if payload is None:
        return FilePart(name, None, mime_hint, True, False, message_index, undecodable=True)
    cleaned: Final = "".join(payload.split())
    if not cleaned:
        return None
    if size_limit is not None and (len(cleaned) * 3) // 4 - 2 > size_limit:
        return FilePart(name, None, mime_hint, True, True, message_index)
    data: Final = _decoded_base64(cleaned)
    if data is None:
        return FilePart(name, None, mime_hint, True, False, message_index, undecodable=True)
    if size_limit is not None and len(data) > size_limit:
        return FilePart(name, None, mime_hint, True, True, message_index)
    return FilePart(name, data, mime_hint, True, False, message_index)


def _name_from_url(url: str) -> str | None:
    try:
        return unquote(posixpath.basename(urlparse(url).path)) or None
    except ValueError:
        return None


def _part_from_file_block(block: Mapping[str, object], size_limit: int | None, message_index: int) -> FilePart | None:
    file_obj: Final = as_str_mapping(block.get("file"))
    if file_obj is None:
        return None
    name: Final = str_or_none(file_obj.get("filename")) or str_or_none(file_obj.get("file_id"))
    file_data: Final = str_or_none(file_obj.get("file_data"))
    inline: Final = _inline_part(name, file_data, size_limit, message_index) if file_data else None
    return inline or FilePart(name, None, None, False, False, message_index)


def _part_from_image_url_block(
    block: Mapping[str, object], size_limit: int | None, message_index: int
) -> FilePart | None:
    image_url: Final = block.get("image_url")
    image_url_fields: Final = as_str_mapping(image_url)
    url: Final = image_url_fields.get("url") if image_url_fields is not None else image_url
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        return _inline_part(None, url, size_limit, message_index)
    return FilePart(_name_from_url(url), None, None, False, False, message_index)


def _part_from_input_file_block(
    block: Mapping[str, object], size_limit: int | None, message_index: int
) -> FilePart | None:
    declared_name: Final = str_or_none(block.get("filename")) or str_or_none(block.get("file_id"))
    file_data: Final = str_or_none(block.get("file_data"))
    inline: Final = _inline_part(declared_name, file_data, size_limit, message_index) if file_data else None
    if inline is not None:
        return inline
    file_url: Final = block.get("file_url")
    name: Final = (
        _name_from_url(file_url) if isinstance(file_url, str) and file_url and not declared_name else declared_name
    )
    return FilePart(name, None, None, False, False, message_index)


def _part_from_input_audio_block(
    block: Mapping[str, object], size_limit: int | None, message_index: int
) -> FilePart | None:
    audio: Final = as_str_mapping(block.get("input_audio"))
    if audio is None:
        return None
    data_b64: Final = audio.get("data")
    if not isinstance(data_b64, str) or not data_b64:
        return None
    name: Final = f"audio.{audio.get('format') or 'bin'}"
    return _inline_part(name, data_b64, size_limit, message_index) or FilePart(
        name, None, None, False, False, message_index
    )


_BLOCK_PARSERS: Final[Mapping[str, Callable[[Mapping[str, object], int | None, int], FilePart | None]]] = (
    MappingProxyType(
        {
            "file": _part_from_file_block,
            "image_url": _part_from_image_url_block,
            "input_image": _part_from_image_url_block,
            "input_file": _part_from_input_file_block,
            "input_audio": _part_from_input_audio_block,
        }
    )
)


def _file_parts_of_message(
    message: Mapping[str, object], size_limit: int | None, message_index: int
) -> Iterator[FilePart]:
    for block in map(as_str_mapping, _list_items(message.get("content"))):
        if block is None:
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str):
            continue
        parser = _BLOCK_PARSERS.get(block_type)
        if parser is None:
            continue
        try:
            part = parser(block, size_limit, message_index)
        except (TypeError, ValueError, AttributeError, KeyError):
            continue
        if part is not None and (part.inline or part.name):
            yield part


def _file_parts_of_messages(structured_messages: Sequence[object], size_limit: int | None) -> Iterator[FilePart]:
    for message_index, message in enumerate(map(as_str_mapping, structured_messages)):
        if message is not None:
            yield from _file_parts_of_message(message, size_limit, message_index)


def extract_file_parts_from_messages(
    structured_messages: Sequence[object] | None, size_limit: int | None = None
) -> tuple[FilePart, ...]:
    return tuple(_file_parts_of_messages(structured_messages or (), size_limit))


def _file_part_of_image(value: str, size_limit: int | None, index: int) -> FilePart | None:
    if value.startswith(("http://", "https://")):
        name: Final = _name_from_url(value)
        return FilePart(name, None, None, False, False, index) if name else None
    return _inline_part(None, value, size_limit, index)


def extract_file_parts_from_images(
    images: Sequence[object] | None, size_limit: int | None = None
) -> tuple[FilePart, ...]:
    candidates: Final = (
        _file_part_of_image(value, size_limit, index)
        for index, value in enumerate(images or ())
        if isinstance(value, str) and value
    )
    return tuple(part for part in candidates if part is not None)


def make_tool_data(
    name: str, content: str | None, tool_input: Mapping[str, object] | None = None
) -> Mapping[str, object]:
    action_name: Final = str(name) if str(name).strip() else _DEFAULT_TOOL_RESULT_NAME
    truncated: Final = action_name[:_TOOL_NAME_MAX_LENGTH]
    tool_name: Final = truncated if truncated.strip() else _DEFAULT_TOOL_RESULT_NAME
    return {
        "content": content,
        "tool_name": tool_name,
        "action_name": action_name,
        "tool_input": dict(tool_input or ()),
    }


def _tool_call_field(tool_call: object, key: str) -> object:
    fields: Final = as_str_mapping(tool_call)
    return fields.get(key) if fields is not None else getattr(tool_call, key, None)


def _json_or_str(value: object) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _parsed_tool_input(raw_arguments: str) -> Mapping[str, object]:
    if not raw_arguments:
        return _NO_TOOL_INPUT
    try:
        parsed: Final = json.loads(raw_arguments)
    except (ValueError, TypeError):
        return _NO_TOOL_INPUT
    return as_str_mapping(parsed) or _NO_TOOL_INPUT


def _tool_content_and_input(raw_arguments: object) -> tuple[str, Mapping[str, object]]:
    if isinstance(raw_arguments, str):
        return raw_arguments, _parsed_tool_input(raw_arguments)
    if raw_arguments is None:
        return "", _NO_TOOL_INPUT
    arguments: Final = as_str_mapping(raw_arguments)
    return _json_or_str(raw_arguments), arguments if arguments is not None else _NO_TOOL_INPUT


def tool_call_to_tool_data(tool_call: object) -> Mapping[str, object] | None:
    function: Final = _tool_call_field(tool_call, "function")
    name: Final = _tool_call_field(function, "name")
    if not name or not str(name).strip():
        return None
    content, tool_input = _tool_content_and_input(_tool_call_field(function, "arguments"))
    return make_tool_data(str(name), content, tool_input)


def _message_tool_calls(message: object) -> Sequence[object]:
    fields: Final = as_str_mapping(message)
    return _list_items(fields.get("tool_calls")) if fields is not None else ()


def extract_tool_calls_from_messages(structured_messages: Sequence[object] | None) -> tuple[object, ...]:
    """Tool calls declared on the messages themselves.

    Surfaces such as the Anthropic request path populate ``structured_messages`` but leave the
    top-level ``tool_calls`` input empty, so calls made in prior assistant turns are only visible here.
    """
    return tuple(chain.from_iterable(_message_tool_calls(message) for message in structured_messages or ()))


def tool_data_key(tool_data: Mapping[str, object]) -> str:
    """Stable identity for a tool payload, so a call reached from two sources is only scanned once."""
    return json.dumps(tool_data, sort_keys=True, default=str)


def _tool_content_block_text(block: object) -> str | None:
    if isinstance(block, str):
        return block
    fields: Final = as_str_mapping(block)
    return str_or_none(fields.get("text")) if fields is not None else None


def _tool_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    fields: Final = as_str_mapping(content)
    if fields is not None:
        return _json_or_str(fields)
    texts: Final = (_tool_content_block_text(block) for block in _list_items(content))
    return "\n".join(text for text in texts if text is not None)


def _extract_tool_content(content: object) -> str | None:
    text: Final = _tool_content_text(content)
    return text if text.strip() else None


class _Declaration(NamedTuple):
    call_id: str
    message_index: int
    name: str


def _declarations_in(message_index: int, message: object) -> Iterator[_Declaration]:
    fields: Final = as_str_mapping(message)
    if fields is None or fields.get("role") != "assistant":
        return
    for tool_call in map(as_str_mapping, _list_items(fields.get("tool_calls"))):
        if tool_call is None:
            continue
        call_id = tool_call.get("id")
        function = as_str_mapping(tool_call.get("function"))
        name = str_or_none(function.get("name")) if function is not None else None
        if isinstance(call_id, str) and call_id and name and name.strip():
            yield _Declaration(call_id, message_index, name)


def _declarations_by_call_id(messages: Sequence[object]) -> Mapping[str, tuple[_Declaration, ...]]:
    """Every named declaration of each tool call id, in message order, built in one pass."""
    declarations: Final = chain.from_iterable(
        _declarations_in(index, message) for index, message in enumerate(messages)
    )
    ordered: Final = sorted(declarations, key=attrgetter("call_id", "message_index"))
    return MappingProxyType({call_id: tuple(group) for call_id, group in groupby(ordered, key=attrgetter("call_id"))})


def _resolve_tool_name(
    declarations: Mapping[str, tuple[_Declaration, ...]], tool_index: int, tool_call_id: object
) -> str:
    """Name from the nearest declaration of ``tool_call_id`` before ``tool_index``."""
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return _DEFAULT_TOOL_RESULT_NAME
    declared: Final = declarations.get(tool_call_id, ())
    earlier: Final = bisect_left(declared, tool_index, key=attrgetter("message_index"))
    return declared[earlier - 1].name if earlier else _DEFAULT_TOOL_RESULT_NAME


def extract_tool_results(
    structured_messages: Sequence[object] | None,
) -> tuple[tuple[str, str, object], ...]:
    messages: Final = tuple(structured_messages or ())
    declarations: Final = _declarations_by_call_id(messages)

    def _results() -> Iterator[tuple[str, str, object]]:
        for index, message in enumerate(map(as_str_mapping, messages)):
            if message is None or message.get("role") != "tool":
                continue
            content = _extract_tool_content(message.get("content"))
            if content is None:
                continue
            tool_call_id = message.get("tool_call_id")
            yield _resolve_tool_name(declarations, index, tool_call_id), content, tool_call_id

    return tuple(_results())


def _block_texts(content: object) -> Iterator[object]:
    blocks: Final = (as_str_mapping(block) for block in _list_items(content))
    return (block.get("text") for block in blocks if block is not None and block.get("text") is not None)


def _message_text_origins(structured_messages: Sequence[object] | None) -> Iterator[tuple[object, bool]]:
    for message in map(as_str_mapping, structured_messages or ()):
        if message is None:
            continue
        content = message.get("content")
        from_tool_result = message.get("role") == "tool" and _extract_tool_content(content) is not None
        if isinstance(content, str):
            yield content, from_tool_result
        else:
            yield from ((text, from_tool_result) for text in _block_texts(content))


def tool_result_text_indices(structured_messages: Sequence[object] | None, texts: Sequence[str]) -> frozenset[int]:
    """Positions in ``texts`` that hold content already submitted under the TOOL policy.

    The chat-completions guardrail flow builds ``texts`` and ``structured_messages`` from the
    same message list, so tool-role content lands in both and would otherwise be checked twice.
    Other surfaces (e.g. Anthropic messages) build ``texts`` from a differently shaped payload,
    so the mapping is only trusted when replaying it reproduces ``texts`` exactly; anything else
    falls back to checking every text.
    """
    origins: Final = tuple(_message_text_origins(structured_messages))
    if tuple(text for text, _ in origins) != tuple(texts):
        return frozenset()
    return frozenset(index for index, (_, from_tool_result) in enumerate(origins) if from_tool_result)
