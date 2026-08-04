import base64
import json
import posixpath
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import NamedTuple
from urllib.parse import unquote, urlparse

_TOOL_NAME_MAX_LENGTH = 100
_DEFAULT_TOOL_RESULT_NAME = "tool_result"
_NO_TOOL_INPUT: Mapping[str, object] = MappingProxyType({})

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<params>(?:;[^;,]+)*?)(?P<b64>;base64)?,", re.IGNORECASE)
_URLSAFE_TO_STANDARD_B64 = str.maketrans("-_", "+/")


class FilePart(NamedTuple):
    name: str | None
    data: bytes | None
    mime_hint: str | None
    inline: bool
    oversize: bool
    message_index: int


def _split_data_url(value: str) -> tuple[str | None, str | None]:
    match = _DATA_URL_RE.match(value)
    if not match:
        if value.lower().startswith("data:"):
            return None, None
        return None, value
    mime = match.group("mime") or None
    if not match.group("b64"):
        return mime, None
    return mime, value[match.end() :]


def _decode_base64_with_limit(b64_payload: str, size_limit: int | None) -> tuple[bytes | None, bool]:
    cleaned = "".join(b64_payload.split())
    if not cleaned:
        return None, False
    if size_limit is not None and (len(cleaned) * 3) // 4 - 2 > size_limit:
        return None, True
    data = None
    try:
        data = base64.b64decode(cleaned, validate=True)
    except ValueError:
        if "-" in cleaned or "_" in cleaned:
            try:
                data = base64.b64decode(cleaned.translate(_URLSAFE_TO_STANDARD_B64), validate=True)
            except ValueError:
                return None, False
        else:
            return None, False
    if size_limit is not None and len(data) > size_limit:
        return None, True
    return (data, False) if data else (None, False)


def _name_from_url(url: str) -> str | None:
    try:
        return unquote(posixpath.basename(urlparse(url).path)) or None
    except ValueError:
        return None


def _part_from_file_block(block: Mapping[str, object], size_limit: int | None, message_index: int) -> FilePart | None:
    file_obj = block.get("file")
    if not isinstance(file_obj, dict):
        return None
    name = file_obj.get("filename") or file_obj.get("file_id") or None
    file_data = file_obj.get("file_data")
    if isinstance(file_data, str) and file_data:
        mime_hint, payload = _split_data_url(file_data)
        data, oversize = _decode_base64_with_limit(payload, size_limit) if payload else (None, False)
        if data is not None or oversize:
            return FilePart(name, data, mime_hint, True, oversize, message_index)
    return FilePart(name, None, None, False, False, message_index)


def _part_from_image_url_block(
    block: Mapping[str, object], size_limit: int | None, message_index: int
) -> FilePart | None:
    image_url = block.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        mime_hint, payload = _split_data_url(url)
        data, oversize = _decode_base64_with_limit(payload, size_limit) if payload else (None, False)
        if data is not None or oversize:
            return FilePart(None, data, mime_hint, True, oversize, message_index)
        return None
    return FilePart(_name_from_url(url), None, None, False, False, message_index)


def _part_from_input_file_block(
    block: Mapping[str, object], size_limit: int | None, message_index: int
) -> FilePart | None:
    name = block.get("filename") or block.get("file_id") or None
    file_data = block.get("file_data")
    if isinstance(file_data, str) and file_data:
        mime_hint, payload = _split_data_url(file_data)
        data, oversize = _decode_base64_with_limit(payload, size_limit) if payload else (None, False)
        if data is not None or oversize:
            return FilePart(name, data, mime_hint, True, oversize, message_index)
    file_url = block.get("file_url")
    if isinstance(file_url, str) and file_url and not name:
        name = _name_from_url(file_url)
    return FilePart(name, None, None, False, False, message_index)


def _part_from_input_audio_block(
    block: Mapping[str, object], size_limit: int | None, message_index: int
) -> FilePart | None:
    audio = block.get("input_audio")
    if not isinstance(audio, dict):
        return None
    data_b64 = audio.get("data")
    if not isinstance(data_b64, str) or not data_b64:
        return None
    name = f"audio.{audio.get('format') or 'bin'}"
    data, oversize = _decode_base64_with_limit(data_b64, size_limit)
    if data is None and not oversize:
        return FilePart(name, None, None, False, False, message_index)
    return FilePart(name, data, None, True, oversize, message_index)


_BLOCK_PARSERS: Mapping[str, Callable[[Mapping[str, object], int | None, int], FilePart | None]] = MappingProxyType(
    {
        "file": _part_from_file_block,
        "image_url": _part_from_image_url_block,
        "input_image": _part_from_image_url_block,
        "input_file": _part_from_input_file_block,
        "input_audio": _part_from_input_audio_block,
    }
)


def _file_parts_of_message(
    message: Mapping[str, object], size_limit: int | None, message_index: int
) -> Iterator[FilePart]:
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, Mapping):
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


def extract_file_parts_from_messages(
    structured_messages: Sequence[Mapping[str, object]] | None, size_limit: int | None = None
) -> tuple[FilePart, ...]:
    return tuple(
        part
        for message_index, message in enumerate(structured_messages or ())
        if isinstance(message, Mapping)
        for part in _file_parts_of_message(message, size_limit, message_index)
    )


def _file_part_of_image(value: str, size_limit: int | None, index: int) -> FilePart | None:
    if value.startswith(("http://", "https://")):
        name = _name_from_url(value)
        return FilePart(name, None, None, False, False, index) if name else None
    mime_hint, payload = _split_data_url(value)
    data, oversize = _decode_base64_with_limit(payload, size_limit) if payload else (None, False)
    if data is None and not oversize:
        return None
    return FilePart(None, data, mime_hint, True, oversize, index)


def extract_file_parts_from_images(images: Sequence[str] | None, size_limit: int | None = None) -> tuple[FilePart, ...]:
    candidates = (
        _file_part_of_image(value, size_limit, index)
        for index, value in enumerate(images or ())
        if isinstance(value, str) and value
    )
    return tuple(part for part in candidates if part is not None)


def make_tool_data(
    name: str, content: str | None, tool_input: Mapping[str, object] | None = None
) -> Mapping[str, object]:
    action_name = str(name) if str(name).strip() else _DEFAULT_TOOL_RESULT_NAME
    tool_name = action_name[:_TOOL_NAME_MAX_LENGTH]
    if not tool_name.strip():
        tool_name = _DEFAULT_TOOL_RESULT_NAME
    return {
        "content": content,
        "tool_name": tool_name,
        "action_name": action_name,
        "tool_input": dict(tool_input or ()),
    }


def _tool_call_field(tool_call: object, key: str) -> object:
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _json_or_str(value: object) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _parsed_tool_input(raw_arguments: str) -> Mapping[str, object]:
    if not raw_arguments:
        return _NO_TOOL_INPUT
    try:
        parsed = json.loads(raw_arguments)
    except (ValueError, TypeError):
        return _NO_TOOL_INPUT
    return parsed if isinstance(parsed, dict) else _NO_TOOL_INPUT


def _tool_content_and_input(raw_arguments: object) -> tuple[str, Mapping[str, object]]:
    if isinstance(raw_arguments, str):
        return raw_arguments, _parsed_tool_input(raw_arguments)
    if raw_arguments is None:
        return "", _NO_TOOL_INPUT
    if isinstance(raw_arguments, dict):
        return _json_or_str(raw_arguments), raw_arguments
    return _json_or_str(raw_arguments), _NO_TOOL_INPUT


def tool_call_to_tool_data(tool_call: object) -> Mapping[str, object] | None:
    function = _tool_call_field(tool_call, "function")
    name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
    if not name or not str(name).strip():
        return None
    raw_arguments = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
    content, tool_input = _tool_content_and_input(raw_arguments)
    return make_tool_data(name, content, tool_input)


def _tool_content_blocks(content: Sequence[object]) -> Iterator[str]:
    for block in content:
        if isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str) and text:
                yield text
        elif isinstance(block, str):
            yield block


def _extract_tool_content(content: object) -> str | None:
    if isinstance(content, list):
        content = "\n".join(_tool_content_blocks(content))
    elif isinstance(content, dict):
        content = _json_or_str(content)
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _declared_names_for_call(message: Mapping[str, object], call_id: str) -> Iterator[str]:
    for tool_call in message.get("tool_calls") or ():
        if not isinstance(tool_call, Mapping) or tool_call.get("id") != call_id:
            continue
        function = tool_call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if name and str(name).strip():
            yield name


def _resolve_tool_name(messages: Sequence[Mapping[str, object]], tool_index: int, tool_call_id: object) -> str:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return _DEFAULT_TOOL_RESULT_NAME
    declared = tuple(
        name
        for message in messages[:tool_index]
        if isinstance(message, Mapping) and message.get("role") == "assistant"
        for name in _declared_names_for_call(message, tool_call_id)
    )
    return declared[-1] if declared else _DEFAULT_TOOL_RESULT_NAME


def extract_tool_results(
    structured_messages: Sequence[Mapping[str, object]] | None,
) -> tuple[tuple[str, str, str | None], ...]:
    messages = tuple(structured_messages or ())

    def _results() -> Iterator[tuple[str, str, str | None]]:
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping) or message.get("role") != "tool":
                continue
            content = _extract_tool_content(message.get("content"))
            if content is None:
                continue
            tool_call_id = message.get("tool_call_id")
            yield _resolve_tool_name(messages, index, tool_call_id), content, tool_call_id

    return tuple(_results())


def _message_text_origins(structured_messages: Sequence[Mapping[str, object]] | None) -> Iterator[tuple[str, bool]]:
    for message in structured_messages or ():
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        from_tool_result = message.get("role") == "tool" and _extract_tool_content(content) is not None
        if isinstance(content, str):
            yield content, from_tool_result
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and block.get("text") is not None:
                    yield block["text"], from_tool_result


def tool_result_text_indices(
    structured_messages: Sequence[Mapping[str, object]] | None, texts: Sequence[str]
) -> frozenset[int]:
    """Positions in ``texts`` that hold content already submitted under the TOOL policy.

    The chat-completions guardrail flow builds ``texts`` and ``structured_messages`` from the
    same message list, so tool-role content lands in both and would otherwise be checked twice.
    Other surfaces (e.g. Anthropic messages) build ``texts`` from a differently shaped payload,
    so the mapping is only trusted when replaying it reproduces ``texts`` exactly; anything else
    falls back to checking every text.
    """
    origins = tuple(_message_text_origins(structured_messages))
    if tuple(text for text, _ in origins) != tuple(texts):
        return frozenset()
    return frozenset(index for index, (_, from_tool_result) in enumerate(origins) if from_tool_result)
