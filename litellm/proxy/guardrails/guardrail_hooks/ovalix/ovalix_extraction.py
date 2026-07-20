import base64
import json
import posixpath
import re
from collections.abc import Callable
from typing import Any, NamedTuple
from urllib.parse import unquote, urlparse

_TOOL_NAME_MAX_LENGTH = 100
_DEFAULT_TOOL_RESULT_NAME = "tool_result"

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


def _part_from_file_block(block: dict[str, Any], size_limit: int | None, message_index: int) -> FilePart | None:
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


def _part_from_image_url_block(block: dict[str, Any], size_limit: int | None, message_index: int) -> FilePart | None:
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


def _part_from_input_file_block(block: dict[str, Any], size_limit: int | None, message_index: int) -> FilePart | None:
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


def _part_from_input_audio_block(block: dict[str, Any], size_limit: int | None, message_index: int) -> FilePart | None:
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


_BLOCK_PARSERS: dict[str, Callable[[dict[str, Any], int | None, int], FilePart | None]] = {
    "file": _part_from_file_block,
    "image_url": _part_from_image_url_block,
    "input_image": _part_from_image_url_block,
    "input_file": _part_from_input_file_block,
    "input_audio": _part_from_input_audio_block,
}


def extract_file_parts_from_messages(
    structured_messages: list[dict[str, Any]] | None, size_limit: int | None = None
) -> list[FilePart]:
    parts: list[FilePart] = []
    for message_index, message in enumerate(structured_messages or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
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
                parts.append(part)
    return parts


def extract_file_parts_from_images(images: list[str] | None, size_limit: int | None = None) -> list[FilePart]:
    parts: list[FilePart] = []
    for index, value in enumerate(images or []):
        if not isinstance(value, str) or not value:
            continue
        if value.startswith(("http://", "https://")):
            name = _name_from_url(value)
            if name:
                parts.append(FilePart(name, None, None, False, False, index))
            continue
        mime_hint, payload = _split_data_url(value)
        data, oversize = _decode_base64_with_limit(payload, size_limit) if payload else (None, False)
        if data is not None or oversize:
            parts.append(FilePart(None, data, mime_hint, True, oversize, index))
    return parts


def make_tool_data(name: str, content: str | None, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    action_name = str(name) if str(name).strip() else _DEFAULT_TOOL_RESULT_NAME
    tool_name = action_name[:_TOOL_NAME_MAX_LENGTH]
    if not tool_name.strip():
        tool_name = _DEFAULT_TOOL_RESULT_NAME
    return {"content": content, "tool_name": tool_name, "action_name": action_name, "tool_input": tool_input or {}}


def _tool_call_field(tool_call: Any, key: str) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def tool_call_to_tool_data(tool_call: Any) -> dict[str, Any] | None:
    function = _tool_call_field(tool_call, "function")
    name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
    if not name or not str(name).strip():
        return None
    raw_arguments = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
    tool_input: dict[str, Any] = {}
    if isinstance(raw_arguments, str):
        content = raw_arguments
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    tool_input = parsed
            except (ValueError, TypeError):
                tool_input = {}
    elif raw_arguments is None:
        content = ""
    elif isinstance(raw_arguments, dict):
        try:
            content = json.dumps(raw_arguments)
        except (TypeError, ValueError):
            content = str(raw_arguments)
        tool_input = raw_arguments
    else:
        try:
            content = json.dumps(raw_arguments)
        except (TypeError, ValueError):
            content = str(raw_arguments)
    return make_tool_data(name, content, tool_input)


def _extract_tool_content(content: Any) -> str | None:
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    elif isinstance(content, dict):
        try:
            content = json.dumps(content)
        except (TypeError, ValueError):
            content = str(content)
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def extract_tool_results(structured_messages: list[dict[str, Any]] | None) -> list[tuple[str, str, str | None]]:
    id_to_name: dict[str, str] = {}
    results: list[tuple[str, str, str | None]] = []
    for message in structured_messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = tool_call.get("id")
                function = tool_call.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if isinstance(call_id, str) and call_id and name and str(name).strip():
                    id_to_name[call_id] = name
        elif role == "tool":
            content = _extract_tool_content(message.get("content"))
            if content is None:
                continue
            tool_call_id = message.get("tool_call_id")
            resolved_name = id_to_name.get(tool_call_id) if isinstance(tool_call_id, str) else None
            name = resolved_name or _DEFAULT_TOOL_RESULT_NAME
            results.append((name, content, tool_call_id))
    return results
