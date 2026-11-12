import base64
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, cast  # noqa: TID251  # Existing casts retained unchanged in this pure move

from litellm.litellm_core_utils.core_helpers import as_str_mapping


@dataclass(frozen=True, slots=True)
class _TPMMediaProjection:
    content: object
    media: tuple[Mapping[str, object], ...] = ()


def _tpm_image_part(url: str, detail: object = "auto") -> Mapping[str, object]:
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def _tpm_blank_fields(fields: Mapping[str, object], names: tuple[str, ...]) -> Mapping[str, object]:
    return {key: "" if key in names and isinstance(value, str) else value for key, value in fields.items()}


def _tpm_file_media(part: Mapping[str, object]) -> _TPMMediaProjection:
    from litellm.litellm_core_utils.prompt_templates.factory import parse_mime_type

    nested: Final = part.get("type") == "file"
    fields: Final = as_str_mapping(part.get("file")) if nested else part
    if fields is None:
        return _TPMMediaProjection(part)
    raw_data: Final = fields.get("file_data")
    data: Final = raw_data if isinstance(raw_data, str) else ""
    declared: Final = next(
        (
            value
            for key in ("format", "mime_type", "content_type")
            if isinstance(value := fields.get(key), str) and value
        ),
        "",
    )
    mime: Final = (declared or parse_mime_type(data[:256]) or "").split(";", 1)[0].strip().lower()
    if not mime.startswith(("image/", "audio/")):
        return _TPMMediaProjection(part)
    raw_url: Final = fields.get("file_url") or fields.get("file_id")
    url: Final = data or (raw_url if isinstance(raw_url, str) else "")
    header_end: Final = data.find(",", 0, 256)
    encoded: Final = (
        data[header_end + 1 :]
        if header_end >= 0 and data[:header_end].endswith(";base64")
        else ""
        if data.startswith(("data:", "https://", "http://", "gs://"))
        else data
    )
    media: Final = (
        _tpm_image_part(
            url
            if not data or url.startswith(("data:", "https://", "http://", "gs://"))
            else f"data:{mime};base64,{url}",
            fields.get("detail", "auto"),
        )
        if mime.startswith("image/")
        else {"type": "input_audio", "input_audio": {"data": encoded, "format": mime.removeprefix("audio/")}}
    )
    projected: Final = _tpm_blank_fields(fields, ("file_data", "file_url"))
    return _TPMMediaProjection({**part, "file": projected} if nested else projected, (media,))


def _tpm_media_leaf(value: object) -> _TPMMediaProjection:
    part: Final = as_str_mapping(value)
    if part is None:
        return _TPMMediaProjection(value)
    kind: Final = part.get("type")
    if kind in ("file", "input_file"):
        return _tpm_file_media(part)
    if kind == "input_audio":
        audio: Final = as_str_mapping(part.get("input_audio"))
        return _TPMMediaProjection(
            {**part, "input_audio": _tpm_blank_fields(audio, ("data",))} if audio is not None else value,
            ({"type": "input_audio", "input_audio": part.get("input_audio")},),
        )
    if kind in ("image_url", "input_image", "image", "computer_screenshot"):
        image: Final = as_str_mapping(part.get("image_url"))
        source: Final = as_str_mapping(part.get("source"))
        canonical: Final = _tpm_image_part(
            _image_part_url({**part, "type": "input_image"} if kind == "computer_screenshot" else part) or "",
            (image or part).get("detail", part.get("detail", "auto")),
        )
        projected: Final = (
            {**part, "image_url": _tpm_blank_fields(image, ("url",))}
            if image is not None and kind in ("image_url", "input_image", "computer_screenshot")
            else {**part, "source": _tpm_blank_fields(source, ("data", "url"))}
            if source is not None and kind == "image" and source.get("type") in ("base64", "url")
            else _tpm_blank_fields(part, ("image_url", "url"))
        )
        return _TPMMediaProjection(projected, (canonical,))
    return _TPMMediaProjection(value)


def _tpm_media_list(value: object, split: Callable[[object], _TPMMediaProjection]) -> _TPMMediaProjection:
    if not isinstance(value, (list, tuple)):
        return _TPMMediaProjection(value)
    parts: Final = cast(Sequence[object], value)  # cast-ok: finite JSON content arrays
    projected: Final = tuple(map(split, parts))
    media: Final = tuple(chain.from_iterable(part.media for part in projected))
    content: Final = tuple(part.content for part in projected)
    return _TPMMediaProjection(parts if not media else content if isinstance(value, tuple) else list(content), media)


def _tpm_replace_field(part: Mapping[str, object], key: str, projection: _TPMMediaProjection) -> _TPMMediaProjection:
    return _TPMMediaProjection({**part, key: projection.content} if projection.media else part, projection.media)


def _tpm_document_media(value: object) -> _TPMMediaProjection:
    part: Final = as_str_mapping(value)
    source: Final = as_str_mapping(part.get("source")) if part is not None else None
    if part is None or part.get("type") != "document" or source is None or source.get("type") != "content":
        return _tpm_media_leaf(value)
    content: Final = _tpm_media_list(source.get("content"), _tpm_media_leaf)
    return _tpm_replace_field(part, "source", _tpm_replace_field(source, "content", content))


def _tpm_tool_result_media(value: object) -> _TPMMediaProjection:
    part: Final = as_str_mapping(value)
    if part is None or part.get("type") != "tool_result":
        return _tpm_document_media(value)
    return _tpm_replace_field(part, "content", _tpm_media_list(part.get("content"), _tpm_document_media))


def _tpm_code_output_media(value: object) -> _TPMMediaProjection:
    part: Final = as_str_mapping(value)
    url: Final = part.get("url") if part is not None else None
    if part is None or part.get("type") != "image" or not isinstance(url, str) or not url:
        return _TPMMediaProjection(value)
    return _TPMMediaProjection(_tpm_blank_fields(part, ("url",)), (_tpm_image_part(url),))


def _tpm_input_media(value: object) -> _TPMMediaProjection:
    part: Final = as_str_mapping(value)
    if part is None:
        return _TPMMediaProjection(value)
    kind: Final = part.get("type")
    if kind in ("function_call_output", "custom_tool_call_output"):
        return _tpm_replace_field(part, "output", _tpm_media_list(part.get("output"), _tpm_tool_result_media))
    if kind == "computer_call_output":
        return _tpm_replace_field(part, "output", _tpm_media_leaf(part.get("output")))
    if kind == "code_interpreter_call":
        return _tpm_replace_field(part, "outputs", _tpm_media_list(part.get("outputs"), _tpm_code_output_media))
    result: Final = part.get("result")
    if kind == "image_generation_call" and isinstance(result, str) and result:
        return _TPMMediaProjection(
            _tpm_blank_fields(part, ("result",)),
            (_tpm_image_part("data:application/octet-stream;base64," + result),),
        )
    if kind == "message" or (kind is None and isinstance(part.get("role"), str)):
        return _tpm_replace_field(part, "content", _tpm_media_list(part.get("content"), _tpm_tool_result_media))
    return _tpm_tool_result_media(value)


def split_media_for_tpm(value: object) -> tuple[object, tuple[Mapping[str, object], ...]]:
    projection: Final = _tpm_media_list(value, _tpm_input_media)
    return projection.content, projection.media


_TPM_EMBEDDING_FILE_MIME_PREFIXES: Final = ("application/pdf", "video/")


def _tpm_embedding_file(data_uri: str) -> tuple[Mapping[str, object], ...]:
    from litellm.litellm_core_utils.prompt_templates.factory import parse_mime_type

    mime: Final = (parse_mime_type(data_uri[:256]) or "").strip().lower()
    if not mime.startswith(_TPM_EMBEDDING_FILE_MIME_PREFIXES):
        return ()
    return ({"type": "file", "file": {"file_data": data_uri, "format": mime}},)


_BASE64_PAYLOAD: Final = re.compile(r"[A-Za-z0-9+/]*={0,2}")


def _is_base64_data_uri(value: str) -> bool:
    header_end: Final = value.find(",", 0, 256)
    if header_end < 0 or not value[:header_end].endswith(";base64"):
        return False
    payload_start: Final = header_end + 1
    if (len(value) - payload_start) % 4 or _BASE64_PAYLOAD.fullmatch(value, payload_start) is None:
        return False
    last_quantum: Final = value[max(payload_start, len(value) - 4) :]
    return base64.b64encode(base64.b64decode(last_quantum)).decode("ascii") == last_quantum


def _tpm_embedding_file_block(part: Mapping[str, object]) -> _TPMMediaProjection:
    projection: Final = _tpm_file_media(part)
    fields: Final = as_str_mapping(part.get("file"))
    data: Final = fields.get("file_data") if fields is not None else None
    if projection.media or fields is None or not isinstance(data, str) or not _is_base64_data_uri(data):
        return projection
    media: Final = _tpm_embedding_file(data)
    return _TPMMediaProjection({**part, "file": _tpm_blank_fields(fields, ("file_data",))} if media else part, media)


def _tpm_embedding_leaf(value: object) -> _TPMMediaProjection:
    part: Final = as_str_mapping(value)
    if part is not None and part.get("type") == "file":
        return _tpm_embedding_file_block(part)
    if not isinstance(value, str) or is_plain_embedding_text(value):
        return _tpm_media_leaf(value)
    if not _is_base64_data_uri(value):
        return _TPMMediaProjection(value)
    media: Final = _tpm_file_media({"file_data": value}).media or _tpm_embedding_file(value)
    return _TPMMediaProjection("" if media else value, media)


def is_plain_embedding_text(value: object) -> bool:
    return isinstance(value, str) and not value.startswith("data:")


def _is_json_array(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _tpm_embedding_item(value: object) -> _TPMMediaProjection:
    return _tpm_media_list(value, _tpm_embedding_leaf) if _is_json_array(value) else _tpm_embedding_leaf(value)


def split_embedding_media_for_tpm(value: object) -> tuple[object, tuple[Mapping[str, object], ...]]:
    projection: Final = (
        _tpm_media_list(value, _tpm_embedding_item) if _is_json_array(value) else _tpm_embedding_leaf(value)
    )
    return projection.content, projection.media


def embedding_input_is_all_media(value: object) -> bool:
    items: Final = cast(Sequence[object], value) if _is_json_array(value) else (value,)  # cast-ok: JSON input array
    return all(_tpm_embedding_leaf(item).media for item in items)


def anthropic_image_source_to_openai_url(image_source: Mapping[str, object]) -> str | None:
    """Data or remote URL for an Anthropic ``source`` block, in the form chat completions expects."""
    source_type: Final = image_source.get("type")
    if source_type == "base64":
        media_type: Final = image_source.get("media_type") or "image/jpeg"
        image_data: Final = image_source.get("data") or ""
        return f"data:{media_type};base64,{image_data}" if image_data else None
    if source_type == "url":
        url: Final = image_source.get("url")
        return url if isinstance(url, str) else ""
    return None


def _image_part_url(part: Mapping[str, object]) -> str | None:
    """The image URL carried by one content part, whichever of the three dialects wrote it."""
    part_type: Final = part.get("type")
    if part_type in ("image_url", "input_image"):
        image_url: Final = part.get("image_url") or part.get("url")
        if isinstance(image_url, str):
            return image_url
        image_fields: Final = as_str_mapping(image_url)
        url: Final = image_fields.get("url") if image_fields is not None else None
        return url if isinstance(url, str) else None
    if part_type == "image":
        source: Final = part.get("source")
        return anthropic_image_source_to_openai_url(source) if isinstance(source, Mapping) else None
    return None
