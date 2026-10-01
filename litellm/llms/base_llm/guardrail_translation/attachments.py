from __future__ import annotations

import base64
import binascii
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, cast
from urllib.parse import unquote

TEXT_MEDIA_TYPE_PREFIX: Final = "text/"
UNSCANNABLE_MEDIA_PART_TYPES: Final = frozenset({"input_audio", "audio_url", "video_url", "container_upload"})
DOCUMENT_PART_TYPES: Final = frozenset({"file", "document", "input_file"})
NESTED_CONTENT_KEYS: Final = ("content", "output")
ATTACHMENT_SCAN_MAX_DEPTH: Final = 8


@dataclass(frozen=True, slots=True)
class AttachmentText:
    part_type: str
    text: str


@dataclass(frozen=True, slots=True)
class RequestAttachments:
    texts: tuple[AttachmentText, ...]
    unscannable: tuple[str, ...]


NO_ATTACHMENTS: Final = RequestAttachments(texts=(), unscannable=())


def as_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast("Mapping[str, object]", value)  # cast-ok: isinstance narrows only to Mapping[Unknown, Unknown]


def _sequence(value: object) -> Sequence[object] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return cast("Sequence[object]", value)  # cast-ok: isinstance narrows only to Sequence[Unknown]


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _decoded_base64_text(payload: str) -> str | None:
    try:
        return base64.b64decode(payload).decode("utf-8", errors="replace")
    except (binascii.Error, ValueError):
        return None


def _text_data_url_content(value: object) -> str | None:
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    header, separator, payload = value.partition(",")
    if not separator:
        return None
    media_type: Final = header[len("data:") :].split(";", 1)[0].strip().lower()
    if not media_type.startswith(TEXT_MEDIA_TYPE_PREFIX):
        return None
    if ";base64" not in header:
        return unquote(payload)
    return _decoded_base64_text(payload)


def _block_text(block: object) -> str | None:
    block_mapping: Final = as_mapping(block)
    return None if block_mapping is None else _string(block_mapping.get("text"))


def _text_blocks(content: object) -> str | None:
    if isinstance(content, str):
        return content
    blocks: Final = _sequence(content)
    if blocks is None:
        return None
    texts: Final = tuple(text for text in map(_block_text, blocks) if text is not None)
    return "\n".join(texts)


def _document_source_text(source: Mapping[str, object] | None) -> str | None:
    if source is None:
        return None
    source_type: Final = source.get("type")
    data: Final = source.get("data")
    if source_type == "text":
        return _string(data)
    if source_type == "content":
        return _text_blocks(source.get("content"))
    if source_type != "base64" or not isinstance(data, str):
        return None
    media_type: Final = _string(source.get("media_type"))
    if media_type is None or not media_type.lower().startswith(TEXT_MEDIA_TYPE_PREFIX):
        return None
    return _decoded_base64_text(data)


def _document_part_text(part: Mapping[str, object], part_type: str) -> str | None:
    if part_type == "file":
        file: Final = as_mapping(part.get("file"))
        return None if file is None else _text_data_url_content(file.get("file_data"))
    if part_type == "input_file":
        return _text_data_url_content(part.get("file_data"))
    return _document_source_text(as_mapping(part.get("source")))


def _attachment_of(part: Mapping[str, object]) -> AttachmentText | str | None:
    part_type: Final = _string(part.get("type"))
    if part_type is None:
        return None
    if part_type in UNSCANNABLE_MEDIA_PART_TYPES:
        return part_type
    if part_type not in DOCUMENT_PART_TYPES:
        return None
    text: Final = _document_part_text(part, part_type)
    return AttachmentText(part_type=part_type, text=text) if text is not None else part_type


def _children(node: object) -> tuple[object, ...]:
    node_mapping: Final = as_mapping(node)
    if node_mapping is not None:
        return tuple(node_mapping.get(key) for key in NESTED_CONTENT_KEYS if key in node_mapping)
    node_sequence: Final = _sequence(node)
    return () if node_sequence is None else tuple(node_sequence)


def _iter_parts(root: object) -> Iterator[Mapping[str, object]]:
    frontier: tuple[object, ...] = (root,)  # rebind-ok: depth-bounded frontier walk
    for _ in range(ATTACHMENT_SCAN_MAX_DEPTH):
        if not frontier:
            return
        yield from (part for part in map(as_mapping, frontier) if part is not None)
        frontier = tuple(chain.from_iterable(map(_children, frontier)))


def content_attachments(content: object) -> RequestAttachments:
    found: Final = tuple(
        attachment for attachment in map(_attachment_of, _iter_parts(content)) if attachment is not None
    )
    return RequestAttachments(
        texts=tuple(attachment for attachment in found if isinstance(attachment, AttachmentText)),
        unscannable=tuple(dict.fromkeys(attachment for attachment in found if isinstance(attachment, str))),
    )


def request_attachments(request_data: Mapping[str, object]) -> RequestAttachments:
    return content_attachments((request_data.get("messages"), request_data.get("input")))
