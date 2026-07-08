"""Resolve inline file/document attachments in chat messages to Model Armor byte payloads.

Model Armor scans documents through its ``byteItem`` API (PDF, Office docs, CSV, plaintext).
This module walks message content blocks (``type: file`` with inline ``file_data`` and
``type: document`` with an inline base64 ``source``), validates each block into a typed model,
maps its MIME type to a Model Armor ``byteDataType``, and returns the decoded bytes so the
guardrail hooks can submit them.

``plan_file_scans`` classifies each block: blocks with no inline bytes (``file_id`` or remote
``gs://`` / ``http(s)`` references) and supported documents whose base64 will not decode are
reported as unscannable so the guardrail hook can fail closed (blocking unless ``fail_on_error``
is false) rather than letting an unscanned document reach the model.
"""

import base64
import binascii
import mimetypes
from dataclasses import dataclass
from typing import Annotated, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.types.llms.openai import AllMessageValues

MODEL_ARMOR_MAX_FILE_SIZE_BYTES = 4 * 1024 * 1024

# Hard cap on how many attachments a single request may submit to Model Armor, to bound
# per-request fan-out (latency and quota).
MAX_FILE_ATTACHMENTS_PER_REQUEST = 10

_REMOTE_URI_SCHEMES = ("gs://", "http://", "https://")

ModelArmorByteDataType = Literal["PDF", "WORD_DOCUMENT", "EXCEL_DOCUMENT", "POWERPOINT_DOCUMENT", "CSV", "TXT"]

_MIME_TO_BYTE_DATA_TYPE: tuple[tuple[str, ModelArmorByteDataType], ...] = (
    ("application/pdf", "PDF"),
    # Word family: legacy, OOXML, macro-enabled, and templates all map to WORD_DOCUMENT
    ("application/msword", "WORD_DOCUMENT"),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "WORD_DOCUMENT"),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.template", "WORD_DOCUMENT"),
    ("application/vnd.ms-word.document.macroenabled.12", "WORD_DOCUMENT"),
    ("application/vnd.ms-word.template.macroenabled.12", "WORD_DOCUMENT"),
    # Excel family
    ("application/vnd.ms-excel", "EXCEL_DOCUMENT"),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "EXCEL_DOCUMENT"),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.template", "EXCEL_DOCUMENT"),
    ("application/vnd.ms-excel.sheet.macroenabled.12", "EXCEL_DOCUMENT"),
    ("application/vnd.ms-excel.template.macroenabled.12", "EXCEL_DOCUMENT"),
    # PowerPoint family
    ("application/vnd.ms-powerpoint", "POWERPOINT_DOCUMENT"),
    ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "POWERPOINT_DOCUMENT"),
    ("application/vnd.openxmlformats-officedocument.presentationml.template", "POWERPOINT_DOCUMENT"),
    ("application/vnd.openxmlformats-officedocument.presentationml.slideshow", "POWERPOINT_DOCUMENT"),
    ("application/vnd.ms-powerpoint.presentation.macroenabled.12", "POWERPOINT_DOCUMENT"),
    ("application/vnd.ms-powerpoint.template.macroenabled.12", "POWERPOINT_DOCUMENT"),
    ("application/vnd.ms-powerpoint.slideshow.macroenabled.12", "POWERPOINT_DOCUMENT"),
    ("text/csv", "CSV"),
    ("text/plain", "TXT"),
)


@dataclass(frozen=True, slots=True)
class ModelArmorFileAttachment:
    file_bytes: bytes
    byte_data_type: ModelArmorByteDataType


@dataclass(frozen=True, slots=True)
class FileScanPlan:
    # Decoded attachments ready to submit to Model Armor.
    attachments: tuple[ModelArmorFileAttachment, ...]
    # Document/file blocks the guardrail recognized but could not turn into scannable bytes
    # (file_id/remote references, or a supported type whose inline base64 failed to decode).
    unscannable_count: int


class _FileData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_data: str | None = None
    format: str | None = None
    filename: str | None = None


class _FileBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["file"]
    file: _FileData


class _DocumentSource(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: str | None = None
    media_type: str | None = None


class _DocumentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["document"]
    source: _DocumentSource


_AttachmentBlock = Annotated[_FileBlock | _DocumentBlock, Field(discriminator="type")]
_BLOCK_ADAPTER: TypeAdapter[_FileBlock | _DocumentBlock] = TypeAdapter(_AttachmentBlock)


def plan_file_scans(messages: Sequence[AllMessageValues]) -> FileScanPlan:
    """Classify every document/file block into scannable attachments vs unscannable ones.

    Unscannable covers references with no inline bytes and supported documents whose inline
    base64 fails to decode; the hook fails closed on these. Inline content of an unsupported
    type (for example an image) is neither scanned nor counted, it is simply left alone.
    """
    classified = tuple(_classify_block(block) for message in messages for block in _content_blocks(message))
    attachments = tuple(attachment for attachment, _ in classified if attachment is not None)
    unscannable_count = sum(1 for attachment, is_unscannable in classified if attachment is None and is_unscannable)
    return FileScanPlan(attachments=attachments, unscannable_count=unscannable_count)


def _content_blocks(message: AllMessageValues) -> tuple[object, ...]:
    content = message.get("content")
    return tuple(content) if isinstance(content, list) else ()


def _classify_block(block: object) -> tuple[ModelArmorFileAttachment | None, bool]:
    """Return (attachment, is_unscannable). At most one is meaningful; (None, False) means skip."""
    parsed = _parse_block(block)
    if parsed is None:
        return None, False
    if _is_reference(parsed):
        return None, True

    byte_data_type, data = _block_byte_data_type_and_data(parsed)
    if data is None:
        return None, True
    if byte_data_type is None:
        # Recognized inline content of a type Model Armor's byte API does not scan (e.g. an image).
        return None, False

    decoded = _safe_b64decode(data)
    if decoded is None:
        # A supported document whose base64 will not decode cannot be scanned, so fail closed.
        return None, True

    return ModelArmorFileAttachment(file_bytes=decoded, byte_data_type=byte_data_type), False


def _is_reference(block: _FileBlock | _DocumentBlock) -> bool:
    if isinstance(block, _DocumentBlock):
        return not block.source.data
    raw = block.file.file_data
    return not raw or _is_remote_uri(raw)


def _parse_block(block: object) -> _FileBlock | _DocumentBlock | None:
    try:
        return _BLOCK_ADAPTER.validate_python(block)
    except ValidationError:
        return None


def _block_byte_data_type_and_data(
    block: _FileBlock | _DocumentBlock,
) -> tuple[ModelArmorByteDataType | None, str | None]:
    if isinstance(block, _DocumentBlock):
        return _mime_to_byte_data_type(block.source.media_type), block.source.data

    raw = block.file.file_data
    if not raw:
        return None, None
    uri_mime, data = _parse_data_uri(raw)
    if data is None:
        data = raw
    # The data URI header is the least reliable signal: it can be generic (application/octet-stream)
    # or mislabeled (text/plain for a PDF). Prefer the explicit format and filename, falling back to
    # the header only when neither resolves, and warn rather than let a conflicting header downgrade a
    # recognized document to the wrong filter.
    declared = _first_supported_byte_data_type((block.file.format, _mime_from_filename(block.file.filename)))
    header = _mime_to_byte_data_type(uri_mime)
    if declared is None:
        return header, data
    if header is not None and header != declared:
        verbose_proxy_logger.warning(
            "Model Armor: data URI MIME %s maps to %s but the attachment declares %s; scanning as %s",
            uri_mime,
            header,
            declared,
            declared,
        )
    return declared, data


def _first_supported_byte_data_type(
    mimes: tuple[str | None, ...],
) -> ModelArmorByteDataType | None:
    return next(
        (byte_data_type for mime in mimes for byte_data_type in (_mime_to_byte_data_type(mime),) if byte_data_type),
        None,
    )


def _parse_data_uri(raw: str) -> tuple[str | None, str | None]:
    if not raw.startswith("data:") or ";base64," not in raw:
        return None, None
    header, data = raw.split(";base64,", 1)
    return header[len("data:") :] or None, data


def _mime_to_byte_data_type(mime: str | None) -> ModelArmorByteDataType | None:
    if mime is None:
        return None
    normalized = mime.split(";")[0].strip().lower()
    return next(
        (byte_data_type for candidate, byte_data_type in _MIME_TO_BYTE_DATA_TYPE if candidate == normalized), None
    )


def _mime_from_filename(filename: str | None) -> str | None:
    if filename is None:
        return None
    guessed, _ = mimetypes.guess_type(filename)
    return guessed


def _safe_b64decode(data: str) -> bytes | None:
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        verbose_proxy_logger.warning("Model Armor: skipping attachment with undecodable base64 content")
        return None


def _is_remote_uri(raw: str) -> bool:
    return raw.strip().lower().startswith(_REMOTE_URI_SCHEMES)
