"""Attachment blocks sent to Akto's file guardrail, including those inside ``tool_result`` blocks:

  OpenAI chat     ``image_url``, ``input_audio``, ``file``, ``video_url``
  Anthropic       ``image``, ``document`` (except text documents, which stay in the text check)
  Responses API   ``input_image``, ``input_file``

A block with neither inline bytes nor a URL (an OpenAI ``file_id``) is unsendable.
"""

import base64
import binascii
import mimetypes
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias, TypeVar
from urllib.parse import unquote, unquote_to_bytes, urlparse

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, ValidationError

AttachmentType: TypeAlias = Literal["image", "audio", "file"]

_REMOTE_URI_SCHEMES: Final = ("http://", "https://")
_URL_SAFE_TO_STANDARD: Final = str.maketrans("-_", "+/")
# Per attachment type, the fields dropped from the text check because they hold bytes, URLs or file references
_FILE_CHECKED_FIELDS: Final = MappingProxyType(
    {
        "image_url": frozenset(("image_url", "url")),
        "input_image": frozenset(("image_url", "url", "file_id")),
        "input_audio": frozenset(("input_audio",)),
        "video_url": frozenset(("video_url",)),
        "file": frozenset(("file",)),
        "input_file": frozenset(("file_data", "file_url", "file_id")),
        "image": frozenset(("source",)),
        "document": frozenset(("source",)),
    }
)
_TEXT_SOURCE_TYPES: Final = frozenset(("text", "content"))
_FILE_SOURCE_FIELDS: Final = frozenset(("file_data", "file_id"))
_ATTACHMENT_BLOCK_TYPES: Final = frozenset(_FILE_CHECKED_FIELDS)
_OBJECT_MAPPING: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])

_T: Final = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Attachment:
    filename: str
    type: AttachmentType
    content: str | None = None
    url: str | None = None

    def as_payload(self) -> Mapping[str, str]:
        fields: Final = (("filename", self.filename), ("type", self.type), ("content", self.content), ("url", self.url))
        return MappingProxyType({key: value for key, value in fields if value is not None})


@dataclass(frozen=True, slots=True)
class RequestAttachments:
    attachments: tuple[Attachment, ...]
    unsendable_count: int
    malformed_count: int = 0


def _text_or_none(value: object) -> object:
    return value if isinstance(value, str) else None


# Optional metadata the provider ignores when malformed, so a bad value must not fail the whole block
_Metadata: TypeAlias = Annotated[str | None, BeforeValidator(_text_or_none)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _ImageURL(_Model):
    url: str | None = None


class _ImageURLBlock(_Model):
    type: Literal["image_url"]
    image_url: _ImageURL | str | None = None
    url: _ImageURL | str | None = None


class _VideoURLBlock(_Model):
    type: Literal["video_url"]
    video_url: _ImageURL | str


class _InputImageBlock(_Model):
    type: Literal["input_image"]
    image_url: _ImageURL | str | None = None
    url: _ImageURL | str | None = None
    file_id: str | None = None


class _InputAudio(_Model):
    data: str | None = None
    format: _Metadata = None


class _InputAudioBlock(_Model):
    type: Literal["input_audio"]
    input_audio: _InputAudio


class _FileData(_Model):
    file_data: str | None = None
    file_id: str | None = None
    filename: _Metadata = None


class _FileBlock(_Model):
    type: Literal["file"]
    file: _FileData


class _InputFileBlock(_Model):
    type: Literal["input_file"]
    file_data: str | None = None
    file_url: str | None = None
    file_id: str | None = None
    filename: _Metadata = None


class _Source(_Model):
    type: _Metadata = None
    data: str | None = None
    media_type: _Metadata = None
    url: str | None = None
    content: object = None


class _ImageBlock(_Model):
    type: Literal["image"]
    source: _Source


class _DocumentBlock(_Model):
    type: Literal["document"]
    source: _Source
    title: _Metadata = None


class _ToolResultBlock(_Model):
    type: Literal["tool_result"]
    content: object = None


class _MalformedBlock(_Model):
    """An attachment type that doesn't parse; it can't be checked, so it blocks."""


class _Message(_Model):
    content: object = None
    output: object = None


_AttachmentBlock: TypeAlias = (
    _ImageURLBlock
    | _VideoURLBlock
    | _InputImageBlock
    | _InputAudioBlock
    | _FileBlock
    | _InputFileBlock
    | _ImageBlock
    | _DocumentBlock
    | _ToolResultBlock
)
_BLOCK_ADAPTER: Final[TypeAdapter[_AttachmentBlock]] = TypeAdapter(
    Annotated[_AttachmentBlock, Field(discriminator="type")]
)
_Block: TypeAlias = _AttachmentBlock | _MalformedBlock
_MESSAGE_ADAPTER: Final[TypeAdapter[_Message]] = TypeAdapter(_Message)
_ITEMS_ADAPTER: Final[TypeAdapter[list[object]]] = TypeAdapter(list[object])

# (attachment, is_unsendable); (None, False) is a block that isn't an attachment
_Classified: TypeAlias = tuple[Attachment | None, bool]
_NOT_AN_ATTACHMENT: Final[_Classified] = (None, False)
_UNSENDABLE: Final[_Classified] = (None, True)


def request_attachments(request_data: Mapping[str, object]) -> RequestAttachments:
    # Both, so a decoy "messages" can't hide attachments in a Responses API "input"
    messages: Final = request_data.get("messages")
    responses_input: Final = request_data.get("input")
    sources: Final = (messages,) if responses_input is messages else (messages, responses_input)
    containers: Final = (_parse(_ITEMS_ADAPTER, source) or () for source in sources)
    blocks: Final = tuple(chain.from_iterable(_message_blocks(message) for message in chain.from_iterable(containers)))
    classified: Final = tuple(
        chain.from_iterable(_block_attachments(block, index) for index, block in enumerate(blocks))
    )
    return RequestAttachments(
        attachments=tuple(attachment for attachment, _ in classified if attachment is not None),
        unsendable_count=sum(1 for _, is_unsendable in classified if is_unsendable),
        malformed_count=sum(1 for block in blocks if isinstance(block, _MalformedBlock)),
    )


def _message_blocks(message: object) -> tuple[_Block, ...]:
    parsed: Final = _parse(_MESSAGE_ADAPTER, message)
    top: Final = (_blocks(parsed.content) + _blocks(parsed.output)) if parsed else ()
    nested: Final = _nested_blocks(top)
    # tool_result -> document -> image is the deepest the APIs nest
    return top + nested + _nested_blocks(nested)


def _nested_blocks(blocks: tuple[_Block, ...]) -> tuple[_Block, ...]:
    return tuple(chain.from_iterable(_blocks(_nested_content(block)) for block in blocks))


def _nested_content(block: _Block) -> object:
    match block:
        case _ToolResultBlock():
            return block.content
        case _DocumentBlock(source=_Source(type="content")):
            return block.source.content
        case _:
            return None


def _blocks(content: object) -> tuple[_Block, ...]:
    items: Final = _parse(_ITEMS_ADAPTER, content)
    parsed: Final = (_block(item) for item in items or ())
    return tuple(block for block in parsed if block is not None)


def _block(item: object) -> _Block | None:
    parsed: Final = _parse(_BLOCK_ADAPTER, item)
    if parsed is not None:
        return parsed
    block_type: Final = (_parse(_OBJECT_MAPPING, item) or {}).get("type")
    return _MalformedBlock() if isinstance(block_type, str) and block_type in _ATTACHMENT_BLOCK_TYPES else None


def _block_attachments(block: _Block, index: int) -> tuple[_Classified, ...]:
    """A file block can name several sources and providers differ on which they send, so all are checked."""
    match block:
        case _FileBlock():
            return _file_sources((block.file.file_data,), block.file.file_id, block.file.filename, index)
        case _InputFileBlock():
            return _file_sources((block.file_data, block.file_url), block.file_id, block.filename, index)
        case _ImageURLBlock():
            return _file_sources((_url(block.image_url), _url(block.url)), None, None, index, "image")
        case _InputImageBlock():
            return _file_sources((_url(block.image_url), _url(block.url)), block.file_id, None, index, "image")
        case _DocumentBlock():
            return (_from_source(block.source, block.title, index, "file"),)
        case _:
            return (_classify_block(block, index),)


def _file_sources(
    inline: tuple[str | None, ...], file_id: str | None, name: str | None, index: int, kind: AttachmentType = "file"
) -> tuple[_Classified, ...]:
    found: Final = (
        *(_from_uri(source, name, index, kind) for source in inline if source),
        *((_from_file_id(file_id, name, index, kind),) if file_id else ()),
    )
    return found or (_UNSENDABLE,)


def _from_file_id(file_id: str, name: str | None, index: int, kind: AttachmentType) -> _Classified:
    """A URL is checked; an uploaded file's id has no content to send."""
    is_url: Final = file_id.strip().lower().startswith(_REMOTE_URI_SCHEMES)
    return _from_uri(file_id, name, index, kind) if is_url else _UNSENDABLE


def _classify_block(block: _Block, index: int) -> _Classified:
    match block:
        case _VideoURLBlock():
            return _from_uri(_url(block.video_url), None, index, "file")
        case _InputAudioBlock(input_audio=_InputAudio(data=str(data), format=audio_format)):
            name: Final = f"attachment-{index}.{audio_format}" if audio_format else None
            return _from_base64(data, name, index, "audio", None)
        case _InputAudioBlock():
            return _UNSENDABLE
        case _ImageBlock(source=source):
            return _from_source(source, None, index, "image")
        case _:
            return _NOT_AN_ATTACHMENT


def _url(value: _ImageURL | str | None) -> str | None:
    return value.url if isinstance(value, _ImageURL) else value


def _from_uri(raw_uri: str | None, name: str | None, index: int, kind: AttachmentType) -> _Classified:
    uri: Final = (raw_uri or "").strip()
    if not uri:
        return _UNSENDABLE
    if uri.lower().startswith(_REMOTE_URI_SCHEMES):
        return Attachment(_filename(name, index, url=uri), kind, url=uri), False
    media_type, data = _parse_data_uri(uri)
    return _from_base64(data, name, index, kind, media_type)


def _from_source(source: _Source, name: str | None, index: int, kind: AttachmentType) -> _Classified:
    """base64 or a URL; text sources stay in the text check, and a file_id has nothing to send."""
    match source:
        case _Source(type="base64", data=str(data)):
            return _from_base64(data, name, index, kind, source.media_type)
        case _Source(type=str(source_type)) if source_type in _TEXT_SOURCE_TYPES and kind == "file":
            return _NOT_AN_ATTACHMENT
        case _Source(type="url", url=str(url)) if url:
            return Attachment(_filename(name, index, url=url), kind, url=url), False
        case _:
            return _UNSENDABLE


def _from_base64(data: str, name: str | None, index: int, kind: AttachmentType, media_type: str | None) -> _Classified:
    content: Final = _standard_base64(data)
    if content is None:
        return _UNSENDABLE
    return Attachment(_filename(name, index, media_type), kind, content=content), False


def _standard_base64(data: str) -> str | None:
    """Padded standard base64, accepting line breaks, missing padding and URL-safe characters."""
    compact: Final = "".join(data.split()).translate(_URL_SAFE_TO_STANDARD)
    padded: Final = compact + "=" * (-len(compact) % 4)
    return padded if compact and _is_base64(padded) else None


def _parse_data_uri(uri: str) -> tuple[str | None, str]:
    """(media type, base64 data); a plain data URI's text is encoded, anything else is taken as raw base64."""
    if uri[:5].lower() != "data:" or "," not in uri:
        return None, uri
    header, data = uri[5:].split(",", 1)
    params: Final = header.split(";")
    encoded: Final = params[-1].strip().lower() == "base64"
    return params[0], data if encoded else base64.b64encode(
        unquote_to_bytes(data.encode(errors="surrogatepass"))
    ).decode()


def _filename(name: str | None, index: int, media_type: str | None = None, url: str | None = None) -> str:
    """The client's name, else the URL's, with an extension from the media type when it has none."""
    stem: Final = posixpath.basename((name or "").strip()) or _url_basename(url) or f"attachment-{index}"
    extension: Final = mimetypes.guess_extension(media_type.split(";")[0].strip()) if media_type else None
    return stem if posixpath.splitext(stem)[1] or not extension else f"{stem}{extension}"


def _url_basename(url: str | None) -> str:
    try:
        return posixpath.basename(unquote(urlparse(url or "").path))
    except ValueError:
        return ""


def _is_base64(data: str) -> bool:
    try:
        base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def without_attachment_content(messages: object) -> object:
    items: Final = _parse(_ITEMS_ADAPTER, messages)
    return (
        messages
        if items is None
        else tuple(_without_content(_without_content(message, "content"), "output") for message in items)
    )


def _without_content(value: object, key: str) -> object:
    mapping: Final = _parse(_OBJECT_MAPPING, value)
    blocks: Final = _parse(_ITEMS_ADAPTER, mapping.get(key)) if mapping else None
    if mapping is None or blocks is None:
        return value
    return {**mapping, key: tuple(_block_without_content(block) for block in blocks)}


def _block_without_content(block: object) -> object:
    mapping: Final = _parse(_OBJECT_MAPPING, block) or {}
    block_type: Final = mapping.get("type")
    if block_type == "tool_result":
        return _without_content(block, "content")
    dropped: Final = _FILE_CHECKED_FIELDS.get(block_type) if isinstance(block_type, str) else None
    if dropped is None:
        return block
    source: Final = _parse(_OBJECT_MAPPING, mapping.get("source")) or {}
    source_type: Final = source.get("type")
    if block_type == "document" and isinstance(source_type, str) and source_type in _TEXT_SOURCE_TYPES:
        # A text document is prompt text, so it is checked here; only images nested in it go to the file check
        return {**mapping, "source": _without_content(source, "content")}
    kept: Final = {key: value for key, value in mapping.items() if key not in dropped}
    file: Final = _parse(_OBJECT_MAPPING, mapping.get("file")) if block_type == "file" else None
    if file is None:
        return kept
    return {**kept, "file": {key: value for key, value in file.items() if key not in _FILE_SOURCE_FIELDS}}


def _parse(adapter: TypeAdapter[_T], value: object) -> _T | None:
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None
