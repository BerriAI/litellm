import base64
import binascii
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, TypeAlias, TypeGuard  # noqa: TID251  # narrows request JSON to str keys

from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentItem,
    BedrockImageContent,
    BedrockImageSource,
)
from litellm.types.utils import CallTypes

BedrockImageFormat: TypeAlias = Literal["png", "jpeg"]


class RequestAttachments(NamedTuple):
    images: tuple[BedrockContentItem, ...]
    unscannable: tuple[str, ...]
    document_texts: tuple[str, ...] = ()


class _Image(NamedTuple):
    item: BedrockContentItem


class _Unscannable(NamedTuple):
    label: str


class _DocumentText(NamedTuple):
    text: str


class _Block(NamedTuple):
    block: Mapping[str, object]
    from_tool: bool
    document_depth: int = 0


_Classified: TypeAlias = _Image | _Unscannable | _DocumentText | None
_BlockClassifier: TypeAlias = Callable[[Mapping[str, object]], _Classified]
_NestedToolBlocks: TypeAlias = Callable[
    [Mapping[str, object]],
    tuple[Mapping[str, object], ...],
]

NO_ATTACHMENTS: Final = RequestAttachments(images=(), unscannable=())

_IMAGE_FORMAT_BY_MIME: Final[Mapping[str, BedrockImageFormat]] = MappingProxyType(
    {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg"}
)
_CHAT_CALL_TYPES: Final = frozenset({CallTypes.completion.value, CallTypes.acompletion.value})
_RESPONSES_CALL_TYPES: Final = frozenset({CallTypes.responses.value, CallTypes.aresponses.value})
_OPENAI_IMAGE_TYPES: Final = frozenset({"image_url", "input_image", "computer_screenshot"})
_OPENAI_UNSCANNABLE_TYPES: Final = frozenset(
    {"file", "input_file", "input_audio", "video_url", "audio_url", "document", "container_upload"}
)
_ANTHROPIC_UNSCANNABLE_TYPES: Final = frozenset({"document", "container_upload"})
_TEXT_DOCUMENT_SOURCE_TYPES: Final = frozenset({"text", "content"})
_MAX_DOCUMENT_DEPTH: Final = 3
_CONVERSE_UNSCANNABLE_KEYS: Final = ("document", "video", "audio")
_CONVERSE_BINARY_SOURCE_KEYS: Final = ("bytes", "s3Location")
_MAX_IMAGE_BYTES: Final = 4 * 1024 * 1024
_MAX_IMAGE_BASE64_CHARS: Final = -(-_MAX_IMAGE_BYTES // 3) * 4
_URL_SAFE_TO_STANDARD_BASE64: Final = str.maketrans("-_", "+/")
_CONVERSE_ACTIONS: Final = frozenset({"converse", "converse-stream"})
_TOOL_ROLES: Final = frozenset({"tool", "function"})
_TOOL_OUTPUT_ITEM_TYPES: Final = frozenset({"function_call_output", "custom_tool_call_output", "computer_call_output"})
_MAX_LABEL_MIME_CHARS: Final = 40
_MAX_IMAGE_SIDE_PIXELS: Final = 8000
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE: Final = b"\xff\xd8\xff"
_JPEG_STANDALONE_MARKERS: Final = frozenset({0x01, *range(0xD0, 0xD9)})
_JPEG_FRAME_MARKERS: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def find_request_attachments(
    data: Mapping[str, object],
    call_type: str,
    skip_tool_messages: bool,
    latest_user_message_only: bool,
    scan_only_tool_results: bool = False,
) -> RequestAttachments:
    messages, classify, nested_tool_blocks = _messages_and_classifier(data, call_type)
    selected: Final = messages if not latest_user_message_only else _latest_user_message(messages)
    entries: Final = chain.from_iterable(_message_blocks(message, nested_tool_blocks) for message in selected)
    classified: Final = tuple(
        chain.from_iterable(
            _classify_entry(entry, classify)
            for entry in entries
            if _in_scope(entry, skip_tool_messages, scan_only_tool_results)
        )
    )
    return RequestAttachments(
        images=tuple(result.item for result in classified if isinstance(result, _Image)),
        unscannable=tuple(result.label for result in classified if isinstance(result, _Unscannable)),
        document_texts=tuple(result.text for result in classified if isinstance(result, _DocumentText)),
    )


def _classify_entry(entry: _Block, classify: _BlockClassifier) -> tuple[_Image | _Unscannable | _DocumentText, ...]:
    if entry.document_depth >= _MAX_DOCUMENT_DEPTH and _content_document_source(entry.block) is not None:
        return (_Unscannable("document (nested too deep)"),)
    text: Final = _document_text(entry)
    result: Final = classify(entry.block)
    return tuple(item for item in (_DocumentText(text) if text else None, result) if item is not None)


def _document_text(entry: _Block) -> str:
    block: Final = entry.block
    if entry.document_depth > 0 and block.get("type") == "text":
        text: Final = block.get("text")
        return text if isinstance(text, str) else ""
    if block.get("type") != "document":
        return ""
    source: Final = block.get("source")
    source_type: Final = source.get("type") if _is_mapping(source) else None
    source_text: Final = (
        (source.get("data") if source_type == "text" else source.get("content"))
        if _is_mapping(source) and isinstance(source_type, str) and source_type in _TEXT_DOCUMENT_SOURCE_TYPES
        else None
    )
    return "\n".join(
        part for part in (block.get("title"), block.get("context"), source_text) if isinstance(part, str) and part
    )


def _messages_and_classifier(
    data: Mapping[str, object], call_type: str
) -> tuple[Sequence[Mapping[str, object]], _BlockClassifier, _NestedToolBlocks]:
    if call_type in _CHAT_CALL_TYPES:
        return _mappings(data.get("messages")), _classify_openai_block, _no_nested_blocks
    if call_type == CallTypes.anthropic_messages.value:
        return _mappings(data.get("messages")), _classify_anthropic_block, _anthropic_tool_result_blocks
    if call_type in _RESPONSES_CALL_TYPES:
        return _mappings(data.get("input")), _classify_openai_block, _no_nested_blocks
    if call_type == CallTypes.allm_passthrough_route.value and _is_bedrock_converse(data):
        body: Final = data.get("data")
        messages: Final = _mappings(body.get("messages") if _is_mapping(body) else None)
        return messages, _classify_converse_block, _converse_tool_result_blocks
    return (), _classify_nothing, _no_nested_blocks


def _is_bedrock_converse(data: Mapping[str, object]) -> bool:
    endpoint: Final = data.get("endpoint")
    return (
        data.get("custom_llm_provider") == "bedrock"
        and isinstance(endpoint, str)
        and endpoint.rstrip("/").rsplit("/", 1)[-1] in _CONVERSE_ACTIONS
    )


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, Mapping)


def _is_list(value: object) -> TypeGuard[Sequence[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


def _mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not _is_list(value):
        return ()
    return tuple(item for item in value if _is_mapping(item))


def _latest_user_message(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(message for message in messages if message.get("role") == "user")[-1:]


def _message_blocks(message: Mapping[str, object], nested_tool_blocks: _NestedToolBlocks) -> tuple[_Block, ...]:
    message_type: Final = message.get("type")
    if isinstance(message_type, str) and message_type in _TOOL_OUTPUT_ITEM_TYPES:
        output: Final = message.get("output")
        blocks: Final = (output,) if _is_mapping(output) else _mappings(output)
        return _with_document_contents(_Block(block, from_tool=True) for block in blocks)
    role: Final = message.get("role")
    from_tool_message: Final = isinstance(role, str) and role in _TOOL_ROLES
    return _with_document_contents(
        chain.from_iterable(
            (
                _Block(block, from_tool=from_tool_message),
                *(_Block(inner, from_tool=True) for inner in nested_tool_blocks(block)),
            )
            for block in _mappings(message.get("content"))
        )
    )


def _with_document_contents(entries: Iterable[_Block]) -> tuple[_Block, ...]:
    return reduce(_with_document_level_expanded, range(_MAX_DOCUMENT_DEPTH), tuple(entries))


def _with_document_level_expanded(entries: tuple[_Block, ...], depth: int) -> tuple[_Block, ...]:
    return tuple(
        chain.from_iterable(
            _with_document_children(entry) if entry.document_depth == depth else (entry,) for entry in entries
        )
    )


def _with_document_children(entry: _Block) -> tuple[_Block, ...]:
    source: Final = _content_document_source(entry.block)
    if source is None:
        return (entry,)
    return (
        entry,
        *(
            _Block(inner, from_tool=entry.from_tool, document_depth=entry.document_depth + 1)
            for inner in _mappings(source.get("content"))
        ),
    )


def _content_document_source(block: Mapping[str, object]) -> Mapping[str, object] | None:
    source: Final = block.get("source")
    if block.get("type") != "document" or not _is_mapping(source) or source.get("type") != "content":
        return None
    return source


def _in_scope(entry: _Block, skip_tool_messages: bool, scan_only_tool_results: bool) -> bool:
    return not skip_tool_messages if entry.from_tool else not scan_only_tool_results


def _no_nested_blocks(block: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return ()


def _anthropic_tool_result_blocks(block: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return _mappings(block.get("content")) if block.get("type") == "tool_result" else ()


def _converse_tool_result_blocks(block: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    tool_result: Final = block.get("toolResult")
    return _mappings(tool_result.get("content")) if _is_mapping(tool_result) else ()


def _classify_nothing(block: Mapping[str, object]) -> _Classified:
    return None


def _classify_openai_block(block: Mapping[str, object]) -> _Classified:
    block_type: Final = block.get("type")
    if block_type in ("image", "document"):
        return _classify_anthropic_block(block)
    if isinstance(block_type, str) and block_type in _OPENAI_IMAGE_TYPES:
        image_url: Final = block.get("image_url")
        url: Final = image_url.get("url") if _is_mapping(image_url) else image_url
        return _classify_data_uri(url, block_type)
    if isinstance(block_type, str) and block_type in _OPENAI_UNSCANNABLE_TYPES:
        return _Unscannable(block_type)
    return None


def _classify_anthropic_block(block: Mapping[str, object]) -> _Classified:
    block_type: Final = block.get("type")
    if block_type == "image":
        source: Final = block.get("source")
        if _is_mapping(source) and source.get("type") == "base64":
            return _classify_base64(source.get("media_type"), source.get("data"), "image")
        return _Unscannable("image (url or file source)")
    if block_type == "document" and _is_text_document(block):
        return None
    if isinstance(block_type, str) and block_type in _ANTHROPIC_UNSCANNABLE_TYPES:
        return _Unscannable(block_type)
    return None


def _is_text_document(block: Mapping[str, object]) -> bool:
    source: Final = block.get("source")
    source_type: Final = source.get("type") if _is_mapping(source) else None
    return isinstance(source_type, str) and source_type in _TEXT_DOCUMENT_SOURCE_TYPES


def _classify_converse_block(block: Mapping[str, object]) -> _Classified:
    guard_content: Final = block.get("guardContent")
    image: Final = guard_content.get("image") if _is_mapping(guard_content) else block.get("image")
    if _is_mapping(image):
        image_format: Final = image.get("format")
        source: Final = image.get("source")
        encoded: Final = source.get("bytes") if _is_mapping(source) else None
        if encoded is None:
            return _Unscannable("image (no inline bytes)")
        mime: Final = f"image/{image_format}" if isinstance(image_format, str) else None
        return _classify_base64(mime, encoded, "image")
    document: Final = block.get("document")
    document_text: Final = _converse_document_text(document) if _is_mapping(document) else None
    if document_text is not None:
        return _DocumentText(document_text) if document_text else None
    for key in _CONVERSE_UNSCANNABLE_KEYS:
        if block.get(key) is not None:
            return _Unscannable(key)
    return None


def _converse_document_text(document: Mapping[str, object]) -> str | None:
    source: Final = document.get("source")
    if not _is_mapping(source) or any(source.get(key) is not None for key in _CONVERSE_BINARY_SOURCE_KEYS):
        return None
    parts: Final = _converse_document_parts(source)
    if parts is None:
        return None
    return "\n".join(
        part for part in (document.get("name"), document.get("context"), *parts) if isinstance(part, str) and part
    )


def _converse_document_parts(source: Mapping[str, object]) -> tuple[str, ...] | None:
    text: Final = source.get("text")
    if isinstance(text, str):
        return (text,)
    content: Final = source.get("content")
    if not _is_list(content):
        return None
    parts: Final = tuple(item.get("text") if _is_mapping(item) else None for item in content)
    return (
        tuple(part for part in parts if isinstance(part, str)) if all(isinstance(part, str) for part in parts) else None
    )


def _classify_data_uri(url: object, label: str) -> _Classified:
    if not isinstance(url, str) or not url.startswith("data:"):
        return _Unscannable(f"{label} (remote URL or file id)")
    header, _, payload = url.partition(",")
    params: Final = header.removeprefix("data:").lower().split(";")
    if "base64" not in params[1:]:
        return _Unscannable(f"{label} (not base64)")
    return _classify_base64(params[0], payload, label)


def _classify_base64(mime: object, encoded: object, label: str) -> _Classified:
    image_format: Final = _IMAGE_FORMAT_BY_MIME.get(mime.lower()) if isinstance(mime, str) else None
    if image_format is None:
        return _Unscannable(f"{label} ({mime[:_MAX_LABEL_MIME_CHARS]})" if isinstance(mime, str) and mime else label)
    standard: Final = _standard_base64(encoded)
    if len(standard) > _MAX_IMAGE_BASE64_CHARS:
        return _Unscannable(f"{label} (over 4 MB)")
    decoded: Final = _decoded(standard)
    if decoded is None:
        return _Unscannable(f"{label} (invalid base64)")
    if len(decoded) > _MAX_IMAGE_BYTES:
        return _Unscannable(f"{label} (over 4 MB)")
    actual_format: Final = _sniffed_format(decoded)
    if actual_format is None:
        return _Unscannable(f"{label} (not PNG or JPEG data)")
    if max(_pixel_size(decoded, actual_format), default=0) > _MAX_IMAGE_SIDE_PIXELS:
        return _Unscannable(f"{label} (over {_MAX_IMAGE_SIDE_PIXELS} pixels)")
    return _Image(
        BedrockContentItem(image=BedrockImageContent(format=actual_format, source=BedrockImageSource(bytes=standard)))
    )


def _sniffed_format(data: bytes) -> BedrockImageFormat | None:
    if data.startswith(_PNG_SIGNATURE):
        return "png"
    if data.startswith(_JPEG_SIGNATURE):
        return "jpeg"
    return None


def _pixel_size(data: bytes, image_format: BedrockImageFormat) -> tuple[int, ...]:
    if image_format == "png":
        return (
            (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")) if data[12:16] == b"IHDR" else ()
        )
    offset = 2  # rebind-ok: cursor that walks the JPEG segments
    while offset + 4 <= len(data) and data[offset] == 0xFF:
        marker = data[offset + 1]
        if marker == 0xFF or marker in _JPEG_STANDALONE_MARKERS:
            offset += 1 if marker == 0xFF else 2
            continue
        if marker in _JPEG_FRAME_MARKERS:
            return (
                (
                    int.from_bytes(data[offset + 7 : offset + 9], "big"),
                    int.from_bytes(data[offset + 5 : offset + 7], "big"),
                )
                if offset + 9 <= len(data)
                else ()
            )
        offset += 2 + int.from_bytes(data[offset + 2 : offset + 4], "big")
    return ()


def _standard_base64(encoded: object) -> str:
    compact: Final = (
        "".join(encoded.split()).translate(_URL_SAFE_TO_STANDARD_BASE64) if isinstance(encoded, str) else ""
    )
    return compact + "=" * (-len(compact) % 4)


def _decoded(standard: str) -> bytes | None:
    if not standard:
        return None
    try:
        return base64.b64decode(standard, validate=True)
    except (binascii.Error, ValueError):
        return None
