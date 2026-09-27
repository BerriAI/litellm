"""
Find the attachments in a raw request that the Bedrock guardrail has to scan or refuse.

ApplyGuardrail scans inline PNG and JPEG images of up to 4 MB. Every other attachment
(documents, files, audio, video, and images sent as a remote URL, a file id, in another
format or over the size limit) is reported as unscannable so the guardrail can block the
request instead of letting the attachment reach the model unread.
"""

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, TypeGuard

from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentItem,
    BedrockImageContent,
)
from litellm.types.utils import CallTypes

BedrockImageFormat = Literal["png", "jpeg"]


class RequestAttachments(NamedTuple):
    images: tuple[BedrockContentItem, ...]
    unscannable: tuple[str, ...]


class _Image(NamedTuple):
    item: BedrockContentItem


class _Unscannable(NamedTuple):
    label: str


class _Block(NamedTuple):
    block: Mapping[str, object]
    from_tool: bool


_Classified = _Image | _Unscannable | None
_BlockClassifier = Callable[[Mapping[str, object]], _Classified]
_NestedToolBlocks = Callable[[Mapping[str, object]], tuple[Mapping[str, object], ...]]

NO_ATTACHMENTS: Final = RequestAttachments(images=(), unscannable=())

_IMAGE_FORMAT_BY_MIME: Final[Mapping[str, BedrockImageFormat]] = MappingProxyType(
    {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg"}
)
_CHAT_CALL_TYPES: Final = frozenset({CallTypes.completion.value, CallTypes.acompletion.value})
_RESPONSES_CALL_TYPES: Final = frozenset({CallTypes.responses.value, CallTypes.aresponses.value})
_CHAT_UNSCANNABLE_TYPES: Final = frozenset({"file", "input_audio", "video_url", "audio_url", "document"})
_ANTHROPIC_UNSCANNABLE_TYPES: Final = frozenset({"document", "container_upload"})
_RESPONSES_IMAGE_TYPES: Final = frozenset({"input_image", "computer_screenshot"})
_RESPONSES_UNSCANNABLE_TYPES: Final = frozenset({"input_file", "input_audio"})
_CONVERSE_UNSCANNABLE_KEYS: Final = ("document", "video")
_MAX_IMAGE_BYTES: Final = 4 * 1024 * 1024
_CONVERSE_ACTIONS: Final = frozenset({"converse", "converse-stream"})
_TOOL_ROLES: Final = frozenset({"tool", "function"})
_TOOL_OUTPUT_ITEM_TYPES: Final = frozenset({"function_call_output", "custom_tool_call_output", "computer_call_output"})
_MAX_LABEL_MIME_CHARS: Final = 40


def find_request_attachments(
    data: Mapping[str, object],
    call_type: str,
    skip_tool_messages: bool,
    latest_user_message_only: bool,
    scan_only_tool_results: bool = False,
) -> RequestAttachments:
    """List the scannable images and the unscannable attachments in the message content and tool results."""
    messages, classify, nested_tool_blocks = _messages_and_classifier(data, call_type)
    selected: Final = messages if not latest_user_message_only else _latest_user_message(messages)
    classified: Final = tuple(
        result
        for message in selected
        for entry in _message_blocks(message, nested_tool_blocks)
        if _in_scope(entry, skip_tool_messages, scan_only_tool_results)
        and (result := classify(entry.block)) is not None
    )
    return RequestAttachments(
        images=tuple(result.item for result in classified if isinstance(result, _Image)),
        unscannable=tuple(result.label for result in classified if isinstance(result, _Unscannable)),
    )


def _messages_and_classifier(
    data: Mapping[str, object], call_type: str
) -> tuple[Sequence[Mapping[str, object]], _BlockClassifier, _NestedToolBlocks]:
    if call_type in _CHAT_CALL_TYPES:
        return _mappings(data.get("messages")), _classify_chat_block, _no_nested_blocks
    if call_type == CallTypes.anthropic_messages.value:
        return _mappings(data.get("messages")), _classify_anthropic_block, _anthropic_tool_result_blocks
    if call_type in _RESPONSES_CALL_TYPES:
        return _mappings(data.get("input")), _classify_responses_block, _no_nested_blocks
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


def _is_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


def _mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not _is_list(value):
        return ()
    return tuple(item for item in value if _is_mapping(item))


def _latest_user_message(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(message for message in messages if message.get("role") == "user")[-1:]


def _message_blocks(message: Mapping[str, object], nested_tool_blocks: _NestedToolBlocks) -> tuple[_Block, ...]:
    if message.get("type") in _TOOL_OUTPUT_ITEM_TYPES:
        output: Final = message.get("output")
        blocks: Final = (output,) if _is_mapping(output) else _mappings(output)
        return tuple(_Block(block, from_tool=True) for block in blocks)
    from_tool_message: Final = message.get("role") in _TOOL_ROLES
    return tuple(
        entry
        for block in _mappings(message.get("content"))
        for entry in (
            _Block(block, from_tool=from_tool_message),
            *(_Block(inner, from_tool=True) for inner in nested_tool_blocks(block)),
        )
    )


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


def _classify_chat_block(block: Mapping[str, object]) -> _Classified:
    block_type: Final = block.get("type")
    if block_type == "image_url":
        image_url: Final = block.get("image_url")
        url: Final = image_url.get("url") if _is_mapping(image_url) else image_url
        return _classify_data_uri(url, "image_url")
    if isinstance(block_type, str) and block_type in _CHAT_UNSCANNABLE_TYPES:
        return _Unscannable(block_type)
    return None


def _classify_anthropic_block(block: Mapping[str, object]) -> _Classified:
    block_type: Final = block.get("type")
    if block_type == "image":
        source: Final = block.get("source")
        if _is_mapping(source) and source.get("type") == "base64":
            return _classify_base64(source.get("media_type"), source.get("data"), "image")
        return _Unscannable("image (url or file source)")
    if isinstance(block_type, str) and block_type in _ANTHROPIC_UNSCANNABLE_TYPES:
        return _Unscannable(block_type)
    return None


def _classify_responses_block(block: Mapping[str, object]) -> _Classified:
    block_type: Final = block.get("type")
    if isinstance(block_type, str) and block_type in _RESPONSES_IMAGE_TYPES:
        return _classify_data_uri(block.get("image_url"), block_type)
    if isinstance(block_type, str) and block_type in _RESPONSES_UNSCANNABLE_TYPES:
        return _Unscannable(block_type)
    return None


def _classify_converse_block(block: Mapping[str, object]) -> _Classified:
    image: Final = block.get("image")
    if _is_mapping(image):
        image_format: Final = image.get("format")
        source: Final = image.get("source")
        encoded: Final = source.get("bytes") if _is_mapping(source) else None
        if encoded is None:
            return _Unscannable("image (no inline bytes)")
        mime: Final = f"image/{image_format}" if isinstance(image_format, str) else None
        return _classify_base64(mime, encoded, "image")
    for key in _CONVERSE_UNSCANNABLE_KEYS:
        if key in block:
            return _Unscannable(key)
    return None


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
    compact: Final = "".join(encoded.split()) if isinstance(encoded, str) else ""
    decoded_size: Final = _decoded_size(compact)
    if decoded_size is None:
        return _Unscannable(f"{label} (invalid base64)")
    if decoded_size > _MAX_IMAGE_BYTES:
        return _Unscannable(f"{label} (over 4 MB)")
    return _Image(BedrockContentItem(image=BedrockImageContent(format=image_format, source={"bytes": compact})))


def _decoded_size(compact: str) -> int | None:
    if not compact:
        return None
    try:
        return len(base64.b64decode(compact, validate=True))
    except (binascii.Error, ValueError):
        return None
