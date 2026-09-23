"""Anthropic side of ``ResponsesCompactionCodec``: carries threshold compaction blocks
through the Responses chat-completions bridge as an opaque ``encrypted_content`` token."""

import json
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.types.llms.openai import ChatCompletionResponseMessage

_STR_KEY_DICT: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])
_STREAMED_COMPACTION_KEYS: Final = frozenset({"compaction_blocks", "compaction_start", "compaction_delta"})


def _compaction_block(value: object) -> Mapping[str, object] | None:
    try:
        block: Final = _STR_KEY_DICT.validate_python(value)
    except ValidationError:
        return None
    return block if block.get("type") == "compaction" else None


def _decode(encrypted_content: str) -> Mapping[str, object] | None:
    try:
        block: Final = _STR_KEY_DICT.validate_python(json.loads(encrypted_content))
    except (ValueError, ValidationError):
        return None
    return block if block.get("type") == "compaction" else None


def encode(provider_specific_fields: Mapping[str, object]) -> str | None:
    """Pack the newest compaction block verbatim. Server-tool loops can compact several
    times in one response and each block summarizes the prior summary plus everything
    after, so only the last is state worth replaying. Every field is kept because
    Anthropic rejects a replayed block whose content or encrypted_content changed."""
    try:
        blocks: Final = _OBJECT_LIST.validate_python(provider_specific_fields.get("compaction_blocks"))
    except ValidationError:
        return None
    newest: Final = next(
        (block for value in reversed(blocks) if (block := _compaction_block(value)) is not None),
        None,
    )
    return None if newest is None else json.dumps(newest, separators=(",", ":"), sort_keys=True)


def is_streaming_compaction(provider_specific_fields: object) -> bool:
    try:
        fields: Final = _STR_KEY_DICT.validate_python(provider_specific_fields)
    except ValidationError:
        return False
    return not _STREAMED_COMPACTION_KEYS.isdisjoint(fields)


def replay_message(encrypted_content: str) -> ChatCompletionResponseMessage | None:
    """Empty ``content`` keeps the prompt factory from adding a text block, so the signed
    compaction block is the whole assistant turn."""
    block: Final = _decode(encrypted_content)
    if block is None:
        return None
    message: Final = ChatCompletionResponseMessage(role="assistant", content="")
    message["provider_specific_fields"] = {"compaction_blocks": [block]}  # mutable-ok: message psf contract
    return message


def inspectable_text(encrypted_content: str) -> str | None:
    block: Final = _decode(encrypted_content)
    summary: Final = None if block is None else block.get("content")
    return summary if isinstance(summary, str) and summary else None


def _carries_compaction(message: object) -> bool:
    try:
        message_map: Final = _STR_KEY_DICT.validate_python(message)
        fields: Final = _STR_KEY_DICT.validate_python(message_map.get("provider_specific_fields"))
        blocks: Final = _OBJECT_LIST.validate_python(fields.get("compaction_blocks"))
    except ValidationError:
        return False
    return len(blocks) > 0


def superseded_replay_indices(messages: Sequence[object]) -> frozenset[int]:
    """Only the newest compaction block reflects the final state and Anthropic ignores
    everything before it, so every earlier replayed compaction turn is superseded,
    matching the native path's _slice_around_compaction_block."""
    carriers: Final = tuple(index for index, message in enumerate(messages) if _carries_compaction(message))
    return frozenset(carriers[:-1])
