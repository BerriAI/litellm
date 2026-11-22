from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.constants import (
    DEFAULT_MAX_RECURSE_DEPTH,
    LITELLM_TRUNCATED_PAYLOAD_FIELD,
    REDACTED_BY_LITELM_STRING,
)
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR

SIGNED_FIELDS_BY_BLOCK_TYPE: Final = MappingProxyType(
    {
        "thinking": frozenset({"thinking", "signature"}),
        "redacted_thinking": frozenset({"data"}),
        "reasoning": frozenset({"encrypted_content"}),
    }
)
THOUGHT_SIGNATURE_KEYS: Final = frozenset({"thought_signature", "thought_signatures", "response_thought_signature"})
SIGNED_TOOL_CALL_ID_KEYS: Final = frozenset({"id", "call_id"})
_THINKING_BLOCK_TYPES: Final = frozenset({"thinking", "redacted_thinking"})
_CUT_FOR_STORAGE: Final = f"({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped "


def _is_signed_block_field(block: Mapping[str, object], key: str) -> bool:
    block_type: Final = block.get("type")
    if not isinstance(block_type, str) or key not in SIGNED_FIELDS_BY_BLOCK_TYPE.get(block_type, frozenset()):
        return False
    return block_type != "thinking" or bool(block.get("signature"))


def is_provider_signed(parent: Mapping[str, object], key: str, value: object) -> bool:
    if key in SIGNED_TOOL_CALL_ID_KEYS:
        return isinstance(value, str) and THOUGHT_SIGNATURE_SEPARATOR in value
    return key in THOUGHT_SIGNATURE_KEYS or _is_signed_block_field(parent, key)


def _is_altered_text(value: object) -> bool:
    return isinstance(value, str) and (_CUT_FOR_STORAGE in value or REDACTED_BY_LITELM_STRING in value)


def _was_altered(value: object) -> bool:
    return _is_altered_text(value) or any(map(_is_altered_text, as_object_list(value) or ()))


def _altered_signed_keys(fields: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        key
        for key, value in fields.items()
        if key not in SIGNED_TOOL_CALL_ID_KEYS and is_provider_signed(fields, key, value) and _was_altered(value)
    )


def _is_altered_thinking_block(value: object) -> bool:
    block: Final = as_str_mapping(value)
    return block is not None and block.get("type") in _THINKING_BLOCK_TYPES and bool(_altered_signed_keys(block))


def without_altered_provider_state(stored: object, depth: int = 0) -> object:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return stored
    items: Final = as_object_list(stored)
    if items is not None:
        return [
            without_altered_provider_state(item, depth + 1) for item in items if not _is_altered_thinking_block(item)
        ]
    fields: Final = as_str_mapping(stored)
    if fields is None:
        return stored
    altered: Final = _altered_signed_keys(fields)
    return {
        key: without_altered_provider_state(value, depth + 1) for key, value in fields.items() if key not in altered
    }
