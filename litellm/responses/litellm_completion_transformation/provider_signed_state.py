from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.db_truncation import was_truncated_for_db
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR, tool_call_id_prefix

SIGNED_FIELDS_BY_BLOCK_TYPE: Final = MappingProxyType(
    {
        "thinking": frozenset({"thinking", "signature"}),
        "redacted_thinking": frozenset({"data"}),
        "reasoning": frozenset({"encrypted_content"}),
    }
)
THOUGHT_SIGNATURE_KEYS: Final = frozenset({"thought_signature", "thought_signatures", "response_thought_signature"})
SIGNED_TOOL_CALL_ID_KEYS: Final = frozenset({"id", "call_id"})
_REPLAYED_TOOL_CALL_ID_KEYS: Final = SIGNED_TOOL_CALL_ID_KEYS | {"tool_call_id", "tool_use_id"}
_THINKING_BLOCK_TYPES: Final = frozenset({"thinking", "redacted_thinking"})


def _block_type(block: Mapping[str, object]) -> str | None:
    block_type: Final = block.get("type")
    return block_type if isinstance(block_type, str) else None


def _is_signed_block_field(block: Mapping[str, object], key: str) -> bool:
    block_type: Final = _block_type(block)
    if block_type is None or key not in SIGNED_FIELDS_BY_BLOCK_TYPE.get(block_type, frozenset()):
        return False
    return block_type != "thinking" or bool(block.get("signature"))


def is_provider_signed(parent: Mapping[str, object], key: str, value: object) -> bool:
    if key in SIGNED_TOOL_CALL_ID_KEYS:
        return isinstance(value, str) and THOUGHT_SIGNATURE_SEPARATOR in value
    return key in THOUGHT_SIGNATURE_KEYS or _is_signed_block_field(parent, key)


def _is_altered_text(value: object) -> bool:
    return isinstance(value, str) and was_truncated_for_db(value)


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
    return block is not None and _block_type(block) in _THINKING_BLOCK_TYPES and bool(_altered_signed_keys(block))


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
        key: tool_call_id_prefix(value)
        if key in _REPLAYED_TOOL_CALL_ID_KEYS and _is_altered_text(value)
        else without_altered_provider_state(value, depth + 1)
        for key, value in fields.items()
        if key not in altered
    }


def _lost_the_signature_of_its_call(fields: Mapping[str, object]) -> bool:
    return "output" not in fields and any(
        is_provider_signed(fields, key, fields.get(key)) and _is_altered_text(fields.get(key))
        for key in SIGNED_TOOL_CALL_ID_KEYS
    )


def holds_altered_provider_state(stored: object, depth: int = 0) -> bool:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return False
    items: Final = as_object_list(stored)
    if items is not None:
        return any(holds_altered_provider_state(item, depth + 1) for item in items)
    fields: Final = as_str_mapping(stored)
    if fields is None:
        return False
    return (
        bool(_altered_signed_keys(fields))
        or _lost_the_signature_of_its_call(fields)
        or any(holds_altered_provider_state(value, depth + 1) for value in fields.values())
    )
