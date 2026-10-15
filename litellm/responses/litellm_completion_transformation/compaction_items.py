import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from openai.types.responses.response_compaction_item import ResponseCompactionItem
from openai.types.responses.response_create_params import ResponseInputParam
from pydantic import TypeAdapter, ValidationError

from litellm.responses.utils import ResponsesAPIRequestUtils

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])
_INSTRUCTION_ROLES: Final = frozenset(("system", "developer"))


def _json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class CompactionBlock:
    summary: str
    wire_block: Mapping[str, object]


def _compaction_block(value: object) -> CompactionBlock | None:
    block: Final = _json_object(value)
    if block is None:
        return None
    summary: Final = block.get("content")
    if block.get("type") != "compaction" or not isinstance(summary, str) or not summary:
        return None
    return CompactionBlock(summary=summary, wire_block=block)


def _newest_compaction_block(compaction_blocks: object) -> CompactionBlock | None:
    if not isinstance(compaction_blocks, list):
        return None
    blocks: Final = _OBJECT_LIST.validate_python(compaction_blocks)
    return next((block for value in reversed(blocks) if (block := _compaction_block(value)) is not None), None)


def compaction_output_item(provider_specific_fields: object) -> ResponseCompactionItem | None:
    try:
        fields: Final = _JSON_OBJECT.validate_python(provider_specific_fields)
    except ValidationError:
        return None
    block: Final = _newest_compaction_block(fields.get("compaction_blocks"))
    if block is None:
        return None
    return ResponseCompactionItem(
        id=f"cmp_{uuid.uuid4()}",
        encrypted_content=json.dumps(dict(block.wire_block), separators=(",", ":")),
        type="compaction",
    )


def _entry_type(entry: object) -> object:
    fields: Final = _json_object(entry)
    return None if fields is None else fields.get("type")


def _has_entry_of_type(entries: object, entry_type: str) -> bool:
    try:
        rows: Final = _OBJECT_LIST.validate_python(entries)
    except ValidationError:
        return False
    return any(_entry_type(row) == entry_type for row in rows)


def requests_compaction(context_management: object) -> bool:
    if _has_entry_of_type(context_management, "compaction"):
        return True
    anthropic_form: Final = _json_object(context_management)
    return anthropic_form is not None and _has_entry_of_type(anthropic_form.get("edits"), "compact_20260112")


def _without_affinity_wrap(encrypted_content: str) -> str:
    _, original = ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(encrypted_content)
    return original


def decode_compaction_item(encrypted_content: object) -> CompactionBlock | None:
    if not isinstance(encrypted_content, str):
        return None
    try:
        decoded: Final = _JSON_OBJECT.validate_json(_without_affinity_wrap(encrypted_content))
    except ValidationError:
        return None
    return _compaction_block(decoded)


def compaction_summary(item: object) -> str | None:
    fields: Final = _json_object(item)
    if fields is None or fields.get("type") != "compaction":
        return None
    block: Final = decode_compaction_item(fields.get("encrypted_content"))
    return None if block is None else block.summary


def with_compaction_summary(encrypted_content: str, summary: str) -> str:
    original: Final = _without_affinity_wrap(encrypted_content)
    block: Final = decode_compaction_item(original)
    if block is None:
        return encrypted_content
    rewritten: Final = json.dumps({**block.wire_block, "content": summary}, separators=(",", ":"))
    return encrypted_content.removesuffix(original) + rewritten


def _is_instruction_item(item: object) -> bool:
    fields: Final = _json_object(item)
    return fields is not None and fields.get("role") in _INSTRUCTION_ROLES


def without_items_before_compaction(input: str | ResponseInputParam) -> str | ResponseInputParam:
    if not isinstance(input, list):
        return input
    newest: Final = next(
        (index for index in range(len(input) - 1, 0, -1) if compaction_summary(input[index]) is not None), 0
    )
    return [*(item for item in input[:newest] if _is_instruction_item(item)), *input[newest:]]
