import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from openai.types.responses.response_compaction_item import ResponseCompactionItem
from pydantic import TypeAdapter, ValidationError

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])


@dataclass(frozen=True, slots=True)
class CompactionBlock:
    summary: str
    wire_block: Mapping[str, object]


def _compaction_block(value: object) -> CompactionBlock | None:
    try:
        block: Final = _JSON_OBJECT.validate_python(value)
    except ValidationError:
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


def decode_compaction_item(encrypted_content: object) -> CompactionBlock | None:
    if not isinstance(encrypted_content, str):
        return None
    try:
        decoded: Final = _JSON_OBJECT.validate_json(encrypted_content)
    except ValidationError:
        return None
    return _compaction_block(decoded)
