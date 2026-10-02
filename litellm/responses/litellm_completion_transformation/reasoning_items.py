import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

from pydantic import BaseModel, TypeAdapter, ValidationError

REASONING_ITEM_ID_PREFIX: Final = "rs_"
_JSON_LIST: Final = TypeAdapter(list[object])
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def mint_reasoning_item_id() -> str:
    return f"{REASONING_ITEM_ID_PREFIX}{uuid.uuid4()}"


def is_verifiable_thinking_block(block: Mapping[str, object]) -> bool:
    block_type: Final = block.get("type")
    if block_type == "thinking":
        return bool(block.get("signature"))
    if block_type == "redacted_thinking":
        return bool(block.get("data"))
    return False


def encode_thinking_blocks(thinking_blocks: Sequence[Mapping[str, object]]) -> str | None:
    preserved: Final = [block for block in thinking_blocks if is_verifiable_thinking_block(block)]
    return json.dumps(preserved, separators=(",", ":")) if preserved else None


def _json_objects(members: Sequence[object]) -> Iterator[Mapping[str, object]]:
    for member in members:
        try:
            yield _JSON_OBJECT.validate_python(member)
        except ValidationError:
            continue


def decode_thinking_blocks(encrypted_content: object) -> tuple[Mapping[str, object], ...] | None:
    if not isinstance(encrypted_content, str) or not encrypted_content.strip():
        return None
    try:
        decoded: Final = _JSON_LIST.validate_json(encrypted_content)
    except ValidationError:
        return None
    blocks: Final = tuple(block for block in _json_objects(decoded) if is_verifiable_thinking_block(block))
    return blocks or None


def is_minted_reasoning_item_id(item_id: object) -> bool:
    if not isinstance(item_id, str) or not item_id.startswith(REASONING_ITEM_ID_PREFIX):
        return False
    suffix: Final = item_id.removeprefix(REASONING_ITEM_ID_PREFIX)
    try:
        parsed: Final = uuid.UUID(suffix)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == suffix


def is_litellm_minted_reasoning_item(item: object) -> bool:
    try:
        fields: Final = _JSON_OBJECT.validate_python(
            item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else item
        )
    except ValidationError:
        return False
    if fields.get("type") != "reasoning":
        return False
    return (
        is_minted_reasoning_item_id(fields.get("id"))
        or decode_thinking_blocks(fields.get("encrypted_content")) is not None
    )
