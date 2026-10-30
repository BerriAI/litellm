import json
import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.responses.utils import ResponsesAPIRequestUtils

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


def _json_object(member: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(member)
    except ValidationError:
        return None


def _json_members(encrypted_content: object) -> tuple[object, ...]:
    if not isinstance(encrypted_content, str) or not encrypted_content.strip():
        return ()
    try:
        return tuple(_JSON_LIST.validate_json(encrypted_content))
    except ValidationError:
        return ()


def decode_thinking_blocks(encrypted_content: object) -> tuple[Mapping[str, object], ...] | None:
    blocks: Final = tuple(
        block
        for block in map(_json_object, _json_members(encrypted_content))
        if block is not None and is_verifiable_thinking_block(block)
    )
    return blocks or None


def _minted_blob(encrypted_content: object) -> str | None:
    if not isinstance(encrypted_content, str):
        return None
    _, original = ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(encrypted_content)
    return original if decode_thinking_blocks(original) is not None else None


def _thinking_text(block: Mapping[str, object] | None) -> str | None:
    if block is None or block.get("type") != "thinking":
        return None
    text: Final = block.get("thinking")
    return text if isinstance(text, str) else None


def thinking_block_texts(encrypted_content: object) -> Mapping[int, str]:
    blocks: Final = map(_json_object, _json_members(_minted_blob(encrypted_content)))
    return MappingProxyType({index: text for index, block in enumerate(blocks) if (text := _thinking_text(block))})


def with_thinking_text(encrypted_content: str, block_index: int, text: str) -> str:
    original: Final = _minted_blob(encrypted_content)
    if original is None or thinking_block_texts(original).get(block_index, text) == text:
        return encrypted_content
    members: Final = _json_members(original)
    block: Final = _JSON_OBJECT.validate_python(members[block_index])
    rewritten: Final = (*members[:block_index], {**block, "thinking": text}, *members[block_index + 1 :])
    return encrypted_content.removesuffix(original) + json.dumps(rewritten, separators=(",", ":"))


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
