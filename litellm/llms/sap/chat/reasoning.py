from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.types.llms.openai import ChatCompletionThinkingBlock

_OBJECT: Final = TypeAdapter(dict[str, object])
_OBJECTS: Final = TypeAdapter(tuple[object, ...])


class _SAPThoughtBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    content: str | None = None
    signature: str | None = None


class _SAPThoughts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    reasoning_content: tuple[_SAPThoughtBlock, ...]
    thinking_blocks: tuple[ChatCompletionThinkingBlock, ...] = ()


def _thinking_block(block: _SAPThoughtBlock) -> ChatCompletionThinkingBlock:
    return ChatCompletionThinkingBlock(type="thinking", thinking=block.content or "", signature=block.signature)


def _as_object(value: object) -> Mapping[str, object] | None:
    try:
        return _OBJECT.validate_python(value)
    except ValidationError:
        return None


def _as_objects(value: object) -> Sequence[object] | None:
    try:
        return _OBJECTS.validate_python(value)
    except ValidationError:
        return None


def _parse_thoughts(body: object) -> _SAPThoughts | None:
    try:
        return _SAPThoughts.model_validate(body)
    except ValidationError:
        return None


def _normalize_body(body: object) -> object:
    parsed: Final = _parse_thoughts(body)
    fields: Final = _as_object(body)
    if parsed is None or fields is None:
        return body

    blocks: Final = tuple(
        _thinking_block(block) for block in parsed.reasoning_content if block.content is not None
    )
    reasoning_text: Final = "\n".join(
        block.content for block in parsed.reasoning_content if block.content is not None
    )

    return {
        **fields,
        "reasoning_content": reasoning_text or None,
        "thinking_blocks": [*parsed.thinking_blocks, *blocks],
    }


def _normalize_choice(choice: object, body_key: Literal["message", "delta"]) -> object:
    fields: Final = _as_object(choice)
    if fields is None or body_key not in fields:
        return choice

    return {**fields, body_key: _normalize_body(fields[body_key])}


def normalize_reasoning_blocks(result: object, body_key: Literal["message", "delta"]) -> object:
    fields: Final = _as_object(result)
    if fields is None:
        return result

    choices: Final = _as_objects(fields.get("choices"))
    if choices is None:
        return result

    return {**fields, "choices": [_normalize_choice(choice, body_key) for choice in choices]}


def denormalize_reasoning_blocks(
    message: Mapping[str, object],
) -> Mapping[str, object]:
    """Inverse of normalize_reasoning_blocks: rebuild list[ReasoningBlock] from thinking_blocks.

    LiteLLM normalizes SAP's reasoning_content list into a str + thinking_blocks before
    returning a ModelResponse. When that response is replayed as a history message the dict
    must be converted back to the SAP shape before SAPAssistantMessage validation.

    If thinking_blocks is absent but reasoning_content is a plain str (e.g. from a
    non-SAP provider), drop the field: we cannot reconstruct SAP ReasoningBlocks without
    the original signatures.
    """
    if message.get("role") != "assistant":
        return message
    thinking_blocks: Final = message.get("thinking_blocks")
    if isinstance(thinking_blocks, list):
        reasoning_content: Final[list[dict[str, str | None]]] = [  # mutable-ok: wire list passed to validate_dict
            {
                "content": str(
                    block.get("thinking") or ""  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # isinstance guard in comprehension filter; pyright can't narrow
                ),
                "signature": block.get(  # pyright: ignore[reportUnknownMemberType]  # isinstance guard in comprehension filter; pyright can't narrow
                    "signature"
                ),
            }
            for block in thinking_blocks  # pyright: ignore[reportUnknownVariableType]  # isinstance(thinking_blocks, list) guard above; pyright can't narrow into comprehension
            if isinstance(block, dict)
        ]
        return {k: v for k, v in message.items() if k not in ("thinking_blocks", "reasoning_content")} | {
            "reasoning_content": reasoning_content
        }
    if isinstance(message.get("reasoning_content"), str):
        return {k: v for k, v in message.items() if k not in ("thinking_blocks", "reasoning_content")}
    return message


_STREAM_CHUNK_KEYS: Final = ("final_result", "orchestration_result")


def normalize_stream_event(event: object) -> Mapping[str, object]:
    fields: Final = _as_object(event)
    if fields is None:
        return {}

    nested: Final = {
        key: normalize_reasoning_blocks(fields[key], "delta") for key in _STREAM_CHUNK_KEYS if key in fields
    }
    return _as_object(normalize_reasoning_blocks({**fields, **nested}, "delta")) or {}
