from __future__ import annotations

from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.types.llms.openai import ChatCompletionThinkingBlock


class _SAPThoughtBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    content: str = ""
    signature: str | None = None


class _SAPThoughts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    reasoning_content: tuple[_SAPThoughtBlock, ...]
    thinking_blocks: tuple[ChatCompletionThinkingBlock, ...] = ()


def _thinking_block(block: _SAPThoughtBlock) -> ChatCompletionThinkingBlock:
    return ChatCompletionThinkingBlock(type="thinking", thinking=block.content, signature=block.signature)


def _parse_thoughts(body: object) -> _SAPThoughts | None:
    try:
        return _SAPThoughts.model_validate(body)
    except ValidationError:
        return None


def _normalize_body(body: object) -> object:
    parsed: Final = _parse_thoughts(body)
    if parsed is None:
        return body

    blocks: Final = tuple(_thinking_block(block) for block in parsed.reasoning_content)
    reasoning_text: Final = "".join(block.content for block in parsed.reasoning_content)

    return {
        **cast(dict[str, object], body),
        "reasoning_content": reasoning_text or None,
        "thinking_blocks": [*parsed.thinking_blocks, *blocks],
    }


def _normalize_choice(choice: object, body_key: Literal["message", "delta"]) -> object:
    if not isinstance(choice, dict):
        return choice

    choice_dict: Final = cast(dict[str, object], choice)
    if body_key not in choice_dict:
        return choice_dict

    return {**choice_dict, body_key: _normalize_body(choice_dict[body_key])}


def normalize_reasoning_blocks(result: object, body_key: Literal["message", "delta"]) -> object:
    if not isinstance(result, dict):
        return result

    result_dict: Final = cast(dict[str, object], result)
    choices: Final = result_dict.get("choices")
    if not isinstance(choices, list):
        return result_dict

    return {
        **result_dict,
        "choices": [_normalize_choice(choice, body_key) for choice in cast(list[object], choices)],
    }
