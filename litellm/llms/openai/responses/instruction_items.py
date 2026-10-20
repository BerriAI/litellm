from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from openai.types.responses import EasyInputMessageParam, ResponseInputContentParam, ResponseInputItemParam

from litellm.llms.base_llm.base_utils import hoisted_instruction_indices
from litellm.types.llms.openai import ResponseInputParam
from litellm.utils import accepts_mid_conversation_system

_INSTRUCTION_ROLES: Final = frozenset({"system", "developer"})


def _role(item: ResponseInputItemParam) -> str | None:
    match item:
        case {"role": str(role)}:
            return role
        case _:
            return None


def _developer_item_as_system(item: ResponseInputItemParam) -> ResponseInputItemParam:
    if "role" not in item or item["role"] != "developer":
        return item
    return EasyInputMessageParam(role="system", content=item["content"], type="message")


def developer_items_as_system(input: str | ResponseInputParam) -> str | ResponseInputParam:
    if isinstance(input, str):
        return input
    return [_developer_item_as_system(item) for item in input]


def _text_part(part: ResponseInputContentParam) -> str | None:
    match part:
        case {"type": "input_text", "text": str(text)}:
            return text
        case _:
            return None


def _text_only_content(item: ResponseInputItemParam) -> str | None:
    match item:
        case {"role": "system" | "developer", "content": str(text)}:
            return text
        case {"role": "system" | "developer", "content": [*parts]}:
            texts: Final = tuple(map(_text_part, parts))
            return None if any(text is None for text in texts) else "\n\n".join(text for text in texts if text)
        case _:
            return None


def _folded_instructions_and_items(
    items: tuple[ResponseInputItemParam, ...], instructions: str | None
) -> tuple[str | None, ResponseInputParam]:
    folded: Final = MappingProxyType(
        {
            index: text
            for index in hoisted_instruction_indices(tuple(map(_role, items)))
            if (text := _text_only_content(items[index])) is not None
        }
    )
    joined: Final = "\n\n".join(chunk for chunk in (instructions, *folded.values()) if chunk)
    return (
        instructions if not folded else joined or None,
        [_developer_item_as_system(item) for index, item in enumerate(items) if index not in folded],
    )


def instruction_items_folded(
    model: str,
    custom_llm_provider: str,
    input: str | ResponseInputParam,
    params: Mapping[str, object],
) -> tuple[ResponseInputParam, Mapping[str, object]] | None:
    """
    For a model whose chat template takes a single leading system message, the
    leading system and developer items and every later developer item folded into
    top-level `instructions`, or None when the request needs no folding.
    """
    instructions: Final = params.get("instructions")
    if isinstance(input, str) or not isinstance(instructions, str | None):
        return None
    items: Final = tuple(input)
    if not any(_role(item) in _INSTRUCTION_ROLES for item in items):
        return None
    if accepts_mid_conversation_system(model, custom_llm_provider):
        return None
    folded_instructions, folded_items = _folded_instructions_and_items(items, instructions)
    instruction_entries: Final = () if folded_instructions is None else (("instructions", folded_instructions),)
    return folded_items, MappingProxyType(
        {
            key: value
            for key, value in (
                *((key, value) for key, value in params.items() if key != "instructions"),
                *instruction_entries,
            )
        }
    )
