from collections.abc import Iterator, Mapping, Sequence
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputContentParam,
    ResponseInputItemParam,
    ResponseInputTextParam,
)

from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import CONVERTED_SYSTEM_NOTE
from litellm.llms.base_llm.base_utils import hoisted_instruction_indices, leading_instruction_block_length
from litellm.types.llms.openai import ResponseInputParam

UnfoldedDeveloperRole: TypeAlias = Literal["system", "user"]
SystemItemPlacement: TypeAlias = Literal[
    "leading_block_into_instructions", "leading_block_into_system_item", "every_item_in_place"
]
_ConversationItemKind: TypeAlias = Literal["system", "tool_output", "other"]

_INSTRUCTION_ROLES: Final = frozenset({"system", "developer"})


def _role(item: ResponseInputItemParam) -> str | None:
    match item:
        case {"role": str(role)}:
            return role
        case _:
            return None


def _developer_item_as(item: ResponseInputItemParam, role: UnfoldedDeveloperRole) -> ResponseInputItemParam:
    if "role" not in item or item["role"] != "developer" or "content" not in item:
        return item
    return EasyInputMessageParam(role=role, content=item["content"], type="message")


def developer_items_as_system(input: str | ResponseInputParam) -> str | ResponseInputParam:
    if isinstance(input, str):
        return input
    return [_developer_item_as(item, "system") for item in input]


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
    items: tuple[ResponseInputItemParam, ...], instructions: str | None, unfolded_developer_role: UnfoldedDeveloperRole
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
        [_developer_item_as(item, unfolded_developer_role) for index, item in enumerate(items) if index not in folded],
    )


def instruction_items_folded(
    input: str | ResponseInputParam,
    params: Mapping[str, object],
    unfolded_developer_role: UnfoldedDeveloperRole,
) -> tuple[ResponseInputParam, Mapping[str, object]] | None:
    """
    For a model whose chat template takes a single leading system message, the
    leading system and developer items and every later text developer item folded into
    top-level `instructions`, the developer items left over sent with
    `unfolded_developer_role`, or None when the request needs no folding.
    """
    instructions: Final = params.get("instructions")
    if isinstance(input, str) or not isinstance(instructions, str | None):
        return None
    items: Final = tuple(input)
    if not any(_role(item) in _INSTRUCTION_ROLES for item in items):
        return None
    folded_instructions, folded_items = _folded_instructions_and_items(items, instructions, unfolded_developer_role)
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


def _conversation_item_kind(item: ResponseInputItemParam) -> _ConversationItemKind:
    match item:
        case {"role": "system"}:
            return "system"
        case {"type": "function_call_output" | "custom_tool_call_output"}:
            return "tool_output"
        case _:
            return "other"


def _system_item_as_user(item: ResponseInputItemParam) -> ResponseInputItemParam:
    if "role" not in item or item["role"] != "system" or "content" not in item:
        return item
    content: Final = item["content"]
    parts: Final = (
        (ResponseInputTextParam(type="input_text", text=content),) if isinstance(content, str) else tuple(content)
    )
    return EasyInputMessageParam(
        role="user",
        content=[ResponseInputTextParam(type="input_text", text=CONVERTED_SYSTEM_NOTE), *parts],
        type="message",
    )


def _system_run_as_user(run: Sequence[ResponseInputItemParam]) -> tuple[ResponseInputItemParam, ...]:
    return tuple(map(_system_item_as_user, run))


def _conversation_with_system_items_as_user(
    conversation: Sequence[ResponseInputItemParam],
) -> Iterator[ResponseInputItemParam]:
    runs: Final = tuple((kind, tuple(run)) for kind, run in groupby(conversation, key=_conversation_item_kind))

    def emit(index: int) -> tuple[ResponseInputItemParam, ...]:
        kind, run = runs[index]
        follower: Final = runs[index + 1][0] if index + 1 < len(runs) else None
        if kind == "system":
            return () if follower == "tool_output" else _system_run_as_user(run)
        if kind == "tool_output" and index > 0 and runs[index - 1][0] == "system":
            return (*run, *_system_run_as_user(runs[index - 1][1]))
        return run

    return chain.from_iterable(map(emit, range(len(runs))))


def later_system_items_as_user(input: str | ResponseInputParam) -> str | ResponseInputParam:
    """
    For a model whose chat template takes a single leading system message, every `system` item after the
    leading instruction block sent as a user turn in its original position, prefixed with the operator note,
    so each request stays a prefix of the next. A run of them right before tool outputs is sent after those
    outputs, which have to follow their call.
    """
    if isinstance(input, str):
        return input
    items: Final = tuple(input)
    conversation_start: Final = leading_instruction_block_length(tuple(map(_role, items)))
    conversation: Final = items[conversation_start:]
    if not any(_conversation_item_kind(item) == "system" for item in conversation):
        return input
    return [*items[:conversation_start], *_conversation_with_system_items_as_user(conversation)]


def has_later_instruction_items(input: str | ResponseInputParam, continues_conversation: bool) -> bool:
    conversation_start: Final = 0 if continues_conversation else 1
    return not isinstance(input, str) and any(_role(item) in _INSTRUCTION_ROLES for item in input[conversation_start:])


def _instruction_items_as_user_turns(input: str | ResponseInputParam) -> str | ResponseInputParam:
    if isinstance(input, str):
        return input
    return [*_conversation_with_system_items_as_user(tuple(_developer_item_as(item, "system") for item in input))]


def _leading_block_into_system_item(input: str | ResponseInputParam) -> str | ResponseInputParam:
    folded: Final = instruction_items_folded(input, MappingProxyType({}), "user")
    if folded is None:
        return later_system_items_as_user(input)
    folded_items, folded_params = folded
    match folded_params.get("instructions"):
        case str(text):
            return later_system_items_as_user(
                [EasyInputMessageParam(role="system", content=text, type="message"), *folded_items]
            )
        case _:
            return later_system_items_as_user(folded_items)


def system_first_input_and_params(
    input: str | ResponseInputParam, params: Mapping[str, object], placement: SystemItemPlacement
) -> tuple[str | ResponseInputParam, Mapping[str, object]]:
    match placement:
        case "every_item_in_place":
            return _instruction_items_as_user_turns(input), params
        case "leading_block_into_system_item":
            return _leading_block_into_system_item(input), params
        case "leading_block_into_instructions":
            folded_input, folded_params = instruction_items_folded(input, params, "user") or (input, params)
            return later_system_items_as_user(folded_input), folded_params
