"""``clear_tool_uses_20250919`` polyfill, matching the Anthropic API's selection.

``keep`` counts clearable tool uses only. Uses of ``exclude_tools`` are never cleared and never count toward a
``tool_uses`` trigger, and neither are uses whose result is 1 to 46 bytes of text, which the API leaves out of its
count, unless ``clear_tool_inputs`` covers their tool: then their input can still be cleared, and the API counts and
clears them like any other. Clearing goes oldest first in batches that free at least ``clear_at_least`` tokens,
counting the result and any input it clears, and leaves an incomplete last batch, so the cleared prefix only grows
between requests and each batch costs one prompt-cache miss. ``check_clear_tool_uses_spec`` answers a malformed field
the way the API's 400 does, so the dispatcher can reject the edit, or drop only its malformed fields, before any editor
runs.
"""

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AppliedEdit

from ..constants import (
    CLEAR_TOOL_USES_EDIT_TYPE,
    DEFAULT_INPUT_TOKENS_TRIGGER,
    DEFAULT_KEEP_TOOL_USES,
)
from ..errors import AnthropicContextManagementError
from ..placeholders import build_cleared_tool_result_content

if TYPE_CHECKING:
    from litellm.types.llms.openai import ChatCompletionToolParam

T = TypeVar("T")
# api.anthropic.com, measured 2026-10-19: text results of 1 to 46 bytes are never cleared or counted toward keep
_TINY_RESULT_BYTES: Final = 46
ToolNames = Literal["all"] | frozenset[str]


class _ClearAtLeast(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["input_tokens"]
    value: Annotated[StrictInt, Field(ge=0)]


class _ClearToolUsesFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    clear_at_least: _ClearAtLeast | None = None
    exclude_tools: tuple[str, ...] | None = None
    clear_tool_inputs: bool | tuple[str, ...] | None = None


_BLOCKS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
# api.anthropic.com, measured 2026-10-19: its wording for the two shapes pydantic words differently
_API_MESSAGES: Final = MappingProxyType(
    {
        "tuple_type": "Input should be a valid array",
        "model_type": "Input does not match the expected shape.",
    }
)


class ClearToolUsesEditSpec(TypedDict, total=False):
    """The ``clear_tool_uses_20250919`` entry of a ``context_management`` spec."""

    type: ReadOnly[str]
    trigger: ReadOnly[dict[str, object]]
    keep: ReadOnly[dict[str, object]]
    clear_at_least: ReadOnly[object]
    exclude_tools: ReadOnly[object]
    clear_tool_inputs: ReadOnly[object]


def _parse(adapter: TypeAdapter[T], value: object) -> T | None:
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class CheckedEdit:
    edit: dict[str, object]
    error: AnthropicContextManagementError | None


def check_clear_tool_uses_spec(edit_spec: dict[str, object], index: int) -> CheckedEdit:
    """``edits[index]`` without its malformed fields, with the API's 400 for the first one, or None when there is none."""
    try:
        _ClearToolUsesFields.model_validate(edit_spec)
    except ValidationError as error:
        details: Final = error.errors(include_url=False)
        malformed: Final = frozenset(detail["loc"][0] for detail in details)
        first: Final = details[0]
        location: Final = ".".join(map(str, ("edits", index, CLEAR_TOOL_USES_EDIT_TYPE, *first["loc"])))
        return CheckedEdit(
            edit={key: value for key, value in edit_spec.items() if key not in malformed},
            error=AnthropicContextManagementError(
                status_code=400,
                message=f"context_management.{location}: {_API_MESSAGES.get(first['type'], first['msg'])}",
            ),
        )
    return CheckedEdit(edit=edit_spec, error=None)


def _input_tools(clear_tool_inputs: bool | tuple[str, ...] | None) -> ToolNames:
    return "all" if clear_tool_inputs is True else frozenset(clear_tool_inputs or ())


def _all_blocks(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(chain.from_iterable(_parse(_BLOCKS, message.get("content")) or () for message in messages))


def _tool_use_ref(block: Mapping[str, object]) -> tuple[str, str | None] | None:
    block_id: Final = block.get("id")
    name: Final = block.get("name")
    if block.get("type") != "tool_use" or not isinstance(block_id, str):
        return None
    return block_id, name if isinstance(name, str) else None


def _tool_uses_in_order(blocks: Sequence[Mapping[str, object]]) -> tuple[tuple[str, str | None], ...]:
    refs: Final = (_tool_use_ref(block) for block in blocks)
    return tuple(ref for ref in refs if ref is not None)


def _block_id(block: Mapping[str, object], block_type: str, id_key: str) -> str | None:
    block_id: Final = block.get(id_key)
    return block_id if block.get("type") == block_type and isinstance(block_id, str) else None


def _blocks_by_id(
    blocks: Sequence[Mapping[str, object]], block_type: str, id_key: str
) -> Mapping[str, Mapping[str, object]]:
    pairs: Final = ((_block_id(block, block_type, id_key), block) for block in blocks)
    return {block_id: block for block_id, block in pairs if block_id is not None}


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def _text_bytes(content: object) -> int | None:
    """Return the UTF-8 size of a text-only result, or None when it holds anything but text."""
    if isinstance(content, str):
        return _utf8_size(content)
    parts: Final = _parse(_BLOCKS, content) or ()
    texts: Final = tuple(part.get("text") for part in parts)
    if any(part.get("type") != "text" or not isinstance(text, str) for part, text in zip(parts, texts)):
        return None
    return sum(_utf8_size(text) for text in texts if isinstance(text, str))


def _is_tiny(content: object) -> bool:
    size: Final = _text_bytes(content)
    return size is not None and 0 < size <= _TINY_RESULT_BYTES


def _tiny_result_ids(results: Mapping[str, Mapping[str, object]]) -> frozenset[str]:
    """Return ids whose result is 1 to 46 bytes of text; the API neither counts nor clears those uses."""
    return frozenset(tool_use_id for tool_use_id, block in results.items() if _is_tiny(block.get("content")))


def _clears_input(name: str | None, input_tools: ToolNames) -> bool:
    return input_tools == "all" or (name is not None and name in input_tools)


def _cleared_result(block: Mapping[str, object]) -> dict[str, object]:
    return {**block, "content": build_cleared_tool_result_content(block.get("content"))}


def _freed_tokens(model: str, result: Mapping[str, object], use: Mapping[str, object], clears_input: bool) -> int:
    """Tokens that clearing this use frees: its result, and its input when ``clear_tool_inputs`` covers it."""
    before: Final = [result, use] if clears_input else [result]
    after: Final = [_cleared_result(result), {**use, "input": {}}] if clears_input else [_cleared_result(result)]
    counts: Final = tuple(
        litellm.token_counter(model=model, messages=[{"role": "user", "content": blocks}]) for blocks in (before, after)
    )
    return max(counts[0] - counts[1], 0)


def _select_in_batches(candidates: Sequence[str], tokens: Mapping[str, int], clear_at_least: int) -> tuple[str, ...]:
    """Oldest first, close a batch once it holds ``clear_at_least`` tokens; drop the incomplete last batch."""

    def step(state: tuple[int, int], item: tuple[int, str]) -> tuple[int, int]:
        batch_end, batch_tokens = state
        index, tool_use_id = item
        total: Final = batch_tokens + tokens[tool_use_id]
        return (index + 1, 0) if total >= clear_at_least else (batch_end, total)

    batch_end, _ = reduce(step, enumerate(candidates), (0, 0))
    return tuple(candidates[:batch_end])


def _trigger_met(
    trigger: dict[str, object],
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    counted_tool_uses: int,
) -> tuple[bool, int | None]:
    """Return (trigger_met, input_tokens if counted for reuse)."""
    trigger_type: Final = trigger.get("type", "input_tokens")
    threshold = trigger.get("value")

    if trigger_type == "tool_uses":
        if not isinstance(threshold, int):
            return False, None
        return counted_tool_uses > threshold, None

    if not isinstance(threshold, int):
        threshold = DEFAULT_INPUT_TOKENS_TRIGGER
    current_tokens: Final = litellm.token_counter(
        model=model,
        messages=messages,
        tools=cast("list[ChatCompletionToolParam] | None", tools),
    )
    verbose_logger.debug("context_management polyfill: current_tokens: %s", current_tokens)
    verbose_logger.debug("context_management polyfill: threshold: %s", threshold)
    return current_tokens > threshold, current_tokens


def _resolve_keep_count(keep: dict[str, object]) -> int:
    keep_type: Final = keep.get("type", "tool_uses")
    if keep_type != "tool_uses":
        return DEFAULT_KEEP_TOOL_USES
    value: Final = keep.get("value")
    if not isinstance(value, int) or value < 0:
        return DEFAULT_KEEP_TOOL_USES
    return value


def _last_completed_tool_use_id(
    messages: list[dict[str, object]],
) -> str | None:
    """Latest completed tool_result id; never cleared."""
    last_id: str | None = None
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block_id = block.get("tool_use_id")
                    if isinstance(block_id, str):
                        last_id = block_id
    return last_id


def _clear_tool_results(
    messages: list[dict[str, object]], ids_to_clear: set[str]
) -> tuple[list[dict[str, object]], int]:
    """Clear matching tool_result content; return (messages, cleared_count)."""
    cleared = 0
    new_messages: Final[list[dict[str, object]]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_blocks: list[object] = []
        mutated = False
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") in ids_to_clear
            ):
                new_blocks.append(_cleared_result(block))
                mutated = True
                cleared += 1
            else:
                new_blocks.append(block)

        if mutated:
            new_messages.append({**msg, "content": new_blocks})
        else:
            new_messages.append(msg)

    return new_messages, cleared


def _without_input(block: Mapping[str, object], ids_to_clear: Set[str], tool_names: ToolNames) -> Mapping[str, object]:
    match _tool_use_ref(block):
        case (block_id, name) if block_id in ids_to_clear and _clears_input(name, tool_names):
            return {**block, "input": dict[str, object]()}
        case _:
            return block


def _message_without_inputs(
    message: dict[str, object], ids_to_clear: Set[str], tool_names: ToolNames
) -> dict[str, object]:  # mutable-ok: the editor contract hands messages back as list[dict]
    blocks: Final = _parse(_BLOCKS, message.get("content"))
    if blocks is None:
        return message
    return {**message, "content": [_without_input(block, ids_to_clear, tool_names) for block in blocks]}


def apply_clear_tool_uses_20250919(
    *,
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    system: str | list[dict[str, object]] | None,
    edit_spec: ClearToolUsesEditSpec,
) -> tuple[list[dict[str, object]], AppliedEdit | None]:
    """Apply clear_tool_uses to an edit ``check_clear_tool_uses_spec`` returned; return (messages, AppliedEdit or None)."""
    fields: Final = _ClearToolUsesFields.model_validate(edit_spec)
    clear_at_least: Final = 0 if fields.clear_at_least is None else fields.clear_at_least.value
    exclude_tools: Final = frozenset(fields.exclude_tools or ())
    input_tools: Final = _input_tools(fields.clear_tool_inputs)

    trigger: Final[dict[str, object]] = edit_spec.get("trigger") or {
        "type": "input_tokens",
        "value": DEFAULT_INPUT_TOKENS_TRIGGER,
    }
    keep: Final[dict[str, object]] = edit_spec.get("keep") or {
        "type": "tool_uses",
        "value": DEFAULT_KEEP_TOOL_USES,
    }

    blocks: Final = _all_blocks(messages)
    results: Final = _blocks_by_id(blocks, "tool_result", "tool_use_id")
    uses: Final = _blocks_by_id(blocks, "tool_use", "id")
    tiny: Final = _tiny_result_ids(results)
    clearable: Final = tuple(
        (tool_use_id, name)
        for tool_use_id, name in _tool_uses_in_order(blocks)
        if name not in exclude_tools and (tool_use_id not in tiny or _clears_input(name, input_tools))
    )
    met, counted_tokens = _trigger_met(trigger, model, messages, tools, len(clearable))
    if not met:
        return messages, None

    keep_count: Final = _resolve_keep_count(keep)
    if len(clearable) <= keep_count:
        return messages, None

    last_completed_id: Final = _last_completed_tool_use_id(messages)
    candidates: Final = tuple(
        (tool_use_id, name)
        for tool_use_id, name in clearable[: len(clearable) - keep_count]
        if tool_use_id in results and tool_use_id != last_completed_id
    )
    candidate_ids: Final = tuple(tool_use_id for tool_use_id, _ in candidates)
    tokens: Final = {
        tool_use_id: _freed_tokens(model, results[tool_use_id], uses[tool_use_id], _clears_input(name, input_tools))
        for tool_use_id, name in (candidates if clear_at_least > 0 else ())
    }
    ids_to_clear: Final = frozenset(
        _select_in_batches(candidate_ids, tokens, clear_at_least) if clear_at_least > 0 else candidate_ids
    )

    edited, cleared_count = _clear_tool_results(messages, set(ids_to_clear))
    verbose_logger.debug("context_management polyfill: edited: %s", edited)
    if cleared_count == 0:
        return messages, None
    final_messages: Final = (
        [_message_without_inputs(message, ids_to_clear, input_tools) for message in edited] if input_tools else edited
    )

    tokens_before: Final = (
        counted_tokens
        if counted_tokens is not None
        else litellm.token_counter(
            model=model, messages=messages, tools=cast("list[ChatCompletionToolParam] | None", tools)
        )
    )
    tokens_after: Final = litellm.token_counter(
        model=model, messages=final_messages, tools=cast("list[ChatCompletionToolParam] | None", tools)
    )
    cleared_input_tokens: Final = max(tokens_before - tokens_after, 0)

    return final_messages, {
        "type": CLEAR_TOOL_USES_EDIT_TYPE,
        "cleared_tool_uses": cleared_count,
        "cleared_input_tokens": cleared_input_tokens,
    }
