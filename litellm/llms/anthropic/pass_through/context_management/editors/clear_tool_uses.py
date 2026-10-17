"""``clear_tool_uses_20250919`` polyfill, matching the Anthropic API's selection.

``keep`` counts clearable tool uses only, and uses of ``exclude_tools`` are never cleared. Clearing goes oldest
first in batches of at least ``clear_at_least`` tokens and leaves an incomplete last batch, so the cleared prefix
only grows between requests and each batch costs one prompt-cache miss.
"""

from collections.abc import Mapping, Sequence, Set
from functools import reduce
from itertools import chain
from typing import Any, Final, Literal, TypeVar, cast

from pydantic import StrictInt, StrictStr, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AppliedEdit

from ..constants import (
    CLEAR_TOOL_USES_EDIT_TYPE,
    DEFAULT_INPUT_TOKENS_TRIGGER,
    DEFAULT_KEEP_TOOL_USES,
)
from ..placeholders import build_cleared_tool_result_content

T = TypeVar("T")
ToolNames = Literal["all"] | frozenset[str]


class _ClearAtLeast(TypedDict):
    type: NotRequired[ReadOnly[str]]
    value: ReadOnly[StrictInt]


_BLOCKS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_CLEAR_AT_LEAST: Final = TypeAdapter(_ClearAtLeast)
_TOOL_NAMES: Final = TypeAdapter(tuple[StrictStr, ...])


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


def _result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    texts: Final = (part.get("text") for part in _parse(_BLOCKS, content) or () if part.get("type") == "text")
    return "\n".join(text for text in texts if isinstance(text, str))


def _tool_result_text(block: Mapping[str, object]) -> tuple[str, str] | None:
    block_id: Final = block.get("tool_use_id")
    if block.get("type") != "tool_result" or not isinstance(block_id, str):
        return None
    return block_id, _result_text(block.get("content"))


def _tool_result_texts(blocks: Sequence[Mapping[str, object]]) -> Mapping[str, str]:
    """Return tool_use_id -> the text of its tool_result; non-text blocks count as empty."""
    pairs: Final = (_tool_result_text(block) for block in blocks)
    return dict(pair for pair in pairs if pair is not None)


def _resolve_clear_at_least(value: object) -> tuple[int, bool]:
    """Return (minimum tokens per batch, valid). 0 clears every eligible result."""
    if value is None:
        return 0, True
    parsed: Final = _parse(_CLEAR_AT_LEAST, value)
    if parsed is None or parsed.get("type", "input_tokens") != "input_tokens" or parsed["value"] < 0:
        return 0, False
    return parsed["value"], True


def _resolve_tool_names(value: object) -> tuple[frozenset[str], bool]:
    if value is None:
        return frozenset(), True
    names: Final = _parse(_TOOL_NAMES, value)
    return (frozenset(), False) if names is None else (frozenset(names), True)


def _resolve_clear_tool_inputs(value: object) -> tuple[ToolNames, bool]:
    if value is None or value is False:
        return frozenset(), True
    if value is True:
        return "all", True
    return _resolve_tool_names(value)


def _select_in_batches(candidates: Sequence[str], tokens: Mapping[str, int], clear_at_least: int) -> tuple[str, ...]:
    """Oldest first, close a batch once it holds ``clear_at_least`` tokens; drop the incomplete last batch."""

    def step(state: tuple[int, int], item: tuple[int, str]) -> tuple[int, int]:
        batch_end, batch_tokens = state
        index, tool_use_id = item
        total: Final = batch_tokens + tokens[tool_use_id]
        return (index + 1, 0) if total >= clear_at_least else (batch_end, total)

    batch_end, _ = reduce(step, enumerate(candidates), (0, 0))
    return tuple(candidates[:batch_end])


def _count_tool_uses(messages: list[dict[str, object]]) -> int:
    """Return the number of tool_use content blocks across all messages.

    Only counts blocks with a string ``id`` to stay consistent with
    :func:`_tool_uses_in_order`, which is the source of truth for
    which blocks are clearable.
    """
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if isinstance(block.get("id"), str):
                        count += 1
    return count


def _trigger_met(
    trigger: dict[str, object],
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
) -> tuple[bool, int | None]:
    """Return (trigger_met, input_tokens if counted for reuse)."""
    trigger_type: Final = trigger.get("type", "input_tokens")
    threshold = trigger.get("value")

    if trigger_type == "tool_uses":
        if not isinstance(threshold, int):
            return False, None
        return _count_tool_uses(messages) > threshold, None

    if not isinstance(threshold, int):
        threshold = DEFAULT_INPUT_TOKENS_TRIGGER
    current_tokens: Final = litellm.token_counter(
        model=model,
        messages=messages,
        tools=cast(Any, tools),
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
                new_block = {
                    **block,
                    "content": build_cleared_tool_result_content(block.get("content")),
                }
                new_blocks.append(new_block)
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
    if block.get("type") != "tool_use" or block.get("id") not in ids_to_clear:
        return block
    if isinstance(tool_names, frozenset) and block.get("name") not in tool_names:
        return block
    return {**block, "input": dict[str, object]()}


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
    """Apply clear_tool_uses; return (messages, AppliedEdit or None)."""
    clear_at_least, clear_at_least_valid = _resolve_clear_at_least(edit_spec.get("clear_at_least"))
    exclude_tools, exclude_tools_valid = _resolve_tool_names(edit_spec.get("exclude_tools"))
    input_tools, clear_tool_inputs_valid = _resolve_clear_tool_inputs(edit_spec.get("clear_tool_inputs"))
    checks: Final = (
        ("clear_at_least", clear_at_least_valid),
        ("exclude_tools", exclude_tools_valid),
        ("clear_tool_inputs", clear_tool_inputs_valid),
    )
    warnings: Final = tuple(f"{field}_invalid" for field, valid in checks if not valid)
    for warning in warnings:
        verbose_logger.warning(
            "context_management polyfill: %s on %s; the field is ignored",
            warning,
            CLEAR_TOOL_USES_EDIT_TYPE,
        )

    trigger: Final[dict[str, object]] = edit_spec.get("trigger") or {
        "type": "input_tokens",
        "value": DEFAULT_INPUT_TOKENS_TRIGGER,
    }
    keep: Final[dict[str, object]] = edit_spec.get("keep") or {
        "type": "tool_uses",
        "value": DEFAULT_KEEP_TOOL_USES,
    }

    met, counted_tokens = _trigger_met(trigger, model, messages, tools)
    if not met:
        return messages, None

    keep_count: Final = _resolve_keep_count(keep)
    blocks: Final = _all_blocks(messages)
    clearable: Final = tuple(
        tool_use_id for tool_use_id, name in _tool_uses_in_order(blocks) if name not in exclude_tools
    )
    if len(clearable) <= keep_count:
        return messages, None

    last_completed_id: Final = _last_completed_tool_use_id(messages)
    result_texts: Final = _tool_result_texts(blocks)
    candidates: Final = tuple(
        tool_use_id
        for tool_use_id in clearable[: len(clearable) - keep_count]
        if tool_use_id in result_texts and tool_use_id != last_completed_id
    )
    tokens: Final = {
        tool_use_id: litellm.token_counter(model=model, text=result_texts[tool_use_id])
        for tool_use_id in (candidates if clear_at_least > 0 else ())
    }
    ids_to_clear: Final = frozenset(
        _select_in_batches(candidates, tokens, clear_at_least) if clear_at_least > 0 else candidates
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
        else litellm.token_counter(model=model, messages=messages, tools=cast(Any, tools))
    )
    tokens_after: Final = litellm.token_counter(model=model, messages=final_messages, tools=cast(Any, tools))
    cleared_input_tokens: Final = max(tokens_before - tokens_after, 0)

    applied: Final[AppliedEdit] = {
        "type": CLEAR_TOOL_USES_EDIT_TYPE,
        "cleared_tool_uses": cleared_count,
        "cleared_input_tokens": cleared_input_tokens,
    }
    if warnings:
        applied["warnings"] = list(warnings)
    return final_messages, applied
