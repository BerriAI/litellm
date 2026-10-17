"""``clear_tool_uses_20250919`` polyfill.

Honors every field of the edit: ``trigger``, ``keep``, ``clear_at_least``,
``exclude_tools`` and ``clear_tool_inputs``. Selection follows the Anthropic
API, as measured against it:

- ``keep`` counts clearable tool uses only; uses of ``exclude_tools`` are never
  cleared and don't count toward ``keep`` or ``clear_at_least``.
- Clearing goes oldest first in batches of at least ``clear_at_least`` tokens;
  an incomplete last batch stays. The cleared prefix therefore only grows from
  one request to the next instead of moving with every new tool use, so each
  batch costs one prompt-cache miss and the requests between batches hit.
- ``clear_tool_inputs`` (``true`` or a list of tool names) also clears the
  ``input`` of the cleared tool uses.
"""

from typing import Any, Final, cast

from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AppliedEdit

from ..constants import (
    CLEAR_TOOL_USES_EDIT_TYPE,
    DEFAULT_INPUT_TOKENS_TRIGGER,
    DEFAULT_KEEP_TOOL_USES,
)
from ..placeholders import build_cleared_tool_result_content


class ClearToolUsesEditSpec(TypedDict, total=False):
    """The ``clear_tool_uses_20250919`` entry of a ``context_management`` spec."""

    type: ReadOnly[str]
    trigger: ReadOnly[dict[str, object]]
    keep: ReadOnly[dict[str, object]]
    clear_at_least: ReadOnly[object]
    exclude_tools: ReadOnly[object]
    clear_tool_inputs: ReadOnly[object]


def _tool_uses_in_order(messages: list[dict[str, object]]) -> list[tuple[str, str | None]]:
    """Return (id, name) of every tool_use block, in chronological order."""
    uses: Final[list[tuple[str, str | None]]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    block_id = block.get("id")
                    if isinstance(block_id, str):
                        name = block.get("name")
                        uses.append((block_id, name if isinstance(name, str) else None))
    return uses


def _tool_result_texts(messages: list[dict[str, object]]) -> dict[str, str]:
    """Return tool_use_id -> the text of its tool_result (non-text blocks count as empty)."""
    texts: Final[dict[str, str]] = {}
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            block_id = block.get("tool_use_id")
            if not isinstance(block_id, str):
                continue
            result = block.get("content")
            if isinstance(result, str):
                texts[block_id] = result
            elif isinstance(result, list):
                texts[block_id] = "\n".join(
                    part["text"]
                    for part in result
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                )
            else:
                texts[block_id] = ""
    return texts


def _resolve_clear_at_least(value: object) -> tuple[int, bool]:
    """Return (minimum tokens per batch, valid). 0 clears every eligible result."""
    if value is None:
        return 0, True
    if isinstance(value, dict) and value.get("type", "input_tokens") == "input_tokens":
        tokens = value.get("value")
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
            return tokens, True
    return 0, False


def _resolve_tool_names(value: object) -> tuple[frozenset[str], bool]:
    """Return (tool names, valid) for a list-of-names field."""
    if value is None:
        return frozenset(), True
    if isinstance(value, list) and all(isinstance(name, str) for name in value):
        return frozenset(value), True
    return frozenset(), False


def _resolve_clear_tool_inputs(value: object) -> tuple[bool | frozenset[str], bool]:
    """Return (``True`` for all tools, else the tool names whose inputs to clear; valid)."""
    if value is None or value is False:
        return frozenset(), True
    if value is True:
        return True, True
    return _resolve_tool_names(value)


def _select_in_batches(
    candidates: list[str],
    tokens: dict[str, int],
    clear_at_least: int,
) -> list[str]:
    """Oldest first, close a batch once it holds ``clear_at_least`` tokens; drop the incomplete last batch."""
    selected: Final[list[str]] = []
    batch: list[str] = []
    batch_tokens = 0
    for tool_use_id in candidates:
        batch.append(tool_use_id)
        batch_tokens += tokens[tool_use_id]
        if batch_tokens >= clear_at_least:
            selected.extend(batch)
            batch = []
            batch_tokens = 0
    return selected


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


def _clear_tool_inputs(
    messages: list[dict[str, object]],
    ids_to_clear: set[str],
    tools_to_clear: bool | frozenset[str],
) -> list[dict[str, object]]:
    """Empty the ``input`` of the matching tool_use blocks."""
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
                and block.get("type") == "tool_use"
                and block.get("id") in ids_to_clear
                and (tools_to_clear is True or block.get("name") in tools_to_clear)
            ):
                new_blocks.append({**block, "input": {}})
                mutated = True
            else:
                new_blocks.append(block)
        new_messages.append({**msg, "content": new_blocks} if mutated else msg)
    return new_messages


def apply_clear_tool_uses_20250919(
    *,
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    system: str | list[dict[str, object]] | None,
    edit_spec: ClearToolUsesEditSpec,
) -> tuple[list[dict[str, object]], AppliedEdit | None]:
    """Apply clear_tool_uses; return (messages, AppliedEdit or None)."""
    warnings: Final[list[str]] = []
    clear_at_least, valid = _resolve_clear_at_least(edit_spec.get("clear_at_least"))
    if not valid:
        warnings.append("clear_at_least_invalid")
    exclude_tools, valid = _resolve_tool_names(edit_spec.get("exclude_tools"))
    if not valid:
        warnings.append("exclude_tools_invalid")
    tools_to_clear_inputs, valid = _resolve_clear_tool_inputs(edit_spec.get("clear_tool_inputs"))
    if not valid:
        warnings.append("clear_tool_inputs_invalid")
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

    met, tokens_before = _trigger_met(trigger, model, messages, tools)
    if not met:
        return messages, None

    keep_count: Final = _resolve_keep_count(keep)
    # Uses of excluded tools are never cleared and don't count toward ``keep``.
    clearable: Final = [tool_use_id for tool_use_id, name in _tool_uses_in_order(messages) if name not in exclude_tools]
    if len(clearable) <= keep_count:
        return messages, None

    # Never clear the latest completed tool_result (reply context).
    last_completed_id: Final = _last_completed_tool_use_id(messages)
    result_texts: Final = _tool_result_texts(messages)
    candidates: Final = [
        tool_use_id
        for tool_use_id in clearable[: len(clearable) - keep_count]
        if tool_use_id in result_texts and tool_use_id != last_completed_id
    ]
    ids_to_clear: Final = set(candidates)
    if clear_at_least > 0:
        tokens: Final = {
            tool_use_id: litellm.token_counter(model=model, text=result_texts[tool_use_id])
            for tool_use_id in candidates
        }
        ids_to_clear.intersection_update(_select_in_batches(candidates, tokens, clear_at_least))

    edited, cleared_count = _clear_tool_results(messages, ids_to_clear)
    verbose_logger.debug("context_management polyfill: edited: %s", edited)
    if cleared_count == 0:
        return messages, None
    if tools_to_clear_inputs:
        edited = _clear_tool_inputs(edited, ids_to_clear, tools_to_clear_inputs)

    if tokens_before is None:
        tokens_before = litellm.token_counter(model=model, messages=messages, tools=cast(Any, tools))
    tokens_after: Final = litellm.token_counter(model=model, messages=edited, tools=cast(Any, tools))
    cleared_input_tokens: Final = max(tokens_before - tokens_after, 0)

    applied: Final[AppliedEdit] = {
        "type": CLEAR_TOOL_USES_EDIT_TYPE,
        "cleared_tool_uses": cleared_count,
        "cleared_input_tokens": cleared_input_tokens,
    }
    if warnings:
        applied["warnings"] = warnings
    return edited, applied
