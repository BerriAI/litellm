"""``clear_tool_uses_20250919`` polyfill (v0: ``trigger`` and ``keep`` only)."""

from typing import Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AppliedEdit

from ..constants import (
    CLEAR_TOOL_USES_EDIT_TYPE,
    DEFAULT_INPUT_TOKENS_TRIGGER,
    DEFAULT_KEEP_TOOL_USES,
)
from ..placeholders import build_cleared_tool_result_content


def _count_tool_uses(messages: List[Dict[str, Any]]) -> int:
    """Return the number of tool_use content blocks across all messages.

    Only counts blocks with a string ``id`` to stay consistent with
    :func:`_collect_tool_use_ids_in_order`, which is the source of truth for
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


def _collect_tool_use_ids_in_order(messages: List[Dict[str, Any]]) -> List[str]:
    """Return tool_use ids in the chronological order they appear in messages."""
    ids: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    block_id = block.get("id")
                    if isinstance(block_id, str):
                        ids.append(block_id)
    return ids


def _trigger_met(
    trigger: Dict[str, Any],
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, Optional[int]]:
    """Return (trigger_met, input_tokens if counted for reuse)."""
    trigger_type = trigger.get("type", "input_tokens")
    threshold = trigger.get("value")

    if trigger_type == "tool_uses":
        if not isinstance(threshold, int):
            return False, None
        return _count_tool_uses(messages) > threshold, None

    if not isinstance(threshold, int):
        threshold = DEFAULT_INPUT_TOKENS_TRIGGER
    current_tokens = litellm.token_counter(
        model=model,
        messages=messages,
        tools=cast(Any, tools),
    )
    verbose_logger.debug(f"context_management polyfill: current_tokens: {current_tokens}")
    verbose_logger.debug(f"context_management polyfill: threshold: {threshold}")
    return current_tokens > threshold, current_tokens


def _resolve_keep_count(keep: Dict[str, Any]) -> int:
    keep_type = keep.get("type", "tool_uses")
    if keep_type != "tool_uses":
        return DEFAULT_KEEP_TOOL_USES
    value = keep.get("value")
    if not isinstance(value, int) or value < 0:
        return DEFAULT_KEEP_TOOL_USES
    return value


def _last_completed_tool_use_id(
    messages: List[Dict[str, Any]],
) -> Optional[str]:
    """Latest completed tool_result id; never cleared."""
    last_id: Optional[str] = None
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block_id = block.get("tool_use_id")
                    if isinstance(block_id, str):
                        last_id = block_id
    return last_id


def _clear_tool_results(messages: List[Dict[str, Any]], ids_to_clear: set) -> Tuple[List[Dict[str, Any]], int]:
    """Clear matching tool_result content; return (messages, cleared_count)."""
    cleared = 0
    new_messages: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_blocks: List[Any] = []
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


def apply_clear_tool_uses_20250919(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    system: Any,
    edit_spec: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[AppliedEdit]]:
    """Apply clear_tool_uses; return (messages, AppliedEdit or None)."""
    ignored_knobs = [knob for knob in ("clear_at_least", "exclude_tools", "clear_tool_inputs") if knob in edit_spec]
    for ignored_knob in ignored_knobs:
        verbose_logger.warning(
            "context_management polyfill: ignoring '%s' on %s "
            "(supported only on Anthropic-family forwarding path in v0)",
            ignored_knob,
            CLEAR_TOOL_USES_EDIT_TYPE,
        )

    trigger = edit_spec.get("trigger") or {
        "type": "input_tokens",
        "value": DEFAULT_INPUT_TOKENS_TRIGGER,
    }
    keep = edit_spec.get("keep") or {
        "type": "tool_uses",
        "value": DEFAULT_KEEP_TOOL_USES,
    }

    met, tokens_before = _trigger_met(trigger, model, messages, tools)
    if not met:
        return messages, None

    keep_count = _resolve_keep_count(keep)
    tool_use_ids = _collect_tool_use_ids_in_order(messages)
    if len(tool_use_ids) <= keep_count:
        return messages, None

    ids_to_clear = set(tool_use_ids[: len(tool_use_ids) - keep_count])

    # Never clear the latest completed tool_result (reply context).
    last_completed_id = _last_completed_tool_use_id(messages)
    if last_completed_id is not None:
        ids_to_clear.discard(last_completed_id)

    edited, cleared_count = _clear_tool_results(messages, ids_to_clear)
    verbose_logger.debug("context_management polyfill: edited: %s", edited)
    if cleared_count == 0:
        return messages, None

    if tokens_before is None:
        tokens_before = litellm.token_counter(model=model, messages=messages, tools=cast(Any, tools))
    tokens_after = litellm.token_counter(model=model, messages=edited, tools=cast(Any, tools))
    cleared_input_tokens = max(tokens_before - tokens_after, 0)

    applied: AppliedEdit = {
        "type": CLEAR_TOOL_USES_EDIT_TYPE,
        "cleared_tool_uses": cleared_count,
        "cleared_input_tokens": cleared_input_tokens,
    }
    if ignored_knobs:
        applied["warnings"] = [f"{knob}_ignored" for knob in ignored_knobs]
    return edited, applied
