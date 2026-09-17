from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.history import HistorySurface, partition_history


def _exchange(surface: HistorySurface, identifier: str) -> tuple[Mapping[str, object], ...]:
    if surface == "chat":
        return (
            {"role": "assistant", "tool_calls": [{"id": identifier, "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": identifier, "content": identifier},
        )
    if surface == "responses":
        return (
            {"type": "function_call", "call_id": identifier, "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": identifier, "output": identifier},
        )
    return (
        {"role": "assistant", "content": [{"type": "tool_use", "id": identifier, "name": "read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": identifier, "content": identifier}]},
    )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_same_task_can_compact_completed_exchanges(surface: HistorySurface) -> None:
    system: Final = {"role": "system", "content": "Required policy"}
    task: Final = {"role": "user", "content": "Read files and retain the decisions"}
    old: Final = _exchange(surface, "old")
    current: Final = _exchange(surface, "current")
    prefix: Final = () if surface == "messages" else (system,)
    history: Final = (*prefix, task, *old, *current)
    partition: Final = partition_history(history, surface)
    assert partition is not None
    assert partition.invariants == prefix
    assert partition.past == old
    assert partition.active == (task, *current)
    assert partition.requires_active
    assert history == (*prefix, task, *old, *current)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_new_user_turn_releases_prior_tool_exchange(surface: HistorySurface) -> None:
    old: Final = _exchange(surface, "old")
    task: Final = {"role": "user", "content": "A new task"}
    partition: Final = partition_history((*old, task), surface)
    assert partition is not None
    assert partition.past == old
    assert partition.active == (task,)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("fault", ["orphan", "missing", "duplicate", "interrupted", "backwards"])
def test_invalid_exchange_never_produces_a_partition(surface: HistorySurface, fault: str) -> None:
    call, result = _exchange(surface, "call")
    items: Final = {
        "orphan": (result,),
        "missing": (call,),
        "duplicate": (call, result, call, result),
        "interrupted": (call, {"role": "user", "content": "interrupt"}, result),
        "backwards": (result, call),
    }
    assert partition_history(items[fault], surface) is None


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("item", [None, "text", {}, {"role": []}, {"role": "user", "content": [{"type": {}, "text": "data"}]}, {"role": "user", "content": object()}])
def test_invalid_json_message_shapes_fail_closed(surface: HistorySurface, item: object) -> None:
    assert partition_history((item,), surface) is None


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_opaque_fields_and_cross_surface_protocols_fail_closed(surface: HistorySurface) -> None:
    other: Final[HistorySurface] = "chat" if surface == "messages" else "messages"
    assert partition_history(_exchange(other, "foreign"), surface) is None
    assert partition_history(({"role": "assistant", "content": "answer", "provider_specific_fields": {"signature": "opaque"}},), surface) is None
    assert partition_history(({"role": "user", "content": "task"}, {"role": "system", "content": "late policy"}), surface) is None


def test_empty_and_invariant_only_history_stay_invariant() -> None:
    empty: Final = partition_history((), "chat")
    system: Final = {"role": "system", "content": "policy"}
    invariant: Final = partition_history((system,), "chat")
    assert empty is not None and empty.past == empty.active == empty.invariants == ()
    assert invariant is not None and invariant.invariants == (system,)
    assert invariant.past == invariant.active == ()


@pytest.mark.parametrize("new_task", [False, True])
def test_mixed_user_text_retains_whole_parallel_exchange_until_new_task(new_task: bool) -> None:
    task: Final = {"role": "user", "content": "Original task"}
    mixed: Final = (
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": identifier, "name": "read", "input": {}}
            for identifier in ("first", "second")
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "first", "content": "First result"},
            {"type": "text", "text": "Keep this follow-up"},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "second", "content": "Second result"}]},
    )
    terminal: Final = _exchange("messages", "terminal")
    next_task: Final = {"role": "user", "content": "New independent task"}
    history: Final = (task, *mixed, *terminal, *((next_task,) if new_task else ()))
    result: Final = partition_history(history, "messages")
    assert result is not None
    assert result.active == ((next_task,) if new_task else (task, *mixed, *terminal))
    assert result.past == ((task, *mixed, *terminal) if new_task else ())
