import asyncio
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.agentic_followup_failure import (
    carry_agentic_followup_mark,
    is_agentic_followup_failure,
    mark_agentic_followup_failure,
    rounds_served_before,
    sent_tool_calls_before,
)
from litellm.litellm_core_utils.mcp_tool_dispatch import ToolDispatch, active_tool_dispatch
from litellm.types.utils import Usage


def _continuation_failure() -> litellm.InternalServerError:
    return litellm.InternalServerError(message="continuation failed", llm_provider="openai", model="gpt")


def _fail_inside(dispatch: ToolDispatch, error: BaseException) -> None:
    with pytest.raises(type(error)), dispatch.failures_not_replayed():
        raise error


def test_a_failure_after_a_tool_call_was_sent_is_marked_against_retries_and_fallbacks() -> None:
    dispatch: Final = ToolDispatch()
    dispatch.record_sent()
    error: Final = _continuation_failure()

    _fail_inside(dispatch, error)

    assert is_agentic_followup_failure(error)
    assert sent_tool_calls_before(error)


def test_a_cancellation_after_a_tool_call_was_sent_is_marked_too() -> None:
    dispatch: Final = ToolDispatch()
    dispatch.record_sent()
    cancelled: Final = asyncio.CancelledError()

    _fail_inside(dispatch, cancelled)

    assert sent_tool_calls_before(cancelled)


def test_a_failure_before_any_tool_call_was_sent_stays_retryable() -> None:
    error: Final = _continuation_failure()

    _fail_inside(ToolDispatch(), error)

    assert not is_agentic_followup_failure(error)
    assert not sent_tool_calls_before(error)


def test_marking_after_a_sent_tool_call_keeps_the_rounds_already_billed() -> None:
    served: Final = ((Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5), 0.25),)
    error: Final = _continuation_failure()
    mark_agentic_followup_failure(error, served)
    dispatch: Final = ToolDispatch()
    dispatch.record_sent()

    _fail_inside(dispatch, error)

    assert rounds_served_before(error) == served
    assert sent_tool_calls_before(error)


def test_a_later_agentic_mark_and_exception_mapping_keep_the_sent_tool_call() -> None:
    error: Final = _continuation_failure()
    dispatch: Final = ToolDispatch()
    dispatch.record_sent()
    _fail_inside(dispatch, error)
    mapped: Final = litellm.APIConnectionError(message="mapped", llm_provider="openai", model="gpt")

    mark_agentic_followup_failure(error, ())
    carry_agentic_followup_mark(error, mapped)

    assert sent_tool_calls_before(error)
    assert sent_tool_calls_before(mapped)


@pytest.mark.asyncio
async def test_a_send_recorded_by_a_task_spawned_while_recording_counts_for_the_request() -> None:
    dispatch: Final = ToolDispatch()

    async def send_from_a_task() -> None:
        spawned: Final = active_tool_dispatch()
        assert spawned is not None
        spawned.record_sent()

    with dispatch.recording():
        await asyncio.create_task(send_from_a_task())

    assert dispatch.sent


def test_recording_ends_with_its_scope_so_later_calls_are_not_attributed() -> None:
    dispatch: Final = ToolDispatch()

    with dispatch.recording():
        assert active_tool_dispatch() is dispatch

    assert active_tool_dispatch() is None
