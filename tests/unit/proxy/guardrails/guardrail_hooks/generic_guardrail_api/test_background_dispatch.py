import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.background_dispatch import (
    DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT,
    BackgroundDispatcher,
    fire_and_forget_from_config,
    max_inflight_from_config,
)

_request_scoped: Final[contextvars.ContextVar[str | None]] = contextvars.ContextVar("request_scoped", default=None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), (True, True), (False, False), ("true", True), ("false", False), (1, True), (0, False)],
)
def test_fire_and_forget_accepts_bools_and_their_config_spellings(value: object, expected: bool) -> None:
    assert fire_and_forget_from_config(value) is expected


@pytest.mark.parametrize("value", ["maybe", 2, [True]])
def test_an_unparseable_fire_and_forget_is_ignored_with_a_warning(
    value: object, warning_messages: Callable[[str], list[str]]
) -> None:
    assert fire_and_forget_from_config(value) is False
    assert len(warning_messages("Ignoring fire_and_forget=")) == 1


@pytest.mark.parametrize(("value", "expected"), [(None, DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT), (5, 5), ("5", 5)])
def test_max_inflight_accepts_positive_integers(value: object, expected: int) -> None:
    assert max_inflight_from_config(value) == expected


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "many"])
def test_an_invalid_max_inflight_falls_back_to_the_default_with_a_warning(
    value: object, warning_messages: Callable[[str], list[str]]
) -> None:
    assert max_inflight_from_config(value) == DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT
    assert len(warning_messages("Ignoring fire_and_forget_max_inflight=")) == 1


def test_a_dispatcher_needs_room_for_at_least_one_call() -> None:
    with pytest.raises(ValueError, match="fire_and_forget_max_inflight"):
        BackgroundDispatcher(guardrail_name="g", max_inflight=0)


async def test_calls_beyond_the_cap_are_dropped_unprepared_counted_and_warned_once(
    warning_messages: Callable[[str], list[str]],
) -> None:
    gate: Final = asyncio.Event()
    dispatcher: Final = BackgroundDispatcher(guardrail_name="g", max_inflight=2)
    prepared: Final[list[int]] = []  # mutable-ok: records which calls were prepared

    def prepare(call: int) -> Callable[[], Awaitable[bool]]:
        prepared.append(call)
        return gate.wait

    dispatched: Final = [dispatcher.dispatch(partial(prepare, i), context=f"call {i}") for i in range(5)]

    assert (dispatched, dispatcher.pending_count, dispatcher.dropped_count) == ([True, True, False, False, False], 2, 3)
    assert prepared == [0, 1]
    assert len(warning_messages("dropped")) == 1
    gate.set()
    await dispatcher.wait_for_pending()


async def test_a_finished_call_frees_its_slot() -> None:
    dispatcher: Final = BackgroundDispatcher(guardrail_name="g", max_inflight=1)

    async def finish() -> None:
        return None

    for _ in range(3):
        assert dispatcher.dispatch(lambda: finish, context="call") is True
        await dispatcher.wait_for_pending()

    assert (dispatcher.pending_count, dispatcher.dropped_count) == (0, 0)


async def test_a_failing_call_is_logged_with_its_context_and_not_raised(
    warning_messages: Callable[[str], list[str]],
) -> None:
    dispatcher: Final = BackgroundDispatcher(guardrail_name="audit", max_inflight=1)

    async def fail() -> None:
        raise ConnectionError("refused")

    dispatcher.dispatch(lambda: fail, context="input_type=response litellm_call_id=call-1")
    await dispatcher.wait_for_pending()

    assert warning_messages("call failed") == [
        "Generic Guardrail API (audit, fire_and_forget) call failed. "
        "input_type=response litellm_call_id=call-1: refused"
    ]


async def test_a_dispatched_call_does_not_see_the_request_context() -> None:
    dispatcher: Final = BackgroundDispatcher(guardrail_name="g", max_inflight=1)
    seen: Final[list[str | None]] = []  # mutable-ok: records what the background call saw

    async def record() -> None:
        seen.append(_request_scoped.get())

    token: Final = _request_scoped.set("request-1")
    try:
        dispatcher.dispatch(lambda: record, context="call")
    finally:
        _request_scoped.reset(token)
    await dispatcher.wait_for_pending()

    assert seen == [None]
