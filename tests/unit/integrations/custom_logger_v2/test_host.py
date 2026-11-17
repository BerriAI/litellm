import asyncio
import threading
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, field
from typing import Final

import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import (
    Abort,
    CallContext,
    CustomLoggerV2,
    Field,
    Hook,
    Patch,
    on,
)
from litellm.integrations.custom_logger_v2.dispatcher import RequestScope
from litellm.integrations.custom_logger_v2.host import (
    V2Outcome,
    bind_scope,
    current_scope,
    request_scope,
    reset_host_for_tests,
)
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    KwargsFields,
    clear_subscribers_cache_for_tests,
    hook_subscribers,
    kwargs_after,
    run_subscribers,
)

F = Field
ABORT = Abort(status=403, message="blocked")


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    yield
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()


@dataclass
class Seen:
    states: list[object] = field(default_factory=list)


class Masker(CustomLoggerV2):
    @dataclass
    class request_state:
        hits: int = 0

    def __init__(self, seen: Seen, outcome: Abort | None = None) -> None:
        self.seen = seen
        self.outcome = outcome

    @on(Hook.pre_call, reads={F.messages}, writes={F.messages}, may_abort=True)
    async def mask(self, view: object, ctx: CallContext[object]) -> Patch | Abort:
        self.seen.states.append(ctx.state)
        return self.outcome or Patch().set(F.messages, ("masked",))


class PreCallOnly(CustomLogger):
    async def async_pre_call_hook(self, *args: object, **kwargs: object) -> None:
        return None


def _recording(ran: list[object]) -> Callable[[object, object], Awaitable[None]]:
    def start(logger: object, data: object) -> Awaitable[None]:
        async def run() -> None:
            ran.append(logger)

        return run()

    return start


async def _run_pre_call(callbacks: list[object], ran: list[object], scope: RequestScope | None = None) -> object:
    subscribers: Final = hook_subscribers(Hook.pre_call, callbacks)
    fields: Final = KwargsFields({"messages": ("raw",)}, PROXY_REQUEST_KEYS)
    return kwargs_after(await run_subscribers(subscribers, fields, _recording(ran), scope=scope))


async def test_a_v2_callback_abort_raises_v2_callback_outcome_and_skips_the_legacy_entry_after_it() -> None:
    ran: Final[list[object]] = []
    callbacks: Final[list[object]] = [Masker(Seen(), ABORT), PreCallOnly()]

    with pytest.raises(V2Outcome) as raised:
        await _run_pre_call(callbacks, ran)

    assert raised.value.outcome is ABORT
    assert ran == []


async def test_request_scope_is_current_inside_and_shares_state_across_hook_runs() -> None:
    seen: Final = Seen()
    ran: Final[list[object]] = []
    callbacks: Final[list[object]] = [Masker(seen), PreCallOnly()]

    with request_scope() as scope:
        assert current_scope() is scope
        await _run_pre_call(callbacks, ran)
        await _run_pre_call(callbacks, ran)
    await _run_pre_call(callbacks, ran)

    assert current_scope() is None
    assert len(ran) == 3
    assert seen.states[0] is seen.states[1]
    assert seen.states[2] is not seen.states[0]


async def test_an_explicit_scope_is_used_where_no_scope_is_current() -> None:
    seen: Final = Seen()
    callbacks: Final[list[object]] = [Masker(seen)]

    with request_scope() as scope:
        pass
    await _run_pre_call(callbacks, [], scope=scope)
    await _run_pre_call(callbacks, [], scope=scope)

    assert seen.states[0] is seen.states[1]


async def test_bind_scope_carries_a_scope_into_a_thread_that_did_not_inherit_it() -> None:
    found: list[object] = []

    def worker(scope: object) -> None:
        found.append(current_scope())
        with request_scope() as other, bind_scope(other):
            found.append(current_scope() is other)

    with request_scope() as scope:
        thread = threading.Thread(target=worker, args=(scope,))
        thread.start()
        await asyncio.to_thread(thread.join)
        with bind_scope(scope):
            assert current_scope() is scope

    assert found == [None, True]
