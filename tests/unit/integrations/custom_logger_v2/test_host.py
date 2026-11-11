import asyncio
import threading
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field

import pytest

from litellm.integrations.custom_logger_v2 import (
    Abort,
    CallContext,
    CustomLoggerV2,
    Field,
    Hook,
    InvalidDeclarationError,
    Patch,
    build_manifest,
    on,
)
from litellm.integrations.custom_logger_v2.host import (
    NativeOutcome,
    bind_scope,
    current_scope,
    has_natives,
    register_native,
    request_scope,
    reset_host_for_tests,
    run_legacy_hook,
    run_legacy_hook_sync,
)

F = Field
ABORT = Abort(status=403, message="blocked")


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    yield
    reset_host_for_tests()


def _unused_fields() -> Mapping[Field, object]:
    raise AssertionError("fields were built although no native is registered")


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


class Exporter(CustomLoggerV2):
    def __init__(self, done: threading.Event) -> None:
        self.done = done

    @on(Hook.attempt_ended, reads={F.usage})
    def ended(self, view: object, ctx: CallContext[object]) -> None:
        self.done.set()


async def test_async_passthrough_returns_the_legacy_value_without_scope_or_fields() -> None:
    value = object()
    received: list[tuple[Patch, ...]] = []

    async def legacy(patches: tuple[Patch, ...]) -> object:
        received.append(patches)
        return value

    result = await run_legacy_hook(Hook.pre_call, legacy, fields=_unused_fields)

    assert result is value
    assert received == [()]
    assert current_scope() is None


async def test_async_passthrough_propagates_the_legacy_exception_untouched() -> None:
    error = RuntimeError("today's guardrail raised")

    async def legacy(patches: tuple[Patch, ...]) -> None:
        raise error

    with pytest.raises(RuntimeError) as raised:
        await run_legacy_hook(Hook.pre_call, legacy, fields=_unused_fields)

    assert raised.value is error


def test_sync_passthrough_returns_the_legacy_value_and_propagates_its_exception() -> None:
    value = object()
    error = ValueError("sync logger raised")

    def fails(patches: tuple[Patch, ...]) -> object:
        raise error

    assert run_legacy_hook_sync(Hook.attempt_ended, lambda patches: value, fields=_unused_fields) is value
    with pytest.raises(ValueError) as raised:
        run_legacy_hook_sync(Hook.attempt_ended, fails, fields=_unused_fields)
    assert raised.value is error


def test_has_natives_is_true_only_for_hooks_a_registered_native_subscribes() -> None:
    assert not has_natives(Hook.pre_call)

    register_native(build_manifest(Masker(Seen())))

    assert has_natives(Hook.pre_call)
    assert not has_natives(Hook.post_call_success)


async def test_natives_run_before_legacy_and_hand_it_their_patches() -> None:
    register_native(build_manifest(Masker(Seen())))

    async def legacy(patches: tuple[Patch, ...]) -> tuple[Patch, ...]:
        return patches

    patches = await run_legacy_hook(Hook.pre_call, legacy, fields=lambda: {F.messages: ("raw",)})

    assert patches == (Patch().set(F.messages, ("masked",)),)


async def test_a_native_abort_raises_native_outcome_and_skips_legacy() -> None:
    register_native(build_manifest(Masker(Seen(), ABORT)))
    ran: list[bool] = []

    async def legacy(patches: tuple[Patch, ...]) -> None:
        ran.append(True)

    with pytest.raises(NativeOutcome) as raised:
        await run_legacy_hook(Hook.pre_call, legacy)

    assert raised.value.outcome is ABORT
    assert ran == []


async def test_request_scope_is_current_inside_and_shares_state_across_hook_runs() -> None:
    seen = Seen()
    register_native(build_manifest(Masker(seen)))

    async def legacy(patches: tuple[Patch, ...]) -> None:
        return None

    with request_scope() as scope:
        assert current_scope() is scope
        await run_legacy_hook(Hook.pre_call, legacy)
        await run_legacy_hook(Hook.pre_call, legacy)
    await run_legacy_hook(Hook.pre_call, legacy)

    assert current_scope() is None
    assert seen.states[0] is seen.states[1]
    assert seen.states[2] is not seen.states[0]


async def test_an_explicit_scope_is_used_where_no_scope_is_current() -> None:
    seen = Seen()
    register_native(build_manifest(Masker(seen)))

    async def legacy(patches: tuple[Patch, ...]) -> None:
        return None

    with request_scope() as scope:
        pass
    await run_legacy_hook(Hook.pre_call, legacy, scope=scope)
    await run_legacy_hook(Hook.pre_call, legacy, scope=scope)

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


def test_sync_sites_reject_blocking_natives_at_registration() -> None:
    class Prompter(CustomLoggerV2):
        @on(Hook.prompt_resolve, writes={F.messages})
        async def resolve(self, view: object, ctx: CallContext[object]) -> None: ...

    with pytest.raises(InvalidDeclarationError):
        register_native(build_manifest(Prompter()))
    assert not has_natives(Hook.prompt_resolve)


def test_sync_site_runs_legacy_then_delivers_off_path_natives_without_an_event_loop() -> None:
    done = threading.Event()
    register_native(build_manifest(Exporter(done)))

    result = run_legacy_hook_sync(Hook.attempt_ended, lambda patches: "logged", fields=lambda: {F.usage: 1})

    assert result == "logged"
    assert done.wait(timeout=30)
