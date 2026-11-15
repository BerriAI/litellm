import asyncio
import gc
import threading
import weakref
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Set
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType

import pytest

from litellm.integrations.custom_logger_v2 import (
    Abort,
    Activation,
    CallbackManifest,
    CallbackRegistry,
    CallContext,
    ContractViolation,
    CustomLoggerV2,
    Dispatcher,
    Field,
    Hook,
    Patch,
    RequestSelection,
    Reroute,
    Respond,
    Selection,
    UndeclaredReadError,
    build_manifest,
    on,
)

F = Field
Behaviour = Callable[[object, CallContext[object]], Awaitable[object]]


@dataclass
class Calls:
    calls: int = 0


async def _nothing(view: object, ctx: CallContext[object]) -> object:
    return None


def v2_callback(
    name: str,
    hook: Hook = Hook.pre_call,
    behaviour: Behaviour = _nothing,
    *,
    reads: Set[Field] = frozenset(),
    writes: Set[Field] = frozenset(),
    may_abort: bool = False,
    parallel: bool | None = None,
    awaited: bool = False,
    selection: Selection | None = None,
    state: type | None = None,
) -> CallbackManifest:
    async def handle(self: CustomLoggerV2, view: object, ctx: CallContext[object]) -> object:
        return await behaviour(view, ctx)

    decorated = on(hook, reads=reads, writes=writes, may_abort=may_abort, parallel=parallel, awaited=awaited)(handle)
    namespace = {"handle": decorated, "selection": selection or Selection(name), "request_state": state}
    return build_manifest(type(name, (CustomLoggerV2,), namespace)())


def returning(result: object, journal: list[str] | None = None, name: str = "") -> Behaviour:
    async def behaviour(view: object, ctx: CallContext[object]) -> object:
        if journal is not None:
            journal.append(f"{name}+")
            await asyncio.sleep(0)
            journal.append(f"{name}-")
        return result

    return behaviour


class RecordingSink:
    def __init__(self) -> None:
        self.deliveries: list[Coroutine[object, object, None]] = []

    def submit(self, delivery: Coroutine[object, object, None]) -> None:
        self.deliveries.append(delivery)

    async def run_all(self) -> None:
        for delivery in self.deliveries:
            await delivery
        self.deliveries.clear()


def dispatcher(*manifests: CallbackManifest, sink: RecordingSink | None = None, enforce: bool = False) -> Dispatcher:
    return Dispatcher(CallbackRegistry(manifests), sink=sink or RecordingSink(), enforce=enforce)


MESSAGES = (MappingProxyType({"role": "user", "content": "my card is 4111"}),)
FIELDS: Mapping[Field, object] = MappingProxyType({F.messages: MESSAGES, F.metadata: MappingProxyType({"a": 1})})


async def test_v2_callbacks_on_other_hooks_leave_this_hook_a_passthrough() -> None:
    elsewhere = v2_callback("elsewhere", Hook.post_call_success, returning(Patch()), writes={F.response})

    result = await dispatcher(elsewhere).open_scope(RequestSelection()).run_hook(Hook.pre_call, FIELDS, None)

    assert result.fields is FIELDS
    assert result.outcome is None
    assert result.patches == ()


async def test_design_doc_worked_example_runs_in_its_stages_and_audit_sees_the_final_messages() -> None:
    journal: list[str] = []
    masked = (MappingProxyType({"role": "user", "content": "my card is [CARD]"}),)
    scanned = (MappingProxyType({"role": "user", "content": "[SCANNED]"}),)
    seen_by_audit: list[object] = []

    async def audit(view: object, ctx: CallContext[object]) -> None:
        seen_by_audit.append(getattr(view, "messages"))

    async def secret_scan(view: object, ctx: CallContext[object]) -> Patch:
        journal.append("secret_scan+")
        assert getattr(view, "messages") == masked
        return Patch().set(F.messages, scanned)

    sink = RecordingSink()
    scope = dispatcher(
        v2_callback(
            "pii_mask",
            behaviour=returning(Patch().set(F.messages, masked), journal, "pii_mask"),
            reads={F.messages},
            writes={F.messages},
            may_abort=True,
        ),
        v2_callback(
            "tagger",
            behaviour=returning(Patch().merge(F.metadata, {"tag": "x"}), journal, "tagger"),
            writes={F.metadata},
        ),
        v2_callback("limiter", behaviour=returning(None, journal, "limiter"), reads={F.key_identity}, may_abort=True),
        v2_callback("secret_scan", behaviour=secret_scan, reads={F.messages}, writes={F.messages}, may_abort=True),
        v2_callback("audit", behaviour=audit, reads={F.messages}),
        sink=sink,
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert journal == ["pii_mask+", "tagger+", "pii_mask-", "tagger-", "limiter+", "limiter-", "secret_scan+"]
    assert result.fields[F.messages] == scanned
    assert result.fields[F.metadata] == {"a": 1, "tag": "x"}
    assert seen_by_audit == []
    await sink.run_all()
    assert seen_by_audit == [scanned]


async def test_patches_come_back_in_list_order_even_when_stages_run_out_of_it() -> None:
    first = Patch().set(F.messages, ())
    second = Patch().set(F.optional_params, {})
    third = Patch().merge(F.metadata, {"b": 2})
    scope = dispatcher(
        v2_callback("first", behaviour=returning(first), writes={F.messages}, may_abort=True),
        v2_callback("second", behaviour=returning(second), writes={F.optional_params}, may_abort=True),
        v2_callback("third", behaviour=returning(third), writes={F.metadata}),
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert result.patches == (first, second, third)


async def test_a_later_writer_never_shows_an_earlier_reader_its_patch() -> None:
    seen: list[object] = []

    async def scanner(view: object, ctx: CallContext[object]) -> None:
        seen.append(getattr(view, "messages"))
        await asyncio.sleep(0)

    scope = dispatcher(
        v2_callback("tagger", behaviour=returning(Patch().merge(F.metadata, {"t": 1})), writes={F.metadata}),
        v2_callback("scanner", behaviour=scanner, reads={F.messages}, may_abort=True),
        v2_callback("rewriter", behaviour=returning(Patch().set(F.messages, ())), writes={F.messages}),
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert seen == [MESSAGES]
    assert result.fields[F.messages] == ()


@pytest.mark.parametrize("outcome", [Abort(status=400, message="blocked"), Respond(response={"ok": True})])
async def test_terminal_outcome_stops_new_starts_and_discards_in_flight_patches(
    outcome: Abort | Respond,
) -> None:
    journal: list[str] = []
    scope = dispatcher(
        v2_callback("guard", behaviour=returning(outcome, journal, "guard"), reads={F.messages}, may_abort=True),
        v2_callback(
            "tagger", behaviour=returning(Patch().merge(F.metadata, {"t": 1}), journal, "tagger"), writes={F.metadata}
        ),
        v2_callback("later", behaviour=returning(None, journal, "later"), may_abort=True),
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert result.outcome is outcome
    assert result.patches == ()
    assert journal == ["guard+", "tagger+", "guard-", "tagger-"]


async def test_patches_from_finished_stages_survive_a_later_abort() -> None:
    patch = Patch().set(F.messages, ())
    abort = Abort(status=429, message="slow down")
    scope = dispatcher(
        v2_callback("masker", behaviour=returning(patch), writes={F.messages}, may_abort=True),
        v2_callback("limiter", behaviour=returning(abort), may_abort=True),
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert result.outcome is abort
    assert result.patches == (patch,)


async def test_reroute_is_recorded_and_the_rest_of_the_list_still_runs() -> None:
    journal: list[str] = []

    scope = dispatcher(
        v2_callback("router", behaviour=returning(Reroute(model="safe-model")), may_abort=True),
        v2_callback("second_router", behaviour=returning(Reroute(model="other")), may_abort=True),
        v2_callback("scanner", behaviour=returning(None, journal, "scanner"), may_abort=True),
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert result.outcome == Reroute(model="safe-model")
    assert scope.reroute == Reroute(model="safe-model")
    assert journal == ["scanner+", "scanner-"]


async def test_a_later_abort_beats_an_earlier_reroute() -> None:
    abort = Abort(status=400, message="blocked")
    scope = dispatcher(
        v2_callback("router", behaviour=returning(Reroute(model="safe-model")), may_abort=True),
        v2_callback("guard", behaviour=returning(abort), may_abort=True),
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert result.outcome is abort
    assert scope.reroute is None


async def test_unexpected_exception_from_a_blocking_v2_callback_fails_the_request_closed() -> None:
    error = ValueError("vendor down")
    later = Calls()

    async def broken(view: object, ctx: CallContext[object]) -> None:
        raise error

    async def counted(view: object, ctx: CallContext[object]) -> None:
        later.calls += 1

    scope = dispatcher(
        v2_callback("guard", behaviour=broken, may_abort=True),
        v2_callback("later", behaviour=counted, may_abort=True),
    ).open_scope(RequestSelection())

    with pytest.raises(ValueError, match="vendor down") as raised:
        await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert raised.value is error
    assert later.calls == 0


async def test_swallowing_hooks_count_the_failure_and_keep_the_other_patches() -> None:
    async def broken(view: object, ctx: CallContext[object]) -> None:
        raise ValueError("header lookup failed")

    patch = Patch().merge(F.response_headers, {"x-limit": "10"})
    owner = dispatcher(
        v2_callback("broken", Hook.response_headers, broken, writes={F.response_headers}),
        v2_callback("limits", Hook.response_headers, returning(patch), writes={F.response_headers}),
    )

    result = await owner.open_scope(RequestSelection()).run_hook(Hook.response_headers, {}, None)

    assert result.patches == (patch,)
    assert owner.failures[("broken", Hook.response_headers)] == 1


async def test_awaited_readers_run_after_blocking_work_in_parallel_and_never_fail_the_request() -> None:
    journal: list[str] = []

    async def failing(view: object, ctx: CallContext[object]) -> None:
        journal.append("failing+")
        await asyncio.sleep(0)
        raise RuntimeError("redis down")

    owner = dispatcher(
        v2_callback("release", behaviour=returning(None, journal, "release"), reads={F.key_identity}, awaited=True),
        v2_callback("failing", behaviour=failing, awaited=True),
        v2_callback("writer", behaviour=returning(Patch().set(F.messages, ()), journal, "writer"), writes={F.messages}),
    )

    result = await owner.open_scope(RequestSelection()).run_hook(Hook.pre_call, FIELDS, None)

    assert journal == ["writer+", "writer-", "release+", "failing+", "release-"]
    assert result.fields[F.messages] == ()
    assert owner.failures[("failing", Hook.pre_call)] == 1


async def test_during_call_abortables_run_in_parallel_and_list_order_picks_the_winner() -> None:
    journal: list[str] = []
    first = Abort(status=400, message="first")
    second = Abort(status=400, message="second")
    scope = dispatcher(
        v2_callback("slow", Hook.during_call, returning(None, journal, "slow"), may_abort=True),
        v2_callback("first", Hook.during_call, returning(first, journal, "first"), may_abort=True),
        v2_callback("second", Hook.during_call, returning(second, journal, "second"), may_abort=True),
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.during_call, FIELDS, None)

    assert journal == ["slow+", "first+", "second+", "slow-", "first-", "second-"]
    assert result.outcome is first


async def test_parallel_pre_call_checks_see_the_patches_of_the_serial_chain() -> None:
    seen: list[object] = []

    async def check(view: object, ctx: CallContext[object]) -> None:
        seen.append(getattr(view, "messages"))

    scope = dispatcher(
        v2_callback("check", behaviour=check, reads={F.messages}, may_abort=True, parallel=True),
        v2_callback("masker", behaviour=returning(Patch().set(F.messages, ())), writes={F.messages}, may_abort=True),
    ).open_scope(RequestSelection())

    await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert seen == [()]


REQUEST_STASH: ContextVar[object] = ContextVar("request_stash", default=None)


async def test_a_blocking_v2_callback_sets_context_the_next_stage_and_the_caller_see() -> None:
    stash = object()
    seen: list[object] = []

    async def claim(view: object, ctx: CallContext[object]) -> None:
        _ = REQUEST_STASH.set(stash)

    async def look(view: object, ctx: CallContext[object]) -> None:
        seen.append(REQUEST_STASH.get())

    scope = dispatcher(
        v2_callback("claimer", behaviour=claim, may_abort=True),
        v2_callback("looker", behaviour=look, may_abort=True),
    ).open_scope(RequestSelection())

    await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert (seen, REQUEST_STASH.get()) == ([stash], stash)


async def test_selection_mask_removes_opted_out_v2_callbacks_and_their_reads() -> None:
    called: list[str] = []

    async def note(view: object, ctx: CallContext[object]) -> None:
        called.append("guard")

    owner = dispatcher(
        v2_callback(
            "guard",
            behaviour=note,
            reads={F.messages},
            may_abort=True,
            selection=Selection("guard", activation=Activation.opt_in),
        ),
        v2_callback("limiter", reads={F.key_identity}, may_abort=True),
    )

    unrequested = owner.open_scope(RequestSelection())
    requested = owner.open_scope(RequestSelection(requested=frozenset({"guard"})))

    assert unrequested.reads(Hook.pre_call) == {F.key_identity}
    assert requested.reads(Hook.pre_call) == {F.key_identity, F.messages}
    await unrequested.run_hook(Hook.pre_call, FIELDS, None)
    assert called == []
    await requested.run_hook(Hook.pre_call, FIELDS, None)
    assert called == ["guard"]


async def test_a_subscription_activation_override_beats_the_instance_activation() -> None:
    class Masker(CustomLoggerV2):
        selection = Selection("masker", activation=Activation.opt_in)
        activation_overrides = MappingProxyType({Hook.log_prepare: Activation.required})

        @on(Hook.pre_call, reads={F.messages}, may_abort=True)
        async def check(self, view: object, ctx: CallContext[object]) -> None: ...

        @on(Hook.log_prepare, writes={F.standard_logging_payload})
        async def redact(self, view: object, ctx: CallContext[object]) -> None: ...

    scope = dispatcher(build_manifest(Masker())).open_scope(RequestSelection(admin_opt_outs=frozenset({"masker"})))

    assert scope.plan(Hook.pre_call).stages == ()
    assert len(scope.plan(Hook.log_prepare).stages) == 1


async def test_off_path_v2_callbacks_are_delivered_once_per_hook_with_their_request_state() -> None:
    @dataclass
    class Spans:
        open: list[str] = field(default_factory=list)

    states: list[object] = []

    async def remember(view: object, ctx: CallContext[object]) -> None:
        states.append(ctx.state)

    sink = RecordingSink()
    scope = dispatcher(
        v2_callback("otel_pre", Hook.pre_call, remember, reads={F.messages}, state=Spans),
        v2_callback("otel_awaited", Hook.pre_call, remember, awaited=True, state=Spans),
        sink=sink,
    ).open_scope(RequestSelection())

    await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert len(sink.deliveries) == 1
    assert len(states) == 1
    assert isinstance(states[0], Spans)
    await sink.run_all()
    assert isinstance(states[1], Spans)
    assert states[1] is not states[0]


def test_sync_off_path_handler_runs_on_a_worker_thread() -> None:
    threads: list[int] = []

    class SyncLogger(CustomLoggerV2):
        @on(Hook.attempt_ended, reads={F.usage})
        def ended(self, view: object, ctx: CallContext[object]) -> None:
            threads.append(threading.get_ident())

    sink = RecordingSink()
    scope = dispatcher(build_manifest(SyncLogger()), sink=sink).open_scope(RequestSelection())

    async def run() -> None:
        await scope.run_hook(Hook.attempt_ended, {F.usage: 3}, None)
        await sink.run_all()

    asyncio.run(run())

    assert len(threads) == 1
    assert threads[0] != threading.get_ident()


async def test_request_state_is_shared_across_hooks_per_request_and_freed_on_close() -> None:
    @dataclass
    class Tokens:
        tokens: dict[str, str] = field(default_factory=dict)

    states: list[object] = []

    async def remember(view: object, ctx: CallContext[object]) -> None:
        states.append(ctx.state)

    class PiiMask(CustomLoggerV2):
        request_state = Tokens

        @on(Hook.pre_call, writes={F.messages})
        async def mask(self, view: object, ctx: CallContext[object]) -> None:
            await remember(view, ctx)

        @on(Hook.post_call_success, writes={F.response})
        async def unmask(self, view: object, ctx: CallContext[object]) -> None:
            await remember(view, ctx)

    owner = dispatcher(build_manifest(PiiMask()))
    with owner.open_scope(RequestSelection()) as first:
        await first.run_hook(Hook.pre_call, FIELDS, None)
        await first.run_hook(Hook.post_call_success, {}, None)
    with owner.open_scope(RequestSelection()) as second:
        await second.run_hook(Hook.pre_call, FIELDS, None)

    assert isinstance(states[0], Tokens)
    assert states[0] is states[1]
    assert states[2] is not states[0]
    freed = weakref.ref(states[0])
    states.clear()
    gc.collect()
    assert freed() is None


async def test_registry_change_reaches_new_scopes_and_leaves_open_ones_alone() -> None:
    registry = CallbackRegistry()
    owner = Dispatcher(registry, sink=RecordingSink())
    before = owner.open_scope(RequestSelection())
    late = v2_callback("late", reads={F.messages}, may_abort=True)

    registry.register(late)
    after = owner.open_scope(RequestSelection())

    assert before.plan(Hook.pre_call).stages == ()
    assert len(after.plan(Hook.pre_call).stages) == 1
    registry.unregister(late.logger)
    assert owner.open_scope(RequestSelection()).plan(Hook.pre_call).stages == ()


async def test_production_mode_applies_patches_without_checking_declarations() -> None:
    sneaky = Patch().set(F.model, "other")
    scope = dispatcher(v2_callback("sneaky", behaviour=returning(sneaky), writes={F.messages})).open_scope(
        RequestSelection()
    )

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert result.fields[F.model] == "other"


async def test_enforced_mode_restricts_views_to_declared_reads() -> None:
    async def peek(view: object, ctx: CallContext[object]) -> None:
        getattr(view, "metadata")

    scope = dispatcher(
        v2_callback("peek", behaviour=peek, reads={F.messages}, may_abort=True), enforce=True
    ).open_scope(RequestSelection())

    with pytest.raises(UndeclaredReadError):
        await scope.run_hook(Hook.pre_call, FIELDS, None)


@pytest.mark.parametrize(
    ("result", "may_abort", "hook"),
    [
        (Patch().set(F.model, "other"), False, Hook.pre_call),
        (Abort(status=400, message="no"), False, Hook.pre_call),
        (Reroute(model="other"), True, Hook.during_call),
        ("not an outcome", True, Hook.pre_call),
    ],
)
async def test_enforced_mode_rejects_results_outside_the_declaration(
    result: object, may_abort: bool, hook: Hook
) -> None:
    writes = frozenset({F.messages}) if hook is Hook.pre_call else frozenset()
    scope = dispatcher(
        v2_callback("liar", hook, returning(result), writes=writes, may_abort=may_abort, parallel=False), enforce=True
    ).open_scope(RequestSelection())

    with pytest.raises(ContractViolation):
        await scope.run_hook(hook, FIELDS, None)


async def test_enforced_mode_violations_escape_the_awaited_swallow() -> None:
    scope = dispatcher(
        v2_callback("awaited", behaviour=returning(Patch().set(F.messages, ())), awaited=True), enforce=True
    ).open_scope(RequestSelection())

    with pytest.raises(ContractViolation):
        await scope.run_hook(Hook.pre_call, FIELDS, None)


async def test_enforced_mode_accepts_declared_behaviour() -> None:
    async def masker(view: object, ctx: CallContext[object]) -> Patch:
        return Patch().set(F.messages, (*getattr(view, "messages"), "extra"))

    scope = dispatcher(
        v2_callback("masker", behaviour=masker, reads={F.messages}, writes={F.messages}, may_abort=True), enforce=True
    ).open_scope(RequestSelection())

    result = await scope.run_hook(Hook.pre_call, FIELDS, None)

    assert result.fields[F.messages] == (*MESSAGES, "extra")
