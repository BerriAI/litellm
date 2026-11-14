import asyncio
import inspect
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from operator import itemgetter
from types import TracebackType
from typing import Final, Protocol, TypeAlias, TypeVar

from typing_extensions import Self

from litellm.integrations.custom_logger_v2.call import CallContext, CallFacts
from litellm.integrations.custom_logger_v2.enforcement import ContractViolation, Enforced, Policy, Unchecked
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import settle_reads
from litellm.integrations.custom_logger_v2.hooks import HOOK_SPECS
from litellm.integrations.custom_logger_v2.manifest import LegacyBinding
from litellm.integrations.custom_logger_v2.offpath import InProcessSink, OffPathSink
from litellm.integrations.custom_logger_v2.outcomes import Abort, Outcome, Patch, Reroute, Respond
from litellm.integrations.custom_logger_v2.patches import apply_patch, apply_patches
from litellm.integrations.custom_logger_v2.planner import EMPTY_PLAN, Entry, HookPlan
from litellm.integrations.custom_logger_v2.registry import CallbackRegistry, RegistrySnapshot
from litellm.integrations.custom_logger_v2.selection import RequestSelection, is_selected

LegacyT: Final = TypeVar("LegacyT")

Between: TypeAlias = Callable[[Mapping[Field, object]], Awaitable[Mapping[Field, object]]]
SyncBinding: TypeAlias = Callable[[object, object], object]

SWALLOWING_HOOKS: Final = frozenset({Hook.response_headers, Hook.post_call_failure, Hook.post_deployment_failure})

_LOGGER: Final = logging.getLogger("LiteLLM")


class _Handler(Protocol):
    def __call__(self, view: object, ctx: CallContext[object], /) -> object: ...


@dataclass(frozen=True, slots=True)
class HookResult:
    fields: Mapping[Field, object]
    patches: tuple[Patch, ...] = ()
    outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class _Chain:
    fields: Mapping[Field, object]
    patches: tuple[tuple[int, Patch], ...] = ()
    reroute: Reroute | None = None
    terminal: Abort | Respond | None = None

    @property
    def ordered_patches(self) -> tuple[Patch, ...]:
        return tuple(patch for _, patch in sorted(self.patches, key=itemgetter(0)))


def _handler(logger: object, name: str) -> _Handler:
    found: Final[object] = getattr(logger, name, None)
    if not callable(found):
        raise ContractViolation(f"{name} is not callable")
    return found


async def _awaited(returned: object) -> object:
    return await returned if inspect.isawaitable(returned) else returned


def _contained(hook: Hook, entry: Entry, result: object) -> bool:
    if entry.subscription.legacy or HOOK_SPECS[hook].request_path:
        return False
    match result:
        case Abort() | Respond():
            return True
        case ContractViolation():
            return False
        case Exception():
            return True
        case _:
            return False


def _decides(hook: Hook, entry: Entry, result: object) -> bool:
    if _contained(hook, entry, result):
        return False
    match result:
        case Abort() | Respond():
            return True
        case ContractViolation():
            return True
        case Exception():
            return hook not in SWALLOWING_HOOKS
        case BaseException():
            return True
        case _:
            return False


def _loop_running() -> bool:
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@contextmanager
def _private_loop() -> Generator[asyncio.AbstractEventLoop]:
    loop: Final = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()


def _fatal(error: BaseException) -> bool:
    return isinstance(error, ContractViolation) or not isinstance(error, Exception)


def _split(
    stages: tuple[tuple[Entry, ...], ...],
) -> tuple[tuple[tuple[Entry, ...], ...], tuple[tuple[Entry, ...], ...]]:
    if stages and stages[-1][0].subscription.parallel:
        return stages[:-1], stages[-1:]
    return stages, ()


def _legacy_start(
    entry: Entry, binding: Callable[[object, object], LegacyT] | None, fields: Mapping[Field, object]
) -> LegacyT | None:
    call: Final = entry.manifest.legacy_call
    if call is None or binding is None:
        raise ContractViolation(
            f"{entry.manifest.selection.name} is a legacy entry, so the hook point must pass a binding"
        )
    return call(binding, fields)


async def _run_legacy(entry: Entry, binding: LegacyBinding | None, fields: Mapping[Field, object]) -> object:
    started: Final = _legacy_start(entry, binding, fields)
    return None if started is None else await started


async def _gather_started(started: tuple[Awaitable[object] | None, ...], fail_fast: bool) -> tuple[object, ...]:
    pending: Final = tuple(run for run in started if run is not None)
    if not pending:
        return tuple(None for _ in started)
    done: Final = iter(
        await asyncio.gather(*pending) if fail_fast else await asyncio.gather(*pending, return_exceptions=True)
    )
    return tuple(None if run is None else next(done) for run in started)


async def _settle(run: Awaitable[object], fail_fast: bool) -> object:
    if fail_fast:
        return await run
    try:
        return await run
    except BaseException as error:  # noqa: BLE001  # mirrors gather(return_exceptions=True) for a one-entry stage
        return error


def _fail_fast(hook: Hook, stage: tuple[Entry, ...]) -> bool:
    return HOOK_SPECS[hook].fail_fast_parallel and all(entry.subscription.parallel for entry in stage)


async def run_legacy_plan(
    hook: Hook,
    plan: HookPlan,
    fields: Mapping[Field, object],
    binding: LegacyBinding,
    between: Between | None = None,
) -> Mapping[Field, object]:
    serial, parallel = _split(plan.stages)
    current: Mapping[Field, object] = fields  # rebind-ok: each legacy entry may replace the kwargs
    for stage in serial:
        for entry in stage:
            result = await _run_legacy(entry, binding, current)
            if isinstance(result, Patch):
                current = apply_patch(current, result)
    if between is not None:
        current = await between(current)
    for stage in parallel:
        started = tuple(_legacy_start(entry, binding, current) for entry in stage)
        results = await _gather_started(started, _fail_fast(hook, stage))
        failure = next((result for result in results if isinstance(result, BaseException)), None)
        if failure is not None:
            raise failure
    for entry in plan.inline:
        _ = await _run_legacy(entry, binding, current)
    return current


def run_legacy_plan_sync(
    plan: HookPlan, fields: Mapping[Field, object], binding: SyncBinding
) -> Mapping[Field, object]:
    staged: Final = tuple(chain.from_iterable(plan.stages))
    if any(entry.subscription.parallel for entry in staged):
        raise ContractViolation("a synchronous hook cannot run parallel legacy entries")
    current: Mapping[Field, object] = fields  # rebind-ok: each legacy entry may replace the kwargs
    for entry in (*staged, *plan.inline):
        result = _legacy_start(entry, binding, current)
        if isinstance(result, Patch):
            current = apply_patch(current, result)
    return current


class Dispatcher:
    def __init__(
        self,
        registry: CallbackRegistry,
        sink: OffPathSink | None = None,
        enforce: bool = False,
    ) -> None:
        self.registry: Final = registry
        self.sink: Final[OffPathSink] = sink or InProcessSink()
        self.policy: Final[Policy] = Enforced() if enforce else Unchecked()
        self.failures: Final[Counter[tuple[str, Hook]]] = Counter()  # mutable-ok: process-wide failure counts

    def open_scope(self, selection: RequestSelection) -> "RequestScope":
        return RequestScope(self, self.registry.snapshot, selection)

    def record_failure(self, entry: Entry, hook: Hook, error: BaseException | Abort | Respond) -> None:
        self.failures[(entry.manifest.selection.name, hook)] += 1
        _LOGGER.warning("custom logger v2 %s failed on %s: %r", entry.manifest.selection.name, hook.value, error)


class RequestScope:
    def __init__(self, dispatcher: Dispatcher, snapshot: RegistrySnapshot, selection: RequestSelection) -> None:
        self._dispatcher: Final = dispatcher
        self._snapshot: Final = snapshot
        self._selection: Final = selection
        self._states: dict[int, object] = {}  # mutable-ok: request_state instances are created lazily per logger
        self.reroute: Reroute | None = None

    def __enter__(self) -> Self:
        return self

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:  # mutable-ok: copy.deepcopy's memo protocol
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self._states.clear()

    def selects(self, entry: Entry) -> bool:
        return is_selected(entry.manifest.selection, entry.activation, self._selection)

    def plan(self, hook: Hook) -> HookPlan:
        entries: Final = self._snapshot.entries[hook]
        if not entries:
            return EMPTY_PLAN
        selected: Final = tuple(index for index, entry in enumerate(entries) if self.selects(entry))
        return self._snapshot.plan(hook, selected) if selected else EMPTY_PLAN

    def reads(self, hook: Hook) -> frozenset[Field]:
        return self.plan(hook).reads

    async def run_hook(
        self,
        hook: Hook,
        fields: Mapping[Field, object],
        call: CallFacts | None,
    ) -> HookResult:
        plan: Final = self.plan(hook)
        if plan is EMPTY_PLAN:
            return HookResult(fields=fields)
        chain: Final = await self._run_stages(hook, plan.stages, _Chain(fields=fields), call, None)
        patches: Final = chain.ordered_patches
        if chain.terminal is not None:
            return HookResult(fields=chain.fields, patches=patches, outcome=chain.terminal)
        await self._run_awaited(hook, plan.awaited, chain.fields, call)
        self._submit_off_path(hook, plan.off_path, chain.fields, call)
        if chain.reroute is not None:
            self.reroute = chain.reroute
        return HookResult(fields=chain.fields, patches=patches, outcome=chain.reroute)

    async def run_plan(
        self,
        hook: Hook,
        plan: HookPlan,
        fields: Mapping[Field, object],
        call: CallFacts | None,
        binding: LegacyBinding,
        between: Between | None = None,
    ) -> HookResult:
        serial, parallel = _split(plan.stages)
        chain: Final = await self._run_stages(hook, serial, _Chain(fields=fields), call, binding)
        if chain.terminal is not None:
            return HookResult(fields=chain.fields, patches=chain.ordered_patches, outcome=chain.terminal)
        joined: Final = (
            chain
            if between is None
            else _Chain(fields=await between(chain.fields), patches=chain.patches, reroute=chain.reroute)
        )
        done: Final = await self._run_stages(hook, parallel, joined, call, binding)
        if done.terminal is not None:
            return HookResult(fields=done.fields, patches=done.ordered_patches, outcome=done.terminal)
        for entry in plan.inline:
            _ = await _run_legacy(entry, binding, done.fields)
        await self._run_awaited(hook, plan.awaited, done.fields, call)
        self._submit_off_path(hook, plan.off_path, done.fields, call)
        if done.reroute is not None:
            self.reroute = done.reroute
        return HookResult(fields=done.fields, patches=done.ordered_patches, outcome=done.reroute)

    def run_plan_sync(
        self,
        hook: Hook,
        plan: HookPlan,
        fields: Mapping[Field, object],
        call: CallFacts | None,
        binding: SyncBinding,
    ) -> Mapping[Field, object]:
        staged: Final = tuple(chain.from_iterable(plan.stages))
        if not plan.awaited and all(entry.subscription.legacy for entry in staged):
            result: Final = run_legacy_plan_sync(plan, fields, binding)
            self._submit_off_path(hook, plan.off_path, result, call)
            return result
        if any(entry.subscription.parallel for entry in staged):
            raise ContractViolation("a synchronous hook cannot run parallel legacy entries")
        blocking: Final = not _loop_running()
        with _private_loop() if blocking else nullcontext() as loop:
            ran: Final = reduce(
                lambda done, entry: self._run_sync_entry(hook, entry, done, call, binding, loop),
                (*staged, *plan.inline),
                _Chain(fields=fields),
            )
            if loop is not None and plan.awaited:
                awaited: Final = plan.awaited
                self._swallow(
                    hook,
                    awaited,
                    loop.run_until_complete(self._gather(hook, awaited, ran.fields, call, self._states_of(awaited))),
                )
        deferred: Final = tuple(entry for entry in staged if not (blocking or entry.subscription.legacy))
        self._submit_off_path(hook, (*deferred, *(() if blocking else plan.awaited), *plan.off_path), ran.fields, call)
        return ran.fields

    def _run_sync_entry(
        self,
        hook: Hook,
        entry: Entry,
        done: _Chain,
        call: CallFacts | None,
        binding: SyncBinding,
        loop: asyncio.AbstractEventLoop | None,
    ) -> _Chain:
        if entry.subscription.legacy:
            return self._fold(hook, (entry,), (_legacy_start(entry, binding, done.fields),), done)
        if loop is None:
            return done
        settled: Final = loop.run_until_complete(
            _gather_started((self._invoke(hook, entry, done.fields, call, self._state(entry)),), False)
        )
        folded: Final = self._fold(hook, (entry,), settled, done)
        if folded.terminal is not None:
            raise ContractViolation(f"{hook.value} runs synchronously here, so it cannot end the hook")
        return folded

    def _state(self, entry: Entry) -> object:
        factory: Final = entry.manifest.request_state
        if factory is None:
            return None
        key: Final = id(entry.manifest.logger)
        if key not in self._states:
            self._states[key] = factory()
        return self._states[key]

    def _states_of(self, entries: Sequence[Entry]) -> tuple[object, ...]:
        return tuple(self._state(entry) for entry in entries)

    async def _invoke(
        self, hook: Hook, entry: Entry, fields: Mapping[Field, object], call: CallFacts | None, state: object
    ) -> object:
        policy: Final = self._dispatcher.policy
        handler: Final = _handler(entry.manifest.logger, entry.subscription.handler)
        context: Final = CallContext[object](hook=hook, call=call, state=state)
        view: Final = policy.view(entry.subscription, fields)
        result: Final = (
            await _awaited(handler(view, context))
            if inspect.iscoroutinefunction(handler)
            else await asyncio.to_thread(handler, view, context)
        )
        policy.check(entry.subscription, result)
        return result

    def _start(
        self,
        hook: Hook,
        entry: Entry,
        fields: Mapping[Field, object],
        call: CallFacts | None,
        state: object,
        binding: LegacyBinding | None,
    ) -> Awaitable[object] | None:
        if entry.subscription.legacy:
            return _legacy_start(entry, binding, fields)
        return self._invoke(hook, entry, fields, call, state)

    async def _gather(
        self,
        hook: Hook,
        entries: Sequence[Entry],
        fields: Mapping[Field, object],
        call: CallFacts | None,
        states: tuple[object, ...],
        binding: LegacyBinding | None = None,
    ) -> tuple[object, ...]:
        if len(entries) == 1 and entries[0].subscription.legacy:
            return (await _run_legacy(entries[0], binding, fields),)
        started: Final = tuple(
            self._start(hook, entry, fields, call, state, binding) for entry, state in zip(entries, states, strict=True)
        )
        return await _gather_started(started, _fail_fast(hook, tuple(entries)))

    async def _run_stage(
        self,
        hook: Hook,
        stage: tuple[Entry, ...],
        fields: Mapping[Field, object],
        call: CallFacts | None,
        binding: LegacyBinding | None,
    ) -> tuple[object, ...]:
        match stage:
            case (only,) if not only.subscription.legacy:
                return (
                    await _settle(self._invoke(hook, only, fields, call, self._state(only)), _fail_fast(hook, stage)),
                )
            case _:
                return await self._gather(hook, stage, fields, call, self._states_of(stage), binding)

    async def _run_stages(
        self,
        hook: Hook,
        stages: tuple[tuple[Entry, ...], ...],
        chain: _Chain,
        call: CallFacts | None,
        binding: LegacyBinding | None,
    ) -> _Chain:
        if not stages:
            return chain
        stage: Final = stages[0]
        results: Final = await self._run_stage(hook, stage, chain.fields, call, binding)
        folded: Final = self._fold(hook, stage, results, chain)
        if folded.terminal is not None:
            return folded
        return await self._run_stages(hook, stages[1:], folded, call, binding)

    def _fold(self, hook: Hook, stage: tuple[Entry, ...], results: tuple[object, ...], chain: _Chain) -> _Chain:
        pairs: Final = tuple(zip(stage, results, strict=True))
        decisive: Final = next((pair for pair in pairs if _decides(hook, pair[0], pair[1])), None)
        self._count_failures(hook, tuple(pair for pair in pairs if pair is not decisive))
        match decisive:
            case (_, BaseException() as error):
                raise error
            case (_, Abort() | Respond() as outcome):
                return _Chain(fields=chain.fields, patches=chain.patches, reroute=chain.reroute, terminal=outcome)
            case _:
                return self._merge(pairs, chain)

    def _merge(self, pairs: tuple[tuple[Entry, object], ...], chain: _Chain) -> _Chain:
        stage_patches: Final = tuple((entry.position, result) for entry, result in pairs if isinstance(result, Patch))
        reroutes: Final = tuple(result for _, result in pairs if isinstance(result, Reroute))
        return _Chain(
            fields=apply_patches(chain.fields, (patch for _, patch in stage_patches)),
            patches=(*chain.patches, *stage_patches),
            reroute=chain.reroute or next(iter(reroutes), None),
        )

    def _count_failures(self, hook: Hook, pairs: Sequence[tuple[Entry, object]]) -> None:
        for entry, result in pairs:
            match result:
                case BaseException() if not entry.subscription.legacy:
                    self._dispatcher.record_failure(entry, hook, result)
                case Abort() | Respond() if _contained(hook, entry, result):
                    self._dispatcher.record_failure(entry, hook, result)
                case _:
                    pass

    def _swallow(self, hook: Hook, entries: Sequence[Entry], results: tuple[object, ...]) -> None:
        pairs: Final = tuple(zip(entries, results, strict=True))
        fatal: Final = next(
            (result for _, result in pairs if isinstance(result, BaseException) and _fatal(result)),
            None,
        )
        self._count_failures(hook, tuple(pair for pair in pairs if pair[1] is not fatal))
        if fatal is not None:
            raise fatal

    async def _run_awaited(
        self, hook: Hook, entries: tuple[Entry, ...], fields: Mapping[Field, object], call: CallFacts | None
    ) -> None:
        if entries:
            self._swallow(hook, entries, await self._gather(hook, entries, fields, call, self._states_of(entries)))

    def _submit_off_path(
        self, hook: Hook, entries: tuple[Entry, ...], fields: Mapping[Field, object], call: CallFacts | None
    ) -> None:
        if entries:
            settle_reads(fields, chain.from_iterable(entry.subscription.reads for entry in entries))
            self._dispatcher.sink.submit(self._deliver_off_path(hook, entries, fields, call, self._states_of(entries)))

    async def _deliver_off_path(
        self,
        hook: Hook,
        entries: tuple[Entry, ...],
        fields: Mapping[Field, object],
        call: CallFacts | None,
        states: tuple[object, ...],
    ) -> None:
        self._swallow(hook, entries, await self._gather(hook, entries, fields, call, states))
