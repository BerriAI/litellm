import asyncio
import inspect
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from types import TracebackType
from typing import Final, Generic, Protocol, TypeAlias, TypeVar

from typing_extensions import Self

from litellm.integrations.custom_logger_v2.call import CallContext, CallFacts
from litellm.integrations.custom_logger_v2.enforcement import ContractViolation, Enforced, Policy, Unchecked
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.offpath import InProcessSink, OffPathSink
from litellm.integrations.custom_logger_v2.outcomes import Abort, Outcome, Patch, Reroute, Respond
from litellm.integrations.custom_logger_v2.patches import apply_patches
from litellm.integrations.custom_logger_v2.planner import EMPTY_PLAN, Entry, HookPlan
from litellm.integrations.custom_logger_v2.registry import CallbackRegistry, RegistrySnapshot
from litellm.integrations.custom_logger_v2.selection import RequestSelection, is_selected

LegacyT: Final = TypeVar("LegacyT")

LegacyGroup: TypeAlias = Callable[[tuple[Patch, ...]], Awaitable[LegacyT]]

SWALLOWING_HOOKS: Final = frozenset({Hook.response_headers, Hook.post_call_failure})

_LOGGER: Final = logging.getLogger("LiteLLM")


class _Handler(Protocol):
    def __call__(self, view: object, ctx: CallContext[object], /) -> object: ...


@dataclass(frozen=True, slots=True)
class HookResult(Generic[LegacyT]):
    fields: Mapping[Field, object]
    patches: tuple[Patch, ...] = ()
    outcome: Outcome | None = None
    legacy: LegacyT | None = None


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


def _decides(hook: Hook, result: object) -> bool:
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


def _fatal(error: BaseException) -> bool:
    return isinstance(error, ContractViolation) or not isinstance(error, Exception)


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

    def record_failure(self, entry: Entry, hook: Hook, error: BaseException) -> None:
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

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self._states.clear()

    def plan(self, hook: Hook) -> HookPlan:
        entries: Final = self._snapshot.entries[hook]
        if not entries:
            return EMPTY_PLAN
        selected: Final = tuple(
            index
            for index, entry in enumerate(entries)
            if is_selected(entry.manifest.selection, entry.activation, self._selection)
        )
        return self._snapshot.plan(hook, selected) if selected else EMPTY_PLAN

    def reads(self, hook: Hook) -> frozenset[Field]:
        return self.plan(hook).reads

    async def run_hook(
        self,
        hook: Hook,
        fields: Mapping[Field, object],
        call: CallFacts | None,
        legacy: LegacyGroup[LegacyT] | None = None,
    ) -> HookResult[LegacyT]:
        plan: Final = self.plan(hook)
        if plan is EMPTY_PLAN:
            return HookResult(fields=fields, legacy=None if legacy is None else await legacy(()))
        chain: Final = await self._run_stages(hook, plan.stages, _Chain(fields=fields), call)
        patches: Final = chain.ordered_patches
        if chain.terminal is not None:
            return HookResult(fields=chain.fields, patches=patches, outcome=chain.terminal)
        legacy_result: Final = None if legacy is None else await legacy(patches)
        await self._run_awaited(hook, plan.awaited, chain.fields, call)
        if plan.off_path:
            self._dispatcher.sink.submit(self._deliver_off_path(hook, plan.off_path, chain.fields, call))
        if chain.reroute is not None:
            self.reroute = chain.reroute
        return HookResult(fields=chain.fields, patches=patches, outcome=chain.reroute, legacy=legacy_result)

    def run_sync_hook(
        self,
        hook: Hook,
        fields: Mapping[Field, object],
        call: CallFacts | None,
        legacy: Callable[[tuple[Patch, ...]], LegacyT],
    ) -> LegacyT:
        plan: Final = self.plan(hook)
        if plan.stages or plan.awaited:
            raise ContractViolation(f"{hook.value} runs synchronously here, so only off-path natives may subscribe")
        result: Final = legacy(())
        if plan.off_path:
            self._dispatcher.sink.submit(self._deliver_off_path(hook, plan.off_path, fields, call))
        return result

    def _state(self, entry: Entry) -> object:
        factory: Final = entry.manifest.request_state
        if factory is None:
            return None
        if entry.position not in self._states:
            self._states[entry.position] = factory()
        return self._states[entry.position]

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

    async def _gather(
        self, hook: Hook, entries: Sequence[Entry], fields: Mapping[Field, object], call: CallFacts | None, live: bool
    ) -> tuple[object, ...]:
        runs: Final = (
            self._invoke(hook, entry, fields, call, self._state(entry) if live else None) for entry in entries
        )
        return tuple(await asyncio.gather(*runs, return_exceptions=True))

    async def _run_stages(
        self, hook: Hook, stages: tuple[tuple[Entry, ...], ...], chain: _Chain, call: CallFacts | None
    ) -> _Chain:
        if not stages:
            return chain
        stage: Final = stages[0]
        results: Final = await self._gather(hook, stage, chain.fields, call, live=True)
        folded: Final = self._fold(hook, stage, results, chain)
        if folded.terminal is not None:
            return folded
        return await self._run_stages(hook, stages[1:], folded, call)

    def _fold(self, hook: Hook, stage: tuple[Entry, ...], results: tuple[object, ...], chain: _Chain) -> _Chain:
        pairs: Final = tuple(zip(stage, results, strict=True))
        decisive: Final = next((pair for pair in pairs if _decides(hook, pair[1])), None)
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
            if isinstance(result, BaseException):
                self._dispatcher.record_failure(entry, hook, result)

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
            self._swallow(hook, entries, await self._gather(hook, entries, fields, call, live=True))

    async def _deliver_off_path(
        self, hook: Hook, entries: tuple[Entry, ...], fields: Mapping[Field, object], call: CallFacts | None
    ) -> None:
        self._swallow(hook, entries, await self._gather(hook, entries, fields, call, live=False))
