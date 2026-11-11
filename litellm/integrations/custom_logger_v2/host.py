from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Generic, TypeAlias, TypeVar

from litellm.integrations.custom_logger_v2.call import CallFacts
from litellm.integrations.custom_logger_v2.dispatcher import Dispatcher, LegacyGroup, RequestScope
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.hooks import HOOK_SPECS, InvalidDeclarationError
from litellm.integrations.custom_logger_v2.manifest import CallbackManifest
from litellm.integrations.custom_logger_v2.outcomes import Outcome, Patch
from litellm.integrations.custom_logger_v2.registry import CallbackRegistry
from litellm.integrations.custom_logger_v2.selection import RequestSelection

ResultT: Final = TypeVar("ResultT")

FieldsProvider: TypeAlias = Callable[[], Mapping[Field, object]]
SyncLegacy: TypeAlias = Callable[[tuple[Patch, ...]], ResultT]

SYNC_HOOKS: Final = frozenset({Hook.prompt_resolve}) | frozenset(
    hook for hook, spec in HOOK_SPECS.items() if not spec.request_path
)

_NO_FIELDS: Final[Mapping[Field, object]] = MappingProxyType({})
_NO_SELECTION: Final = RequestSelection()
_CURRENT_SCOPE: Final[ContextVar[RequestScope | None]] = ContextVar("custom_logger_v2_scope", default=None)


class NativeOutcome(Exception):
    def __init__(self, outcome: Outcome | None, patches: tuple[Patch, ...]) -> None:
        super().__init__(f"a native callback ended the hook with {type(outcome).__name__}")
        self.outcome: Final = outcome
        self.patches: Final = patches


@dataclass(frozen=True, slots=True)
class _Ran(Generic[ResultT]):
    value: ResultT


class _Host:
    def __init__(self) -> None:
        self.native_hooks: frozenset[Hook] = frozenset()
        self._dispatcher: Dispatcher | None = None

    def dispatcher(self) -> Dispatcher:
        if self._dispatcher is None:
            self._dispatcher = Dispatcher(CallbackRegistry())
        return self._dispatcher

    def register(self, manifest: CallbackManifest) -> None:
        self.dispatcher().registry.register(manifest)
        self.native_hooks = self.native_hooks | manifest.subscriptions.keys()

    def reset(self) -> None:
        self._dispatcher = None
        self.native_hooks = frozenset()


_HOST: Final = _Host()


def _sync_problem(manifest: CallbackManifest) -> str | None:
    blocking: Final = sorted(
        hook.value
        for hook, subscription in manifest.subscriptions.items()
        if hook in SYNC_HOOKS and (subscription.writes or subscription.may_abort or subscription.awaited)
    )
    return f"{', '.join(blocking)} run synchronously, so only off-path natives may subscribe" if blocking else None


def get_dispatcher() -> Dispatcher:
    return _HOST.dispatcher()


def register_native(manifest: CallbackManifest) -> None:
    problem: Final = _sync_problem(manifest)
    if problem is not None:
        raise InvalidDeclarationError(f"{manifest.selection.name}: {problem}")
    _HOST.register(manifest)


def reset_host_for_tests() -> None:
    _HOST.reset()


def has_natives(hook: Hook) -> bool:
    return hook in _HOST.native_hooks


def current_scope() -> RequestScope | None:
    return _CURRENT_SCOPE.get()


@contextmanager
def bind_scope(scope: RequestScope) -> Generator[RequestScope]:
    token: Final = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


@contextmanager
def request_scope(selection: RequestSelection = _NO_SELECTION) -> Generator[RequestScope]:
    with _HOST.dispatcher().open_scope(selection) as scope, bind_scope(scope):
        yield scope


def _fields(provider: FieldsProvider | None) -> Mapping[Field, object]:
    return _NO_FIELDS if provider is None else provider()


async def _run_in(
    scope: RequestScope,
    hook: Hook,
    legacy: LegacyGroup[ResultT],
    fields: FieldsProvider | None,
    call: CallFacts | None,
) -> ResultT:
    async def ran(patches: tuple[Patch, ...]) -> _Ran[ResultT]:
        return _Ran(await legacy(patches))

    result: Final = await scope.run_hook(hook, _fields(fields), call, ran)
    if result.legacy is None:
        raise NativeOutcome(result.outcome, result.patches)
    return result.legacy.value


async def _run_with_natives(
    hook: Hook,
    legacy: LegacyGroup[ResultT],
    fields: FieldsProvider | None,
    call: CallFacts | None,
    scope: RequestScope | None,
) -> ResultT:
    bound: Final = scope or _CURRENT_SCOPE.get()
    if bound is not None:
        return await _run_in(bound, hook, legacy, fields, call)
    with _HOST.dispatcher().open_scope(_NO_SELECTION) as transient:
        return await _run_in(transient, hook, legacy, fields, call)


async def run_legacy_hook(
    hook: Hook,
    legacy: LegacyGroup[ResultT],
    *,
    fields: FieldsProvider | None = None,
    call: CallFacts | None = None,
    scope: RequestScope | None = None,
) -> ResultT:
    if hook not in _HOST.native_hooks:
        return await legacy(())
    return await _run_with_natives(hook, legacy, fields, call, scope)


def run_legacy_hook_sync(
    hook: Hook,
    legacy: SyncLegacy[ResultT],
    *,
    fields: FieldsProvider | None = None,
    call: CallFacts | None = None,
    scope: RequestScope | None = None,
) -> ResultT:
    if hook not in _HOST.native_hooks:
        return legacy(())
    bound: Final = scope or _CURRENT_SCOPE.get()
    if bound is not None:
        return bound.run_sync_hook(hook, _fields(fields), call, legacy)
    with _HOST.dispatcher().open_scope(_NO_SELECTION) as transient:
        return transient.run_sync_hook(hook, _fields(fields), call, legacy)
