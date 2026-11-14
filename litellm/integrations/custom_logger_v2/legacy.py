import weakref
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, reduce
from itertools import count
from types import MappingProxyType
from typing import Final, TypeAlias, TypeVar

from typing_extensions import TypeIs  # noqa: TID251  # narrows the untyped legacy kwargs without copying them

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger, subclass_overrides_hook
from litellm.integrations.custom_logger_v2.call import CallFacts
from litellm.integrations.custom_logger_v2.dispatcher import (
    Between,
    HookResult,
    RequestScope,
    run_legacy_plan,
    run_legacy_plan_sync,
)
from litellm.integrations.custom_logger_v2.enums import Activation, Field, Hook
from litellm.integrations.custom_logger_v2.hooks import HOOK_SPECS, Subscription
from litellm.integrations.custom_logger_v2.host import V2Outcome, current_scope, get_dispatcher
from litellm.integrations.custom_logger_v2.logger import CustomLoggerV2
from litellm.integrations.custom_logger_v2.manifest import CallbackManifest, LegacyBinding, build_manifest
from litellm.integrations.custom_logger_v2.outcomes import Abort, InvalidPatchError, MergeField, Patch, PatchOp, Respond
from litellm.integrations.custom_logger_v2.patches import LiveFields
from litellm.integrations.custom_logger_v2.planner import Entry, HookPlan, plan_hook
from litellm.integrations.custom_logger_v2.selection import RequestSelection, Selection

ResultT: Final = TypeVar("ResultT")

SyncLegacyBinding: TypeAlias = Callable[[object, object], object]

PARALLEL_GUARDRAIL_HOOKS: Final = frozenset({Hook.pre_call, Hook.post_call_success})
LEGACY_READS: Final = frozenset({Field.deprecated_kwargs, Field.deprecated_response_obj})
PROXY_REQUEST_KEYS: Final[Mapping[Field, str]] = MappingProxyType(
    {
        Field.model: "model",
        Field.messages: "messages",
        Field.input: "input",
        Field.tools: "tools",
    }
)
SUBSCRIBERS_CACHE_SIZE: Final = 128
PLAN_CACHE_SIZE: Final = 64

_SERIALS: Final = count()


class Admit(str, Enum):
    overriders = "overriders"
    every_logger = "every_logger"
    every_entry = "every_entry"


_ABSENT: Final = object()
_NO_SELECTION: Final = RequestSelection()


@lru_cache(maxsize=None)
def legacy_subscription(hook: Hook, parallel: bool) -> Subscription:
    spec: Final = HOOK_SPECS[hook]
    may_abort: Final = bool(spec.outcomes)
    gathered: Final = may_abort and (parallel or hook is Hook.during_call)
    return Subscription(
        hook=hook,
        handler="call",
        reads=LEGACY_READS,
        writes=frozenset({Field.deprecated_kwargs}) if spec.writes and not gathered else frozenset(),
        may_abort=may_abort,
        parallel=gathered,
        legacy=True,
    )


def _subscribes(cls: type, guardrail: bool, hook: Hook) -> bool:
    spec: Final = HOOK_SPECS[hook]
    if not spec.subscribable:
        return False
    if guardrail and spec.legacy_guardrail_events:
        return True
    return any(subclass_overrides_hook(cls, method) for method in spec.legacy_methods)


@lru_cache(maxsize=512)
def _derived_subscriptions(cls: type, guardrail: bool, parallel: bool) -> Mapping[Hook, Subscription]:
    return MappingProxyType(
        {
            hook: legacy_subscription(hook, parallel and hook in PARALLEL_GUARDRAIL_HOOKS)
            for hook in Hook
            if _subscribes(cls, guardrail, hook)
        }
    )


class LegacyLogger(CustomLoggerV2):
    def __init__(self, inner: CustomLogger) -> None:
        guardrail: Final = isinstance(inner, CustomGuardrail)
        self.parallel = guardrail and bool(getattr(inner, "run_in_parallel", False))
        self.serial: Final = next(_SERIALS)
        self._inner: Final = weakref.ref(inner)
        self.manifest: Final = CallbackManifest(
            logger=self,
            selection=Selection(name=type(inner).__name__, activation=Activation.required),
            subscriptions=_derived_subscriptions(type(inner), guardrail, self.parallel),
            request_state=None,
            legacy_call=self.call,
        )

    @property
    def inner(self) -> CustomLogger | None:
        return self._inner()

    def subscription(self, hook: Hook, admit: Admit) -> Subscription | None:
        derived: Final = self.manifest.subscriptions.get(hook)
        if derived is not None or admit is Admit.overriders or not HOOK_SPECS[hook].subscribable:
            return derived
        return legacy_subscription(hook, self.parallel and hook in PARALLEL_GUARDRAIL_HOOKS)

    def call(self, binding: Callable[[object, object], ResultT], fields: Mapping[Field, object]) -> ResultT | None:
        inner: Final = self._inner()
        return None if inner is None else binding(inner, fields[Field.deprecated_kwargs])


_ADAPTERS: Final[dict[int, LegacyLogger]] = {}  # mutable-ok: adapters live as long as the logger they wrap


def adapter_for(inner: CustomLogger) -> LegacyLogger:
    key: Final = id(inner)
    cached: Final = _ADAPTERS.get(key)
    if cached is not None and cached.inner is inner:
        return cached
    adapter: Final = LegacyLogger(inner)
    _ADAPTERS[key] = adapter
    _ = weakref.finalize(inner, _ADAPTERS.pop, key, None)
    return adapter


def _readable(value: object) -> TypeIs[Mapping[str, object]]:  # guard-ok: legacy kwargs are an untyped dict
    return isinstance(value, Mapping)


def _writable(
    value: object,
) -> TypeIs[MutableMapping[str, object]]:  # guard-ok: legacy kwargs are an untyped dict  # mutable-ok: written in place
    return isinstance(value, MutableMapping)


class OpaqueLegacy(CustomLoggerV2):
    """Holds a callback list entry that is not a CustomLogger (a string name or a plain function) at its list
    position, for hook points that admit every entry. There is no class to derive subscriptions from, so it
    subscribes to the one hook it was built for and runs serially."""

    def __init__(self, inner: object, hook: Hook) -> None:
        self.inner: Final = inner
        self.serial: Final = next(_SERIALS)
        self.manifest: Final = CallbackManifest(
            logger=self,
            selection=Selection(name=type(inner).__name__, activation=Activation.required),
            subscriptions=MappingProxyType({hook: legacy_subscription(hook, False)}),
            request_state=None,
            legacy_call=self.call,
        )

    def call(self, binding: Callable[[object, object], ResultT], fields: Mapping[Field, object]) -> ResultT | None:
        return binding(self.inner, fields[Field.deprecated_kwargs])


class KwargsFields(LiveFields):
    """A live Field view over a legacy hook's untyped kwargs dict, so the dispatcher can run over it without copying.

    Two fields cover the whole call: deprecated_kwargs is the dict itself, and deprecated_response_obj is the
    response, when the hook point passes one. Patching either one swaps in the new value.

    Every other field maps to one key in the dict through the hook point's key table (PROXY_REQUEST_KEYS or
    LOGGING_KEYS), and is present only when that key is. Patching one writes straight into the dict, so later
    callbacks and the request see the change."""

    __slots__ = ("_keys", "_kwargs", "_response")

    def __init__(self, kwargs: object, keys: Mapping[Field, str], response: object = _ABSENT) -> None:
        self._kwargs: Final = kwargs
        self._keys: Final = keys
        self._response: Final = response

    def _key(self, field: Field) -> str | None:
        key: Final = self._keys.get(field)
        return key if key is not None and _readable(self._kwargs) and key in self._kwargs else None

    def __getitem__(self, field: Field) -> object:
        if field is Field.deprecated_kwargs:
            return self._kwargs
        if field is Field.deprecated_response_obj and self._response is not _ABSENT:
            return self._response
        key: Final = self._key(field)
        if key is None or not _readable(self._kwargs):
            raise KeyError(field)
        return self._kwargs[key]

    def __iter__(self) -> Iterator[Field]:
        return iter(tuple(field for field in Field if self._has(field)))

    def __len__(self) -> int:
        return sum(1 for field in Field if self._has(field))

    def _has(self, field: Field) -> bool:
        if field is Field.deprecated_kwargs:
            return True
        if field is Field.deprecated_response_obj:
            return self._response is not _ABSENT
        return self._key(field) is not None

    def apply(self, patch: Patch) -> Mapping[Field, object]:
        return reduce(_apply_op, patch.ops, self)

    def replaced(self, field: Field, value: object) -> "KwargsFields":
        if field is Field.deprecated_kwargs:
            return KwargsFields(value, self._keys, self._response)
        return KwargsFields(self._kwargs, self._keys, value)

    def write(self, field: Field, value: object) -> None:
        target: Final = self._target(field)
        target[self._keys[field]] = value

    def merge(self, field: Field, keys: Mapping[str, object]) -> None:
        target: Final = self._target(field)
        current: Final = target.get(self._keys[field])
        if current is None:
            target[self._keys[field]] = dict(keys)
            return
        if not _writable(current):
            raise InvalidPatchError(f"{field.value} holds no mapping here, so keys cannot be merged into it")
        current.update(keys)

    def _target(self, field: Field) -> MutableMapping[str, object]:  # mutable-ok: legacy kwargs are the live request
        if field not in self._keys:
            raise InvalidPatchError(f"{field.value} has no place in the legacy kwargs of this hook")
        if not _writable(self._kwargs):
            raise InvalidPatchError(f"{field.value} cannot be written because the legacy kwargs are not a mapping")
        return self._kwargs


def _apply_op(fields: Mapping[Field, object], op: PatchOp) -> Mapping[Field, object]:
    if not isinstance(fields, KwargsFields):
        raise InvalidPatchError("legacy kwargs patches need KwargsFields")
    if op.field in (Field.deprecated_kwargs, Field.deprecated_response_obj):
        if isinstance(op, MergeField):
            raise InvalidPatchError(f"{op.field.value} is replaced whole, never merged")
        return fields.replaced(op.field, op.value)
    if isinstance(op, MergeField):
        fields.merge(op.field, op.keys)
        return fields
    fields.write(op.field, op.value)
    return fields


def kwargs_patch(before: object, after: object) -> Patch | None:
    return None if after is before else Patch().set(Field.deprecated_kwargs, after)


@dataclass(frozen=True, slots=True, eq=False)
class HookSubscribers:
    hook: Hook
    entries: tuple[Entry, ...]
    plan: HookPlan
    has_v2: bool
    plans: Callable[[tuple[int, ...]], HookPlan]

    @property
    def legacy(self) -> tuple[object, ...]:
        return tuple(inner for entry in self.entries if (inner := _wrapped(entry.manifest.logger)) is not None)

    def plan_for(self, scope: RequestScope) -> HookPlan:
        selected: Final = tuple(
            index for index, entry in enumerate(self.entries) if entry.subscription.legacy or scope.selects(entry)
        )
        return self.plan if len(selected) == len(self.entries) else self.plans(selected)


def _wrapped(logger: CustomLoggerV2) -> object:
    if isinstance(logger, LegacyLogger | OpaqueLegacy):
        return logger.inner
    return None


def _v2_entry(v2_callback: CustomLoggerV2, hook: Hook) -> tuple[CallbackManifest, Subscription] | None:
    manifest: Final = build_manifest(v2_callback)
    subscription: Final = manifest.subscriptions.get(hook)
    return None if subscription is None else (manifest, subscription)


def _legacy_entry(
    inner: CustomLogger, hook: Hook, admit: Admit, parallel: bool
) -> tuple[CallbackManifest, Subscription] | None:
    adapter: Final = adapter_for(inner)
    subscription: Final = adapter.subscription(hook, admit)
    if subscription is None:
        return None
    if subscription.parallel and not parallel:
        return adapter.manifest, legacy_subscription(hook, False)
    return adapter.manifest, subscription


def _opaque_entry(item: object, hook: Hook) -> tuple[CallbackManifest, Subscription] | None:
    if not HOOK_SPECS[hook].subscribable:
        return None
    opaque: Final = OpaqueLegacy(item, hook)
    return opaque.manifest, opaque.manifest.subscriptions[hook]


def _subscriber(
    item: object, hook: Hook, admit: Admit, include_v2: bool, parallel: bool
) -> tuple[CallbackManifest, Subscription] | None:
    if isinstance(item, CustomLoggerV2):
        return _v2_entry(item, hook) if include_v2 else None
    if isinstance(item, CustomLogger):
        return _legacy_entry(item, hook, admit, parallel)
    return _opaque_entry(item, hook) if admit is Admit.every_entry else None


def _build_subscribers(
    hook: Hook, callbacks: Sequence[object], admit: Admit, include_v2: bool, parallel: bool
) -> HookSubscribers:
    members: Final = tuple(
        member for item in callbacks if (member := _subscriber(item, hook, admit, include_v2, parallel)) is not None
    )
    entries: Final = tuple(
        Entry(position, manifest, subscription) for position, (manifest, subscription) in enumerate(members)
    )

    def subset_plan(selected: tuple[int, ...]) -> HookPlan:
        return plan_hook(tuple(entries[index] for index in selected))

    return HookSubscribers(
        hook=hook,
        entries=entries,
        plan=plan_hook(entries),
        has_v2=any(not entry.subscription.legacy for entry in entries),
        plans=lru_cache(maxsize=PLAN_CACHE_SIZE)(subset_plan),
    )


@dataclass(frozen=True, slots=True, eq=False)
class _CachedSubscribers:
    subscribers: HookSubscribers
    pinned: tuple[object, ...]


_SUBSCRIBERS: Final[dict[tuple[object, ...], _CachedSubscribers]] = {}  # mutable-ok: bounded subscribers cache


def _pinned(item: object) -> bool:
    return not isinstance(item, CustomLogger) or isinstance(item, CustomLoggerV2)


def _cache_subscribers(key: tuple[object, ...], subscribers: HookSubscribers, callbacks: Sequence[object]) -> None:
    if len(_SUBSCRIBERS) >= SUBSCRIBERS_CACHE_SIZE:
        _SUBSCRIBERS.clear()
    _SUBSCRIBERS[key] = _CachedSubscribers(
        subscribers=subscribers, pinned=tuple(item for item in callbacks if _pinned(item))
    )
    for item in callbacks:
        if not _pinned(item):
            _ = weakref.finalize(item, _SUBSCRIBERS.pop, key, None)


def hook_subscribers(
    hook: Hook,
    callbacks: Sequence[object],
    *,
    admit: Admit = Admit.overriders,
    include_v2: bool = True,
    parallel: bool = True,
) -> HookSubscribers:
    key: Final = (hook, admit, include_v2, parallel, *map(id, callbacks))
    cached: Final = _SUBSCRIBERS.get(key)
    if cached is not None:
        return cached.subscribers
    subscribers: Final = _build_subscribers(hook, callbacks, admit, include_v2, parallel)
    _cache_subscribers(key, subscribers, callbacks)
    return subscribers


def clear_subscribers_cache_for_tests() -> None:
    _SUBSCRIBERS.clear()


async def run_subscribers(
    subscribers: HookSubscribers,
    fields: Mapping[Field, object],
    binding: LegacyBinding,
    *,
    between: Between | None = None,
    call: CallFacts | None = None,
    scope: RequestScope | None = None,
) -> HookResult:
    if not subscribers.has_v2:
        return HookResult(fields=await run_legacy_plan(subscribers.hook, subscribers.plan, fields, binding, between))
    bound: Final = scope or current_scope()
    if bound is not None:
        return await bound.run_plan(subscribers.hook, subscribers.plan_for(bound), fields, call, binding, between)
    with get_dispatcher().open_scope(_NO_SELECTION) as transient:
        return await transient.run_plan(
            subscribers.hook, subscribers.plan_for(transient), fields, call, binding, between
        )


def run_subscribers_sync(
    subscribers: HookSubscribers,
    fields: Mapping[Field, object],
    binding: SyncLegacyBinding,
    *,
    call: CallFacts | None = None,
    scope: RequestScope | None = None,
) -> Mapping[Field, object]:
    if not subscribers.has_v2:
        return run_legacy_plan_sync(subscribers.plan, fields, binding)
    bound: Final = scope or current_scope()
    if bound is not None:
        return bound.run_plan_sync(subscribers.hook, subscribers.plan_for(bound), fields, call, binding)
    with get_dispatcher().open_scope(_NO_SELECTION) as transient:
        return transient.run_plan_sync(subscribers.hook, subscribers.plan_for(transient), fields, call, binding)


def kwargs_after(result: HookResult) -> object:
    match result.outcome:
        case Abort(exception=Exception() as error):
            raise error
        case Abort() | Respond() as terminal:
            raise V2Outcome(terminal, result.patches)
        case _:
            return result.fields[Field.deprecated_kwargs]
