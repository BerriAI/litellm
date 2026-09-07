import asyncio
import gc
import inspect
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast

from callback_lifecycle import ReferenceFactory, run_checked, settle


class PreparedInvocation(Protocol):
    def invoke(self) -> object: ...

    def close(self) -> None: ...


class CallFactory(Protocol):
    def prepare(
        self,
        callable: Callable[..., object],
        positional: tuple[object, ...],
        keywords: dict[str, object] | None = None,
        awaited: bool = False,
    ) -> PreparedInvocation: ...


class LiveCallFactory(CallFactory, Protocol):
    @property
    def live(self) -> int: ...


def unchanged_result(value: object) -> object:
    return value


@dataclass(frozen=True, slots=True)
class ResultTransformInvocation:
    inner: PreparedInvocation
    transform: Callable[[object], object]
    awaited: bool

    def invoke(self) -> object:
        result = self.inner.invoke()
        transform = self.transform
        if not self.awaited:
            return transform(result)

        async def run() -> object:
            return transform(await cast(Awaitable[object], result))

        return run()

    def close(self) -> None:
        self.inner.close()


@dataclass(frozen=True, slots=True)
class ResultTransformFactory:
    inner: CallFactory
    transform: Callable[[object], object]

    def prepare(
        self,
        callable: Callable[..., object],
        positional: tuple[object, ...],
        keywords: dict[str, object] | None = None,
        awaited: bool = False,
    ) -> PreparedInvocation:
        return ResultTransformInvocation(
            self.inner.prepare(callable, positional, keywords, awaited=awaited), self.transform, awaited
        )


@dataclass(frozen=True, slots=True)
class ExpiredBorrow:
    edges: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CheckedWeakInvocation:
    callback: weakref.ReferenceType[Callable[..., object]]
    positional: tuple[weakref.ReferenceType[object], ...]
    keywords: tuple[tuple[str, weakref.ReferenceType[object]], ...]
    awaited: bool

    def resolve(self) -> tuple[Callable[..., object], tuple[object, ...], dict[str, object]] | ExpiredBorrow:
        callback = self.callback()
        positional = tuple(reference() for reference in self.positional)
        keywords = {name: reference() for name, reference in self.keywords}
        expired = (
            *(("callable",) if callback is None else ()),
            *(f"positional:{index}" for index, value in enumerate(positional) if value is None),
            *(f"keyword:{name}" for name, value in keywords.items() if value is None),
        )
        if expired:
            return ExpiredBorrow(expired)
        assert callback is not None
        return callback, positional, keywords

    def invoke(self) -> object:
        if not self.awaited:
            resolved = self.resolve()
            if isinstance(resolved, ExpiredBorrow):
                return resolved
            callback, positional, keywords = resolved
            return callback(*positional, **keywords)

        async def run() -> object:
            resolved = self.resolve()
            if isinstance(resolved, ExpiredBorrow):
                return resolved
            callback, positional, keywords = resolved
            return await cast(Awaitable[object], callback(*positional, **keywords))

        return run()

    def close(self) -> None:
        pass


class CheckedWeakFactory:
    def prepare(
        self,
        callable: Callable[..., object],
        positional: tuple[object, ...],
        keywords: dict[str, object] | None = None,
        awaited: bool = False,
    ) -> PreparedInvocation:
        return CheckedWeakInvocation(
            weakref.ref(callable),
            tuple(weakref.ref(value) for value in positional),
            tuple((name, weakref.ref(value)) for name, value in (keywords or {}).items()),
            awaited,
        )


@dataclass(frozen=True, slots=True)
class MissingHandoffInvocation:
    inner: PreparedInvocation
    borrowed: PreparedInvocation

    def invoke(self) -> object:
        return self.borrowed.invoke()

    def close(self) -> None:
        try:
            self.inner.close()
        finally:
            self.borrowed.close()


@dataclass(frozen=True, slots=True)
class MissingHandoffFactory:
    inner: CallFactory

    def prepare(
        self,
        callable: Callable[..., object],
        positional: tuple[object, ...],
        keywords: dict[str, object] | None = None,
        awaited: bool = False,
    ) -> PreparedInvocation:
        borrowed = CheckedWeakFactory().prepare(callable, positional, keywords, awaited=awaited)
        return MissingHandoffInvocation(self.inner.prepare(callable, positional, keywords, awaited=awaited), borrowed)


def control_factory(control: str, inner: CallFactory) -> CallFactory:
    if control == "identity":
        return inner
    if control == "weak":
        return CheckedWeakFactory()
    if control == "missing_handoff":
        return MissingHandoffFactory(inner)
    return {"result_passthrough": ResultTransformFactory(inner, unchanged_result)}[control]


@dataclass
class ControlNode:
    stage: int = 0


@dataclass
class LifetimeCallback:
    awaited: bool

    def __call__(self, value: ControlNode, *, alias: ControlNode) -> object:
        if self.awaited:
            return self.run(value, alias=alias)
        return value.stage + alias.stage

    async def run(self, value: ControlNode, *, alias: ControlNode) -> int:
        await asyncio.sleep(0)
        return value.stage + alias.stage


def prepare_released(
    owners: CallFactory, awaited: bool
) -> tuple[
    PreparedInvocation,
    tuple[
        weakref.ReferenceType[LifetimeCallback], weakref.ReferenceType[ControlNode], weakref.ReferenceType[ControlNode]
    ],
]:
    callback = LifetimeCallback(awaited)
    value = ControlNode(13)
    alias = ControlNode(29)
    return (
        owners.prepare(callback, (value,), {"alias": alias}, awaited=awaited),
        (weakref.ref(callback), weakref.ref(value), weakref.ref(alias)),
    )


@dataclass(frozen=True, slots=True)
class LifetimeObservation:
    alive: tuple[bool, bool, bool]
    result: int | ExpiredBorrow


async def deferred_lifetime(owners: CallFactory, awaited: bool) -> LifetimeObservation:
    owner, references = prepare_released(owners, awaited)
    try:
        gc.collect()
        alive = tuple(reference() is not None for reference in references)
        pending = owner.invoke()
        result = await settle(pending, awaited)
        observation = LifetimeObservation(alive, result)
    finally:
        owner.close()
    gc.collect()
    assert all(reference() is None for reference in references)
    return observation


@dataclass(frozen=True, slots=True)
class BorrowedObservation:
    positional_identity: bool
    keyword_identity: bool
    positional_stage: int
    keyword_stage: int


async def borrowed_lifetime(owners: CallFactory, awaited: bool) -> BorrowedObservation:
    value, alias = ControlNode(), ControlNode()
    value_ref, alias_ref = weakref.ref(value), weakref.ref(alias)

    def observe(value: ControlNode, *, alias: ControlNode) -> BorrowedObservation:
        return BorrowedObservation(value is value_ref(), alias is alias_ref(), value.stage, alias.stage)

    async def observe_async(value: ControlNode, *, alias: ControlNode) -> BorrowedObservation:
        return observe(value, alias=alias)

    owner = owners.prepare(observe_async if awaited else observe, (value,), {"alias": alias}, awaited=awaited)
    try:
        value.stage, alias.stage = 13, 29
        pending = owner.invoke()
        value.stage, alias.stage = 17, 31
        return await settle(pending, awaited)
    finally:
        owner.close()


async def pending_handoff(owners: CallFactory, awaited: bool) -> LifetimeObservation:
    assert awaited
    owner, references = prepare_released(owners, awaited)
    try:
        pending = owner.invoke()
        try:
            owner.close()
            gc.collect()
            alive = tuple(reference() is not None for reference in references)
            observation = LifetimeObservation(alive, await pending)
        finally:
            pending.close()
    finally:
        owner.close()
    gc.collect()
    assert all(reference() is None for reference in references)
    return observation


async def direct_coroutine(owners: CallFactory, awaited: bool) -> bool:
    async def body() -> int:
        return 73

    original = body()
    try:
        owner = owners.prepare(lambda: original, (), awaited=awaited)
        try:
            pending = owner.invoke()
            if awaited:
                assert await pending == 73
                nested = body()
                try:

                    async def returns_coroutine() -> Awaitable[int]:
                        return nested

                    nested_owner = owners.prepare(returns_coroutine, (), awaited=True)
                    try:
                        result = await nested_owner.invoke()
                        return result is nested and inspect.getcoroutinestate(nested) == inspect.CORO_CREATED
                    finally:
                        nested_owner.close()
                finally:
                    nested.close()
            try:
                return pending is original and inspect.getcoroutinestate(original) == inspect.CORO_CREATED
            finally:
                if inspect.iscoroutine(pending):
                    pending.close()
        finally:
            owner.close()
    finally:
        original.close()


def expected_control(witness: str, control: str, awaited: bool) -> object:
    if witness in ("deferred_lifetime", "pending_handoff"):
        if control == "weak" or (witness == "pending_handoff" and control == "missing_handoff"):
            return LifetimeObservation(
                (False, False, False), ExpiredBorrow(("callable", "positional:0", "keyword:alias"))
            )
        return LifetimeObservation((True, True, True), 42)
    if witness == "borrowed_lifetime":
        return BorrowedObservation(True, True, 17 if awaited else 13, 31 if awaited else 29)
    return {"direct_coroutine": True}[witness]


def run_control(witness: str, control: str, retained: bool, awaited: bool, factory: LiveCallFactory) -> None:
    inner = factory if retained else ReferenceFactory()
    owners = control_factory(control, inner)

    async def run() -> None:
        observed = await WITNESSES[witness](owners, awaited)
        assert observed == expected_control(witness, control, awaited), (witness, control, awaited, observed)

    run_checked(inner, run())


WITNESSES: dict[str, Callable[[CallFactory, bool], object]] = {
    "deferred_lifetime": deferred_lifetime,
    "borrowed_lifetime": borrowed_lifetime,
    "pending_handoff": pending_handoff,
    "direct_coroutine": direct_coroutine,
}
