from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, Generic, TypeAlias, TypeVar

BindingT = TypeVar("BindingT")
NativeT = TypeVar("NativeT")
RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class NativeSkipReason(Enum):
    DISABLED = "disabled"
    INELIGIBLE = "ineligible"
    UNAVAILABLE = "unavailable"
    DECLINED = "declined"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Handled(Generic[ResultT]):
    value: ResultT


@dataclass(frozen=True, slots=True)
class NativeSkipped:
    reason: NativeSkipReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class NativeFailed:
    error: Exception


DispatchResult: TypeAlias = Handled[ResultT] | NativeSkipped | NativeFailed


def _select(load: Callable[[], BindingT | None], enabled: bool, eligible: bool) -> BindingT | NativeSkipped:
    if not enabled:
        return NativeSkipped(NativeSkipReason.DISABLED)
    if not eligible:
        return NativeSkipped(NativeSkipReason.INELIGIBLE)
    binding: Final = load()
    return NativeSkipped(NativeSkipReason.UNAVAILABLE) if binding is None else binding


def attempt(
    *,
    load: Callable[[], BindingT | None],
    enabled: bool,
    eligible: bool,
    prepare: Callable[[], RequestT],
    call: Callable[[BindingT, RequestT], NativeT],
    adapt: Callable[[NativeT], ResultT],
) -> DispatchResult[ResultT]:
    binding: Final = _select(load, enabled, eligible)
    if isinstance(binding, NativeSkipped):
        return binding
    try:
        value: Final = call(binding, prepare())
    except Exception as error:  # noqa: BLE001  # orchestration applies the endpoint's declared error policy
        return NativeFailed(error)
    return Handled(adapt(value))


async def aattempt(
    *,
    load: Callable[[], BindingT | None],
    enabled: bool,
    eligible: bool,
    prepare: Callable[[], RequestT],
    call: Callable[[BindingT, RequestT], Awaitable[NativeT]],
    adapt: Callable[[NativeT], ResultT],
) -> DispatchResult[ResultT]:
    binding: Final = _select(load, enabled, eligible)
    if isinstance(binding, NativeSkipped):
        return binding
    try:
        value: Final = await call(binding, prepare())
    except Exception as error:  # noqa: BLE001  # orchestration applies the endpoint's declared error policy
        return NativeFailed(error)
    return Handled(adapt(value))


def identity(value: ResultT) -> ResultT:
    return value
