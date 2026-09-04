from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, Generic, NoReturn, TypeAlias, TypeVar

from litellm.exceptions import APIError
from litellm.rust_bridge.bindings import native_exception_types

NativeT = TypeVar("NativeT")
ResultT = TypeVar("ResultT")


class FallbackMode(Enum):
    PYTHON = "python"
    RUST_REQUIRED = "rust_required"


@dataclass(frozen=True, slots=True)
class RustHandled(Generic[ResultT]):
    value: ResultT


@dataclass(frozen=True, slots=True)
class RustDeclined:
    reason: str


@dataclass(frozen=True, slots=True)
class RustFailed:
    status_code: int
    message: str


@dataclass(frozen=True, slots=True)
class RustUnavailable:
    pass


RustOutcome: TypeAlias = RustHandled[ResultT] | RustDeclined | RustFailed
RustAttempt: TypeAlias = RustOutcome[ResultT] | RustUnavailable


@dataclass(frozen=True, slots=True)
class BridgeErrorContext:
    route: str
    provider: str
    model: str


def invoke(
    *,
    native_call: Callable[[], NativeT] | None,
    fallback: Callable[[], ResultT],
    adapt: Callable[[NativeT], ResultT],
    mode: FallbackMode,
    context: BridgeErrorContext,
) -> ResultT:
    result: Final = attempt(native_call=native_call, adapt=adapt)
    if isinstance(result, RustHandled):
        return result.value
    if isinstance(result, RustFailed):
        raise_failure(result, context)
    if mode is FallbackMode.PYTHON:
        return fallback()
    _raise_required(result, context)


async def ainvoke(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[NativeT], ResultT],
    mode: FallbackMode,
    context: BridgeErrorContext,
) -> ResultT:
    result: Final = await aattempt(native_call=native_call, adapt=adapt)
    if isinstance(result, RustHandled):
        return result.value
    if isinstance(result, RustFailed):
        raise_failure(result, context)
    if mode is FallbackMode.PYTHON:
        return await fallback()
    _raise_required(result, context)


def attempt(
    *,
    native_call: Callable[[], NativeT] | None,
    adapt: Callable[[NativeT], ResultT],
) -> RustAttempt[ResultT]:
    if native_call is None:
        return RustUnavailable()
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return RustHandled(adapt(native_call()))
    declined, upstream = exceptions
    try:
        value: Final = native_call()
    except declined as error:
        return RustDeclined(reason=_decline_reason(error))
    except upstream as error:
        return _failed(error)
    return RustHandled(adapt(value))


async def aattempt(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    adapt: Callable[[NativeT], ResultT],
) -> RustAttempt[ResultT]:
    if native_call is None:
        return RustUnavailable()
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return RustHandled(adapt(await native_call()))
    declined, upstream = exceptions
    try:
        value: Final = await native_call()
    except declined as error:
        return RustDeclined(reason=_decline_reason(error))
    except upstream as error:
        return _failed(error)
    return RustHandled(adapt(value))


def complete(
    *,
    native_call: Callable[[], NativeT],
    adapt: Callable[[NativeT], ResultT],
) -> RustHandled[ResultT] | RustFailed:
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return RustHandled(adapt(native_call()))
    upstream: Final = exceptions[1]
    try:
        value: Final = native_call()
    except upstream as error:
        return _failed(error)
    return RustHandled(adapt(value))


async def acomplete(
    *,
    native_call: Callable[[], Awaitable[NativeT]],
    adapt: Callable[[NativeT], ResultT],
) -> RustHandled[ResultT] | RustFailed:
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return RustHandled(adapt(await native_call()))
    upstream: Final = exceptions[1]
    try:
        value: Final = await native_call()
    except upstream as error:
        return _failed(error)
    return RustHandled(adapt(value))


def call(operation: Callable[[], ResultT], context: BridgeErrorContext) -> ResultT:
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return operation()
    upstream: Final = exceptions[1]
    try:
        return operation()
    except upstream as error:
        raise_failure(_failed(error), context)


async def acall(operation: Callable[[], Awaitable[ResultT]], context: BridgeErrorContext) -> ResultT:
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return await operation()
    upstream: Final = exceptions[1]
    try:
        return await operation()
    except upstream as error:
        raise_failure(_failed(error), context)


def _decline_reason(error: BaseException) -> str:
    reason: Final[object] = error.args[0] if error.args else str(error)
    return reason if isinstance(reason, str) else str(reason)


def _raise_required(
    result: RustDeclined | RustUnavailable,
    context: BridgeErrorContext,
) -> NoReturn:
    raise RuntimeError(f"Rust {context.route} bridge {_required_reason(result)}")


def _required_reason(result: RustDeclined | RustUnavailable) -> str:
    match result:
        case RustUnavailable():
            return "is unavailable"
        case RustDeclined(reason=reason):
            return f"declined the request: {reason}"


def _failed(error: BaseException) -> RustFailed:
    args: Final[tuple[object, ...]] = error.args
    status_value: Final = args[0] if args else 0
    message_value: Final = args[1] if len(args) > 1 else str(error)
    status: Final = status_value if isinstance(status_value, int) else 0
    message: Final = message_value if isinstance(message_value, str) else str(message_value)
    return RustFailed(status_code=status, message=message)


def raise_failure(failure: RustFailed, context: BridgeErrorContext) -> NoReturn:
    raise APIError(
        status_code=failure.status_code or 500,
        message=f"litellm rust {context.route}: {failure.message}",
        llm_provider=context.provider,
        model=context.model,
    )


def identity(value: ResultT) -> ResultT:
    return value


async def async_none() -> None:
    return None
