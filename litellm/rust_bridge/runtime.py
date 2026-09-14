from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, NoReturn, TypeVar

from litellm.exceptions import APIError
from litellm.rust_bridge.bindings import (
    native_exception_types,
    native_host_callback_exception,
    native_unavailable_exception,
)
from litellm.rust_bridge.configuration import ExecutionDecision
from litellm.rust_bridge.errors import RustRouteDeclinedError, RustRouteUnavailableError
from litellm.rust_bridge.route import ComponentExecution

NativeT = TypeVar("NativeT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class BridgeErrorContext:
    route: str
    provider: str
    model: str


def invoke(
    *,
    native_call: Callable[[], NativeT] | None,
    python_fallback: Callable[[], ResultT],
    adapt: Callable[[NativeT], ResultT],
    execution: ComponentExecution,
    context: BridgeErrorContext,
) -> ResultT:
    execution.require_supported()
    if execution.decision is ExecutionDecision.PYTHON:
        return python_fallback()
    if native_call is None:
        return _unavailable_or_fallback(execution, python_fallback)

    exceptions: Final = native_exception_types()
    if exceptions is None:
        return adapt(native_call())
    declined, upstream = exceptions
    unavailable: Final = native_unavailable_exception()
    host_callback: Final = native_host_callback_exception()
    try:
        value: Final = native_call()
    except host_callback as error:
        _raise_host_callback(error)
    except unavailable:
        return _unavailable_or_fallback(execution, python_fallback)
    except declined as error:
        return _declined_or_fallback(execution, python_fallback, error)
    except upstream as error:
        _raise_upstream(error, context)
    return adapt(value)


async def ainvoke(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    python_fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[NativeT], ResultT],
    execution: ComponentExecution,
    context: BridgeErrorContext,
) -> ResultT:
    execution.require_supported()
    if execution.decision is ExecutionDecision.PYTHON:
        return await python_fallback()
    if native_call is None:
        return await _aunavailable_or_fallback(execution, python_fallback)

    exceptions: Final = native_exception_types()
    if exceptions is None:
        return adapt(await native_call())
    declined, upstream = exceptions
    unavailable: Final = native_unavailable_exception()
    host_callback: Final = native_host_callback_exception()
    try:
        value: Final = await native_call()
    except host_callback as error:
        _raise_host_callback(error)
    except unavailable:
        return await _aunavailable_or_fallback(execution, python_fallback)
    except declined as error:
        return await _adeclined_or_fallback(execution, python_fallback, error)
    except upstream as error:
        _raise_upstream(error, context)
    return adapt(value)


def _unavailable_or_fallback(
    execution: ComponentExecution,
    python_fallback: Callable[[], ResultT],
) -> ResultT:
    if execution.decision is ExecutionDecision.RUST_WITH_FALLBACK:
        return python_fallback()
    raise RustRouteUnavailableError(f"Rust {execution.route_name.value} bridge is unavailable")


async def _aunavailable_or_fallback(
    execution: ComponentExecution,
    python_fallback: Callable[[], Awaitable[ResultT]],
) -> ResultT:
    if execution.decision is ExecutionDecision.RUST_WITH_FALLBACK:
        return await python_fallback()
    raise RustRouteUnavailableError(f"Rust {execution.route_name.value} bridge is unavailable")


def _declined_or_fallback(
    execution: ComponentExecution,
    python_fallback: Callable[[], ResultT],
    error: BaseException,
) -> ResultT:
    if execution.decision is ExecutionDecision.RUST_WITH_FALLBACK:
        return python_fallback()
    _raise_declined(execution, error)


async def _adeclined_or_fallback(
    execution: ComponentExecution,
    python_fallback: Callable[[], Awaitable[ResultT]],
    error: BaseException,
) -> ResultT:
    if execution.decision is ExecutionDecision.RUST_WITH_FALLBACK:
        return await python_fallback()
    _raise_declined(execution, error)


def _raise_declined(execution: ComponentExecution, error: BaseException) -> NoReturn:
    reason_value: Final[object] = error.args[0] if error.args else str(error)
    reason: Final = reason_value if isinstance(reason_value, str) else str(reason_value)
    raise RustRouteDeclinedError(
        f"Rust {execution.route_name.value} bridge declined the request: {reason}"
    ) from error


def _raise_host_callback(error: BaseException) -> NoReturn:
    cause: Final = error.__cause__
    if cause is not None:
        raise cause
    raise error


def _raise_upstream(error: BaseException, context: BridgeErrorContext) -> NoReturn:
    args: Final[tuple[object, ...]] = error.args
    status_value: Final = args[0] if args else 0
    message_value: Final = args[1] if len(args) > 1 else str(error)
    status: Final = status_value if isinstance(status_value, int) else 0
    message: Final = message_value if isinstance(message_value, str) else str(message_value)
    raise APIError(
        status_code=status or 500,
        message=f"litellm rust {context.route}: {message}",
        llm_provider=context.provider,
        model=context.model,
    ) from error
