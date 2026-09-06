from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeAlias, TypeVar

from litellm._logging import verbose_logger
from litellm.exceptions import APIError
from litellm.rust_bridge.bindings import native_declined_types, native_upstream_types
from litellm.rust_bridge.runtime import DispatchResult, Handled, NativeFailed, NativeSkipped, NativeSkipReason

NativeT = TypeVar("NativeT")
PythonT = TypeVar("PythonT")


class ErrorAction(Enum):
    RAISE = "raise"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class APIErrorMapping:
    provider: str
    model: str


FailureAction: TypeAlias = ErrorAction | APIErrorMapping


@dataclass(frozen=True, slots=True)
class ErrorHandling:
    declined: FailureAction = ErrorAction.RAISE
    upstream: FailureAction = ErrorAction.RAISE
    unknown: FailureAction = ErrorAction.RAISE
    missing_metadata: FailureAction = ErrorAction.RAISE
    unexpected: FailureAction = ErrorAction.RAISE


PROPAGATE: Final = ErrorHandling()
PYTHON_ON_ERROR: Final = ErrorHandling(
    declined=ErrorAction.SKIP,
    upstream=ErrorAction.SKIP,
    unknown=ErrorAction.SKIP,
    missing_metadata=ErrorAction.SKIP,
    unexpected=ErrorAction.SKIP,
)


def _handle_error(error: Exception, action: FailureAction, route: str, reason: NativeSkipReason) -> NativeSkipped:
    match action:
        case ErrorAction.SKIP:
            return NativeSkipped(reason, str(error))
        case ErrorAction.RAISE:
            raise error
        case APIErrorMapping(provider, model):
            args: Final[tuple[object, ...]] = error.args
            attribute_status: Final = getattr(error, "status_code", None)
            attribute_message: Final = getattr(error, "message", None)
            status_value: Final = attribute_status if isinstance(attribute_status, int) else (args[0] if args else 0)
            message_value: Final = (
                attribute_message if isinstance(attribute_message, str) else (args[1] if len(args) > 1 else str(error))
            )
            status: Final = status_value if isinstance(status_value, int) else 0
            message: Final = message_value if isinstance(message_value, str) else str(message_value)
            raise APIError(
                status_code=status or 500,
                message=f"litellm rust {route}: {message}",
                llm_provider=provider,
                model=model,
            ) from error


def _resolve(result: DispatchResult[NativeT], errors: ErrorHandling, route: str) -> Handled[NativeT] | NativeSkipped:
    if not isinstance(result, NativeFailed):
        return result
    declined: Final = native_declined_types()
    upstream: Final = native_upstream_types()
    if not declined or not upstream:
        return _handle_error(result.error, errors.missing_metadata, route, NativeSkipReason.FAILED)
    if isinstance(result.error, declined):
        return _handle_error(result.error, errors.declined, route, NativeSkipReason.DECLINED)
    if isinstance(result.error, upstream):
        return _handle_error(result.error, errors.upstream, route, NativeSkipReason.FAILED)
    return _handle_error(result.error, errors.unknown, route, NativeSkipReason.FAILED)


def _log_skip(route: str, skipped: NativeSkipped) -> None:
    verbose_logger.debug("Native %s skipped (%s): %s", route, skipped.reason.value, skipped.detail or "")


def dispatch(
    *,
    native: Callable[[], DispatchResult[NativeT]],
    python: Callable[[], PythonT],
    route: str,
    errors: ErrorHandling,
) -> NativeT | PythonT:
    try:
        attempted: Final = native()
    except Exception as error:  # noqa: BLE001  # preserve declared handling of loading and adaptation failures
        unexpected: Final = _handle_error(error, errors.unexpected, route, NativeSkipReason.FAILED)
        _log_skip(route, unexpected)
        return python()
    result: Final = _resolve(attempted, errors, route)
    match result:
        case Handled(value):
            return value
        case NativeSkipped():
            _log_skip(route, result)
            return python()


async def adispatch(
    *,
    native: Callable[[], Awaitable[DispatchResult[NativeT]]],
    python: Callable[[], Awaitable[PythonT]],
    route: str,
    errors: ErrorHandling,
) -> NativeT | PythonT:
    try:
        attempted: Final = await native()
    except Exception as error:  # noqa: BLE001  # preserve declared handling of loading and adaptation failures
        unexpected: Final = _handle_error(error, errors.unexpected, route, NativeSkipReason.FAILED)
        _log_skip(route, unexpected)
        return await python()
    result: Final = _resolve(attempted, errors, route)
    match result:
        case Handled(value):
            return value
        case NativeSkipped():
            _log_skip(route, result)
            return await python()


async def async_none() -> None:
    return None
