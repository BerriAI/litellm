from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Final, ParamSpec, TypeAlias, TypeVar

from litellm._logging import verbose_logger
from litellm.exceptions import APIError, AuthenticationError, InternalServerError, RateLimitError
from litellm.rust_bridge.bindings import native_declined_types, native_upstream_types
from litellm.rust_bridge.runtime import DispatchResult, Handled, NativeFailed, NativeSkipped, NativeSkipReason

NativeT = TypeVar("NativeT")
PythonT = TypeVar("PythonT")
P = ParamSpec("P")


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


def provider_errors(provider: str, model: str) -> ErrorHandling:
    return ErrorHandling(
        declined=ErrorAction.SKIP,
        upstream=APIErrorMapping(provider=provider, model=model),
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
            error_message: Final = f"litellm rust {route}: {message}"
            if status == 401:
                raise AuthenticationError(message=error_message, llm_provider=provider, model=model) from error
            if status == 429:
                raise RateLimitError(message=error_message, llm_provider=provider, model=model) from error
            if status == 500:
                raise InternalServerError(message=error_message, llm_provider=provider, model=model) from error
            raise APIError(
                status_code=status or 500,
                message=error_message,
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


def native_first(
    *,
    native: Callable[P, DispatchResult[NativeT]],
    route: str,
    errors: Callable[P, ErrorHandling],
) -> Callable[[Callable[P, PythonT]], Callable[P, NativeT | PythonT]]:
    def wrap(implementation: Callable[P, PythonT]) -> Callable[P, NativeT | PythonT]:
        @wraps(implementation)
        def run(
            *args: P.args,
            **kwargs: P.kwargs,  # kwargs-ok: ParamSpec preserves the wrapped signature
        ) -> NativeT | PythonT:
            rules: Final = errors(*args, **kwargs)
            try:
                attempted: Final = native(*args, **kwargs)
            except Exception as error:  # noqa: BLE001  # preserve declared handling of loading and adaptation failures
                skipped: Final = _handle_error(error, rules.unexpected, route, NativeSkipReason.FAILED)
                _log_skip(route, skipped)
            else:
                result: Final = _resolve(attempted, rules, route)
                if isinstance(result, Handled):
                    return result.value
                _log_skip(route, result)
            return implementation(*args, **kwargs)

        return run

    return wrap


def anative_first(
    *,
    native: Callable[P, Awaitable[DispatchResult[NativeT]]],
    route: str,
    errors: Callable[P, ErrorHandling],
) -> Callable[[Callable[P, Awaitable[PythonT]]], Callable[P, Awaitable[NativeT | PythonT]]]:
    def wrap(implementation: Callable[P, Awaitable[PythonT]]) -> Callable[P, Awaitable[NativeT | PythonT]]:
        @wraps(implementation)
        async def run(
            *args: P.args,
            **kwargs: P.kwargs,  # kwargs-ok: ParamSpec preserves the wrapped signature
        ) -> NativeT | PythonT:
            rules: Final = errors(*args, **kwargs)
            try:
                attempted: Final = await native(*args, **kwargs)
            except Exception as error:  # noqa: BLE001  # preserve declared handling of loading and adaptation failures
                skipped: Final = _handle_error(error, rules.unexpected, route, NativeSkipReason.FAILED)
                _log_skip(route, skipped)
            else:
                result: Final = _resolve(attempted, rules, route)
                if isinstance(result, Handled):
                    return result.value
                _log_skip(route, result)
            return await implementation(*args, **kwargs)

        return run

    return wrap


def anative_context(
    *,
    native: Callable[P, Awaitable[DispatchResult[AbstractAsyncContextManager[NativeT]]]],
    route: str,
    errors: Callable[P, ErrorHandling],
) -> Callable[
    [Callable[P, AbstractAsyncContextManager[PythonT]]],
    Callable[P, AbstractAsyncContextManager[NativeT | PythonT]],
]:
    def wrap(
        implementation: Callable[P, AbstractAsyncContextManager[PythonT]],
    ) -> Callable[P, AbstractAsyncContextManager[NativeT | PythonT]]:
        @anative_first(native=native, route=route, errors=errors)
        async def acquire(
            *args: P.args,
            **kwargs: P.kwargs,  # kwargs-ok: ParamSpec preserves the wrapped signature
        ) -> AbstractAsyncContextManager[PythonT]:
            return implementation(*args, **kwargs)

        @wraps(implementation)
        @asynccontextmanager
        async def run(
            *args: P.args,
            **kwargs: P.kwargs,  # kwargs-ok: ParamSpec preserves the wrapped signature
        ) -> AsyncGenerator[NativeT | PythonT, None]:
            manager: Final = await acquire(*args, **kwargs)
            async with manager as connection:
                yield connection

        return run

    return wrap
