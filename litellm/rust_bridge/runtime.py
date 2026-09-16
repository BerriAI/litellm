from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Generic, NoReturn, TypeAlias, TypeVar, assert_never

from litellm.exceptions import APIError
from litellm.rust_bridge.bindings import NativeBinding, native_exception_types
from litellm.rust_bridge.catalog import RULES, Context, Rules, decision
from litellm.rust_bridge.configuration import Decision

NativeT = TypeVar("NativeT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RustHandled(Generic[ResultT]):
    value: ResultT


@dataclass(frozen=True, slots=True)
class RustDeclined:
    reason: str


@dataclass(frozen=True, slots=True)
class RustUnavailable:
    pass


RustAttempt: TypeAlias = RustHandled[ResultT] | RustDeclined | RustUnavailable


@dataclass(frozen=True, slots=True)
class BridgeErrorContext:
    route: str
    provider: str
    model: str


def run(
    context: Context,
    *,
    binding: NativeBinding[NativeT],
    native: Callable[[NativeT], ResultT],
    python: Callable[[], ResultT],
    rules: Rules = RULES,
) -> ResultT:
    selected: Final = decision(context, rules)
    match selected:
        case Decision.PYTHON:
            return python()
        case Decision.RUST_WITH_FALLBACK | Decision.RUST_REQUIRED:
            loaded: Final = binding.load()
            result: Final = attempt(
                native_call=None if loaded is None else lambda: native(loaded),
                adapt=_identity,
                context=_error_context(context),
            )
            if isinstance(result, RustHandled):
                return result.value
            if selected is Decision.RUST_REQUIRED:
                _raise_required(result, _error_context(context))
            return python()
        case _:
            assert_never(selected)


async def arun(
    context: Context,
    *,
    binding: NativeBinding[NativeT],
    native: Callable[[NativeT], Awaitable[ResultT]],
    python: Callable[[], Awaitable[ResultT]],
    rules: Rules = RULES,
) -> ResultT:
    selected: Final = decision(context, rules)
    match selected:
        case Decision.PYTHON:
            return await python()
        case Decision.RUST_WITH_FALLBACK | Decision.RUST_REQUIRED:
            loaded: Final = binding.load()
            result: Final = await aattempt(
                native_call=None if loaded is None else lambda: native(loaded),
                adapt=_identity,
                context=_error_context(context),
            )
            if isinstance(result, RustHandled):
                return result.value
            if selected is Decision.RUST_REQUIRED:
                _raise_required(result, _error_context(context))
            return await python()
        case _:
            assert_never(selected)


def _identity(value: ResultT) -> ResultT:
    return value


def _error_context(context: Context) -> BridgeErrorContext:
    return BridgeErrorContext(route=context.route.value, provider=context.provider or "", model=context.model or "")


def attempt(
    *,
    native_call: Callable[[], NativeT] | None,
    adapt: Callable[[NativeT], ResultT],
    context: BridgeErrorContext,
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
        _raise_upstream(error, context)
    return RustHandled(adapt(value))


async def aattempt(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    adapt: Callable[[NativeT], ResultT],
    context: BridgeErrorContext,
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
        _raise_upstream(error, context)
    return RustHandled(adapt(value))


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
