from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Generic, NoReturn, TypeAlias, TypeVar, cast

from litellm.exceptions import APIError
from litellm.rust_bridge.bindings import native_exception_types

NativeT = TypeVar("NativeT")
ResultT = TypeVar("ResultT")

CORE_ENGINE_HIDDEN_PARAM: Final = "core_engine"
CORE_ENGINE_HEADER: Final = "x-litellm-core"
LEGACY_RUST_HEADER: Final = "x-litellm-rust"


class FallbackMode(Enum):
    PYTHON = "python"
    RUST_REQUIRED = "rust_required"


class CoreEngine(str, Enum):
    PYTHON = "python"
    RUST = "rust"


@dataclass(frozen=True, slots=True)
class ExecutionResult(Generic[ResultT]):
    value: ResultT
    source: CoreEngine


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
    upstream_error: Callable[[int, str], Exception] | None = None


def execution_headers(source: CoreEngine) -> dict[str, str]:
    if source is CoreEngine.RUST:
        return {  # mutable-ok: response adapters require a mutable header dict
            CORE_ENGINE_HEADER: source.value,
            LEGACY_RUST_HEADER: "true",
        }
    return {CORE_ENGINE_HEADER: source.value}  # mutable-ok: response adapters require a mutable header dict


def execution_additional_headers(
    additional_headers: Mapping[str, object] | None,
    source: CoreEngine,
) -> dict[str, str]:
    existing: Final = additional_headers or {}  # mutable-ok: empty default is local and never mutated
    reserved: Final = frozenset({CORE_ENGINE_HEADER, LEGACY_RUST_HEADER})
    preserved: Final = {  # mutable-ok: response adapters require a mutable header dict
        str(name): str(value) for name, value in existing.items() if str(name).lower() not in reserved
    }
    return {  # mutable-ok: response adapters require a mutable header dict
        **preserved,
        **execution_headers(source),
    }


def execution_hidden_params(
    hidden_params: Mapping[str, object] | None,
    source: CoreEngine,
) -> dict[str, object]:
    existing: Final = hidden_params or {}  # mutable-ok: empty default is local and never mutated
    raw_headers: Final = existing.get("additional_headers")
    additional_headers: Final[Mapping[str, object]] = (
        cast(Mapping[str, object], raw_headers)  # cast-ok: the isinstance check validates the mapping boundary
        if isinstance(raw_headers, Mapping)
        else {}  # mutable-ok: empty default is local and never mutated
    )
    return {  # mutable-ok: response objects require mutable hidden params
        **existing,
        CORE_ENGINE_HIDDEN_PARAM: source.value,
        "additional_headers": execution_additional_headers(additional_headers, source),
    }


def invoke(
    *,
    native_call: Callable[[], NativeT] | None,
    fallback: Callable[[], ResultT],
    adapt: Callable[[NativeT], ResultT],
    mode: FallbackMode,
    context: BridgeErrorContext,
) -> ExecutionResult[ResultT]:
    result: Final = attempt(native_call=native_call, adapt=adapt, context=context)
    if isinstance(result, RustHandled):
        return ExecutionResult(value=result.value, source=CoreEngine.RUST)
    if mode is FallbackMode.PYTHON:
        return ExecutionResult(value=fallback(), source=CoreEngine.PYTHON)
    _raise_required(result, context)


async def ainvoke(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[NativeT], ResultT],
    mode: FallbackMode,
    context: BridgeErrorContext,
) -> ExecutionResult[ResultT]:
    result: Final = await aattempt(native_call=native_call, adapt=adapt, context=context)
    if isinstance(result, RustHandled):
        return ExecutionResult(value=result.value, source=CoreEngine.RUST)
    if mode is FallbackMode.PYTHON:
        return ExecutionResult(value=await fallback(), source=CoreEngine.PYTHON)
    _raise_required(result, context)


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


def call(operation: Callable[[], ResultT], context: BridgeErrorContext) -> ResultT:
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return operation()
    upstream: Final = exceptions[1]
    try:
        return operation()
    except upstream as error:
        _raise_upstream(error, context)


async def acall(operation: Callable[[], Awaitable[ResultT]], context: BridgeErrorContext) -> ResultT:
    exceptions: Final = native_exception_types()
    if exceptions is None:
        return await operation()
    upstream: Final = exceptions[1]
    try:
        return await operation()
    except upstream as error:
        _raise_upstream(error, context)


def _decline_reason(error: BaseException) -> str:
    reason: Final = error.args[0] if error.args else str(error)
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
    args: Final = cast(tuple[object, ...], error.args)
    status_value: Final = args[0] if args else 0
    message_value: Final = args[1] if len(args) > 1 else str(error)
    status: Final = status_value if isinstance(status_value, int) else 0
    message: Final = message_value if isinstance(message_value, str) else str(message_value)
    api_error: Final = (
        context.upstream_error(status or 500, message)
        if context.upstream_error is not None
        else APIError(
            status_code=status or 500,
            message=f"litellm rust {context.route}: {message}",
            llm_provider=context.provider,
            model=context.model,
        )
    )
    api_error.headers = execution_headers(  # pyright: ignore[reportAttributeAccessIssue]  # proxy reads exception headers
        CoreEngine.RUST
    )
    raise api_error from error


def identity(value: ResultT) -> ResultT:
    return value


async def async_none() -> None:
    return None
