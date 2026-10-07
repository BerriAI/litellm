from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Generic, NoReturn, TypeAlias, TypeVar

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.rust_bridge import failures
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import RouteContext, Rules, decision
from litellm.rust_bridge.configuration import Decision
from litellm.rust_bridge.response_metadata import mark_rerouted_response, mark_rust_response

NativeT = TypeVar("NativeT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RustHandled(Generic[ResultT]):
    value: ResultT


@dataclass(frozen=True, slots=True)
class RustRerouted:
    failure: failures.NativeFailure


@dataclass(frozen=True, slots=True)
class RustUnavailable:
    pass


RustAttempt: TypeAlias = RustHandled[ResultT] | RustRerouted | RustUnavailable


@dataclass(frozen=True, slots=True)
class BridgeErrorContext:
    route: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class NoPythonImplementation:
    pass


NO_PYTHON: Final = NoPythonImplementation()


class NoPythonImplementationError(RuntimeError):
    pass


def run(
    context: RouteContext,
    *,
    binding: NativeBinding[NativeT],
    native: Callable[[NativeT], ResultT],
    python: Callable[[], ResultT] | NoPythonImplementation,
    rules: Rules | None = None,
) -> ResultT:
    selected: Final = decision(context, rules)
    if isinstance(python, NoPythonImplementation):
        _require_rust(context, selected)
        return _required(_attempt_native(context, binding, native, fallback=False), context)
    match selected:
        case Decision.PYTHON:
            return python()
        case Decision.RUST_REQUIRED:
            return _required(_attempt_native(context, binding, native, fallback=False), context)
        case Decision.RUST_WITH_FALLBACK:
            result: Final = _attempt_native(context, binding, native, fallback=True)
            match result:
                case RustHandled(value=value):
                    return mark_rust_response(value)
                case RustRerouted(failure=failure):
                    return mark_rerouted_response(python(), failure.stage)
                case RustUnavailable():
                    return python()
                case _:
                    assert_never(result)
        case _:
            assert_never(selected)


async def arun(
    context: RouteContext,
    *,
    binding: NativeBinding[NativeT],
    native: Callable[[NativeT], Awaitable[ResultT]],
    python: Callable[[], Awaitable[ResultT]] | NoPythonImplementation,
    rules: Rules | None = None,
) -> ResultT:
    selected: Final = decision(context, rules)
    if isinstance(python, NoPythonImplementation):
        _require_rust(context, selected)
        return _required(await _aattempt_native(context, binding, native, fallback=False), context)
    match selected:
        case Decision.PYTHON:
            return await python()
        case Decision.RUST_REQUIRED:
            return _required(await _aattempt_native(context, binding, native, fallback=False), context)
        case Decision.RUST_WITH_FALLBACK:
            result: Final = await _aattempt_native(context, binding, native, fallback=True)
            match result:
                case RustHandled(value=value):
                    return mark_rust_response(value)
                case RustRerouted(failure=failure):
                    return mark_rerouted_response(await python(), failure.stage)
                case RustUnavailable():
                    return await python()
                case _:
                    assert_never(result)
        case _:
            assert_never(selected)


def _require_rust(context: RouteContext, selected: Decision) -> None:
    if selected is not Decision.RUST_REQUIRED:
        raise NoPythonImplementationError(
            f"{context.route.value} has no Python implementation, so its catalog rules must resolve to "
            f"RUST_REQUIRED, but provider={context.provider!r} model={context.model!r} resolved to {selected.name}"
        )


def _attempt_native(
    context: RouteContext, binding: NativeBinding[NativeT], native: Callable[[NativeT], ResultT], *, fallback: bool
) -> RustAttempt[ResultT]:
    loaded: Final = binding.load()
    return attempt(
        native_call=None if loaded is None else lambda: native(loaded),
        adapt=_identity,
        context=_error_context(context),
        fallback=fallback,
    )


async def _aattempt_native(
    context: RouteContext,
    binding: NativeBinding[NativeT],
    native: Callable[[NativeT], Awaitable[ResultT]],
    *,
    fallback: bool,
) -> RustAttempt[ResultT]:
    loaded: Final = binding.load()
    return await aattempt(
        native_call=None if loaded is None else lambda: native(loaded),
        adapt=_identity,
        context=_error_context(context),
        fallback=fallback,
    )


def _required(result: RustAttempt[ResultT], context: RouteContext) -> ResultT:
    if isinstance(result, RustHandled):
        return mark_rust_response(result.value)
    _raise_required(result, _error_context(context))


def _identity(value: ResultT) -> ResultT:
    return value


def _error_context(context: RouteContext) -> BridgeErrorContext:
    return BridgeErrorContext(route=context.route.value, provider=context.provider or "", model=context.model or "")


def attempt(
    *,
    native_call: Callable[[], NativeT] | None,
    adapt: Callable[[NativeT], ResultT],
    context: BridgeErrorContext,
    fallback: bool,
) -> RustAttempt[ResultT]:
    if native_call is None:
        return RustUnavailable()
    try:
        value: Final = native_call()
    except Exception as error:
        return _settle(error, context, fallback)
    return RustHandled(adapt(value))


async def aattempt(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    adapt: Callable[[NativeT], ResultT],
    context: BridgeErrorContext,
    fallback: bool,
) -> RustAttempt[ResultT]:
    if native_call is None:
        return RustUnavailable()
    try:
        value: Final = await native_call()
    except Exception as error:
        return _settle(error, context, fallback)
    return RustHandled(adapt(value))


def _settle(error: Exception, context: BridgeErrorContext, fallback: bool) -> RustRerouted:
    """A native failure either reroutes the call to Python or reaches the caller as its public exception."""
    failure: Final = failures.decode(error)
    if failure is None:
        raise error
    if fallback and failures.reroutes(failure):
        verbose_logger.warning(
            "Rust %s route handed the call to Python after a %s failure (%s, provider=%s, model=%s): %s",
            context.route,
            failure.stage,
            failure.kind.kind,
            context.provider,
            context.model,
            failure.message,
        )
        return RustRerouted(failure)
    raise failures.ensure_public(error, model=context.model, provider=context.provider)


def _raise_required(result: RustRerouted | RustUnavailable, context: BridgeErrorContext) -> NoReturn:
    match result:
        case RustUnavailable():
            raise RuntimeError(f"Rust {context.route} bridge is unavailable")
        case RustRerouted(failure=failure):
            raise RuntimeError(f"Rust {context.route} bridge failed at {failure.stage}: {failure.message}")
        case _:
            assert_never(result)
