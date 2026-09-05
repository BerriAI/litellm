from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Generic, NoReturn, Protocol, TypeAlias, TypeVar, assert_never

from litellm.exceptions import APIError, AuthenticationError, InternalServerError, RateLimitError
from litellm.rust_bridge.bindings import (
    UNCHANGED,
    NativeBinding,
    Unchanged,
    native_declined_types,
    native_upstream_types,
)
from litellm.rust_bridge.protocols import NativeModule, RustRouteDecline
from litellm.rust_bridge.request import NativeRequestCapabilities, NativeRequestContext

BindingT = TypeVar("BindingT")
SelectedT = TypeVar("SelectedT")
SelectedSyncT = TypeVar("SelectedSyncT")
SelectedAsyncT = TypeVar("SelectedAsyncT")
NativeT = TypeVar("NativeT")
RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
SyncBindingT = TypeVar("SyncBindingT")
AsyncBindingT = TypeVar("AsyncBindingT")


class PythonFallbackReason(Enum):
    NATIVE_DISABLED = "native_disabled"
    NATIVE_UNAVAILABLE = "native_unavailable"
    NATIVE_DECLINED = "native_declined"


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
class PythonFallback:
    reason: PythonFallbackReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class NativeSkipped:
    reason: NativeSkipReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class NativeFailed:
    error: Exception


DispatchResult: TypeAlias = Handled[ResultT] | PythonFallback | NativeSkipped | NativeFailed


@dataclass(frozen=True, slots=True)
class BridgeErrorContext:
    provider: str
    model: str


class RustEnablement(Protocol):
    def __call__(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EndpointBinding(Generic[BindingT]):
    route: str
    load: Callable[[], BindingT | None]
    enabled: RustEnablement
    _native_binding: NativeBinding[BindingT] | None = field(default=None, repr=False)

    @staticmethod
    def native(
        *,
        route: str,
        select: Callable[[NativeModule], SelectedT],
        enabled: RustEnablement,
    ) -> EndpointBinding[SelectedT]:
        binding: Final = NativeBinding(select, route=route)
        return EndpointBinding(
            route=route,
            load=binding.load,
            enabled=enabled,
            _native_binding=binding,
        )

    def override(self, value: BindingT | None) -> None:
        if self._native_binding is None:
            raise RuntimeError("only native Rust bridges support binding overrides")
        self._native_binding.override(value)

    def reset(self) -> None:
        if self._native_binding is None:
            raise RuntimeError("only native Rust bridges support binding resets")
        self._native_binding.reset()

    def is_overridden(self) -> bool:
        return self._native_binding is not None and self._native_binding.is_overridden()

    def _attempt(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> DispatchResult[ResultT]:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return binding_or_fallback
        preflight_result: Final = preflight() if preflight is not None else None
        if preflight_result is not None:
            return preflight_result
        return self._attempt_call(
            call=lambda: call(binding_or_fallback, prepare()),
            adapt=adapt,
            error_context=error_context,
        )

    async def _aattempt(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> DispatchResult[ResultT]:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return binding_or_fallback
        preflight_result: Final = preflight() if preflight is not None else None
        if preflight_result is not None:
            return preflight_result
        return await self._attempt_acall(
            call=lambda: call(binding_or_fallback, prepare()),
            adapt=adapt,
            error_context=error_context,
        )

    def invoke(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], NativeT],
        fallback: Callable[[], ResultT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        result: Final = self._attempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                return fallback()
            case _ as unreachable:
                assert_never(unreachable)

    async def ainvoke(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], Awaitable[NativeT]],
        fallback: Callable[[], Awaitable[ResultT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        result: Final = await self._aattempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                return await fallback()
            case _ as unreachable:
                assert_never(unreachable)

    def assess(
        self,
        *,
        check: Callable[[BindingT], str | None],
    ) -> PythonFallback | None:
        binding: Final = self._binding_or_python_fallback(eligible=True)
        if isinstance(binding, PythonFallback):
            return binding
        reason: Final = check(binding)
        return PythonFallback(PythonFallbackReason.NATIVE_DECLINED, reason) if reason is not None else None

    def accepts(
        self,
        *,
        check: Callable[[BindingT], str | None],
        eligible: bool = True,
    ) -> bool:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return False
        try:
            reason: Final = check(binding_or_fallback)
        except Exception:  # noqa: BLE001  # preflight performs no provider I/O, so Python handoff is safe
            return False
        return reason is None

    def require(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        result: Final = self._attempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                self._raise_required(result)
            case _ as unreachable:
                assert_never(unreachable)

    async def arequire(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        result: Final = await self._aattempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                self._raise_required(result)
            case _ as unreachable:
                assert_never(unreachable)

    def can_attempt(
        self,
        *,
        eligible: bool = True,
    ) -> bool:
        return not isinstance(
            self._binding_or_python_fallback(eligible=eligible),
            PythonFallback,
        )

    def _raise_required(self, fallback: PythonFallback) -> NoReturn:
        detail: Final = f": {fallback.detail}" if fallback.detail else ""
        reason: Final = _required_reason(fallback.reason)
        raise RuntimeError(f"native {self.route} endpoint {reason}{detail}")

    def _binding_or_python_fallback(
        self,
        *,
        eligible: bool,
    ) -> BindingT | PythonFallback:
        if not eligible or not self.enabled():
            return PythonFallback(PythonFallbackReason.NATIVE_DISABLED)
        binding: Final = self.load()
        if binding is None:
            return PythonFallback(PythonFallbackReason.NATIVE_UNAVAILABLE)
        return binding

    def _attempt_call(
        self,
        *,
        call: Callable[[], NativeT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
    ) -> DispatchResult[ResultT]:
        declined: Final = native_declined_types()
        upstream: Final = native_upstream_types()
        if not declined or not upstream:
            return Handled(adapt(call()))
        try:
            value: Final = call()
        except declined as error:
            return PythonFallback(PythonFallbackReason.NATIVE_DECLINED, _error_message(error))
        except upstream as error:
            self._raise_upstream(error, error_context)
        return Handled(adapt(value))

    async def _attempt_acall(
        self,
        *,
        call: Callable[[], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
    ) -> DispatchResult[ResultT]:
        declined: Final = native_declined_types()
        upstream: Final = native_upstream_types()
        if not declined or not upstream:
            return Handled(adapt(await call()))
        try:
            value: Final = await call()
        except declined as error:
            return PythonFallback(PythonFallbackReason.NATIVE_DECLINED, _error_message(error))
        except upstream as error:
            self._raise_upstream(error, error_context)
        return Handled(adapt(value))

    def _raise_upstream(self, error: BaseException, error_context: BridgeErrorContext) -> NoReturn:
        args: Final[tuple[object, ...]] = error.args
        attribute_status: Final = getattr(error, "status_code", None)
        attribute_message: Final = getattr(error, "message", None)
        status_value: Final = attribute_status if isinstance(attribute_status, int) else (args[0] if args else 0)
        message_value: Final = (
            attribute_message if isinstance(attribute_message, str) else (args[1] if len(args) > 1 else str(error))
        )
        status: Final = status_value if isinstance(status_value, int) else 0
        message: Final = message_value if isinstance(message_value, str) else str(message_value)
        error_message: Final = f"litellm rust {self.route}: {message}"
        if status == 401:
            raise AuthenticationError(
                message=error_message,
                llm_provider=error_context.provider,
                model=error_context.model,
            ) from error
        if status == 429:
            raise RateLimitError(
                message=error_message,
                llm_provider=error_context.provider,
                model=error_context.model,
            ) from error
        if status == 500:
            raise InternalServerError(
                message=error_message,
                llm_provider=error_context.provider,
                model=error_context.model,
            ) from error
        raise APIError(
            status_code=status or 500,
            message=error_message,
            llm_provider=error_context.provider,
            model=error_context.model,
        ) from error


@dataclass(frozen=True, slots=True)
class EndpointDispatch(Generic[SyncBindingT, AsyncBindingT]):
    sync: EndpointBinding[SyncBindingT]
    asynchronous: EndpointBinding[AsyncBindingT]

    @staticmethod
    def native(
        *,
        route: str,
        sync: Callable[[NativeModule], SelectedSyncT],
        asynchronous: Callable[[NativeModule], SelectedAsyncT],
        enabled: RustEnablement,
    ) -> EndpointDispatch[SelectedSyncT, SelectedAsyncT]:
        return EndpointDispatch(
            sync=EndpointBinding.native(route=route, select=sync, enabled=enabled),
            asynchronous=EndpointBinding.native(
                route=route,
                select=asynchronous,
                enabled=enabled,
            ),
        )

    def override(
        self,
        *,
        sync: SyncBindingT | None | Unchanged = UNCHANGED,
        asynchronous: AsyncBindingT | None | Unchanged = UNCHANGED,
    ) -> None:
        if not isinstance(sync, Unchanged):
            self.sync.override(sync)
        if not isinstance(asynchronous, Unchanged):
            self.asynchronous.override(asynchronous)

    def reset(self) -> None:
        self.sync.reset()
        self.asynchronous.reset()

    def invoke(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[SyncBindingT, RequestT], NativeT],
        fallback: Callable[[], ResultT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        return self.sync.invoke(
            prepare=prepare,
            call=call,
            fallback=fallback,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )

    async def ainvoke(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[AsyncBindingT, RequestT], Awaitable[NativeT]],
        fallback: Callable[[], Awaitable[ResultT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        return await self.asynchronous.ainvoke(
            prepare=prepare,
            call=call,
            fallback=fallback,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )

    def require(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[SyncBindingT, RequestT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        return self.sync.require(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )

    async def arequire(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[AsyncBindingT, RequestT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
        preflight: Callable[[], PythonFallback | None] | None = None,
    ) -> ResultT:
        return await self.asynchronous.arequire(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
            preflight=preflight,
        )


def _error_message(error: BaseException) -> str:
    reason: Final[object] = error.args[0] if error.args else str(error)
    return reason if isinstance(reason, str) else str(reason)


def _required_reason(reason: PythonFallbackReason) -> str:
    match reason:
        case PythonFallbackReason.NATIVE_DISABLED:
            return "is disabled"
        case PythonFallbackReason.NATIVE_UNAVAILABLE:
            return "is unavailable"
        case PythonFallbackReason.NATIVE_DECLINED:
            return "declined the request"


def always_enabled() -> bool:
    return True


def identity(value: ResultT) -> ResultT:
    return value


def adapt_result(result: DispatchResult[NativeT], adapt: Callable[[NativeT], ResultT]) -> DispatchResult[ResultT]:
    if isinstance(result, Handled):
        return Handled(adapt(result.value))
    return result


async def async_none() -> None:
    return None


def assess_route(
    binding: EndpointBinding[RustRouteDecline],
    model: str,
    provider: str,
    *,
    stream: bool = False,
    has_agentic_hook: bool = False,
    has_custom_client: bool = False,
    request_format: str | None = None,
) -> PythonFallback | None:
    context: Final = NativeRequestContext(
        capabilities=NativeRequestCapabilities(
            stream=stream,
            has_agentic_hook=has_agentic_hook,
            has_custom_client=has_custom_client,
            request_format=request_format,
        )
    )
    return binding.assess(
        check=lambda decline: decline(
            model,
            provider,
            context=context,
        ),
    )
