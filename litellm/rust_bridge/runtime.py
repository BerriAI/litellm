from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Final,
    Generic,
    NoReturn,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
)

from ..exceptions import APIError
from .bindings import (
    UNCHANGED,
    NativeBinding,
    Unchanged,
    native_exception_types,
)

BindingT = TypeVar("BindingT")
NativeT = TypeVar("NativeT")
ResultT = TypeVar("ResultT")
SyncBindingT = TypeVar("SyncBindingT")
AsyncBindingT = TypeVar("AsyncBindingT")


class PythonFallbackReason(Enum):
    NATIVE_DISABLED = "native_disabled"
    NATIVE_UNAVAILABLE = "native_unavailable"
    NATIVE_DECLINED = "native_declined"


@dataclass(frozen=True, slots=True)
class Handled(Generic[ResultT]):
    value: ResultT


@dataclass(frozen=True, slots=True)
class PythonFallback:
    reason: PythonFallbackReason
    detail: str | None = None


DispatchResult: TypeAlias = Handled[ResultT] | PythonFallback


@dataclass(frozen=True, slots=True)
class BridgeErrorContext:
    provider: str
    model: str


class RustEnablement(Protocol):
    def __call__(self, *, request_override: bool | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class EndpointBinding(Generic[BindingT]):
    route: str
    load: Callable[[], BindingT | None]
    enabled: RustEnablement
    _native_binding: NativeBinding[BindingT] | None = field(default=None, repr=False)

    @classmethod
    def native(
        cls,
        *,
        route: str,
        attribute: str,
        enabled: RustEnablement,
    ) -> EndpointBinding[BindingT]:
        binding: NativeBinding[BindingT] = cast(  # cast-ok: classmethod loses the caller's generic parameter
            NativeBinding[BindingT],
            NativeBinding.native(route, attribute),
        )
        return EndpointBinding(
            route=route,
            load=binding.load,
            enabled=enabled,
            _native_binding=binding,
        )

    def override(self, value: BindingT | None) -> None:
        if self._native_binding is None:
            raise RuntimeError("only native endpoint bindings support overrides")
        self._native_binding.override(value)

    def reset(self) -> None:
        if self._native_binding is None:
            raise RuntimeError("only native endpoint bindings support resets")
        self._native_binding.reset()

    def is_overridden(self) -> bool:
        return self._native_binding is not None and self._native_binding.is_overridden()

    def _attempt(
        self,
        *,
        call: Callable[[BindingT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> DispatchResult[ResultT]:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            request_override=request_override,
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return binding_or_fallback
        return self._attempt_call(
            call=lambda: call(binding_or_fallback),
            adapt=adapt,
            context=context,
        )

    async def _aattempt(
        self,
        *,
        call: Callable[[BindingT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> DispatchResult[ResultT]:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            request_override=request_override,
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return binding_or_fallback
        return await self._attempt_acall(
            call=lambda: call(binding_or_fallback),
            adapt=adapt,
            context=context,
        )

    def invoke(
        self,
        *,
        call: Callable[[BindingT], NativeT],
        fallback: Callable[[], ResultT],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        result: Final = self._attempt(
            call=call,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                return fallback()

    async def ainvoke(
        self,
        *,
        call: Callable[[BindingT], Awaitable[NativeT]],
        fallback: Callable[[], Awaitable[ResultT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        result: Final = await self._aattempt(
            call=call,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                return await fallback()

    def require(
        self,
        *,
        call: Callable[[BindingT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        result: Final = self._attempt(
            call=call,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                self._raise_required(result)

    async def arequire(
        self,
        *,
        call: Callable[[BindingT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        result: Final = await self._aattempt(
            call=call,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                self._raise_required(result)

    def can_attempt(
        self,
        *,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> bool:
        return not isinstance(
            self._binding_or_python_fallback(request_override=request_override, eligible=eligible),
            PythonFallback,
        )

    def _binding_or_python_fallback(
        self,
        *,
        request_override: bool | None,
        eligible: bool,
    ) -> BindingT | PythonFallback:
        if not eligible or not self.enabled(request_override=request_override):
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
        context: BridgeErrorContext,
    ) -> DispatchResult[ResultT]:
        exceptions: Final = native_exception_types()
        if exceptions is None:
            return Handled(adapt(call()))
        declined, upstream = exceptions
        try:
            value: Final = call()
        except declined as error:
            return PythonFallback(PythonFallbackReason.NATIVE_DECLINED, _error_message(error))
        except upstream as error:
            self._raise_upstream(error, context)
        return Handled(adapt(value))

    async def _attempt_acall(
        self,
        *,
        call: Callable[[], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
    ) -> DispatchResult[ResultT]:
        exceptions: Final = native_exception_types()
        if exceptions is None:
            return Handled(adapt(await call()))
        declined, upstream = exceptions
        try:
            value: Final = await call()
        except declined as error:
            return PythonFallback(PythonFallbackReason.NATIVE_DECLINED, _error_message(error))
        except upstream as error:
            self._raise_upstream(error, context)
        return Handled(adapt(value))

    def _raise_required(self, fallback: PythonFallback) -> NoReturn:
        detail: Final = f": {fallback.detail}" if fallback.detail else ""
        reason: Final = _required_reason(fallback.reason)
        raise RuntimeError(f"native {self.route} endpoint {reason}{detail}")

    def _raise_upstream(self, error: BaseException, context: BridgeErrorContext) -> NoReturn:
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
            message=f"litellm rust {self.route}: {message}",
            llm_provider=context.provider,
            model=context.model,
        ) from error


@dataclass(frozen=True, slots=True)
class EndpointDispatch(Generic[SyncBindingT, AsyncBindingT]):
    sync: EndpointBinding[SyncBindingT]
    asynchronous: EndpointBinding[AsyncBindingT]

    @classmethod
    def native(
        cls,
        *,
        route: str,
        sync: str,
        asynchronous: str,
        enabled: RustEnablement,
    ) -> EndpointDispatch[SyncBindingT, AsyncBindingT]:
        return EndpointDispatch(
            sync=EndpointBinding.native(route=route, attribute=sync, enabled=enabled),
            asynchronous=EndpointBinding.native(
                route=route,
                attribute=asynchronous,
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
        call: Callable[[SyncBindingT], NativeT],
        fallback: Callable[[], ResultT],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        return self.sync.invoke(
            call=call,
            fallback=fallback,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )

    async def ainvoke(
        self,
        *,
        call: Callable[[AsyncBindingT], Awaitable[NativeT]],
        fallback: Callable[[], Awaitable[ResultT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        return await self.asynchronous.ainvoke(
            call=call,
            fallback=fallback,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )

    def require(
        self,
        *,
        call: Callable[[SyncBindingT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        return self.sync.require(
            call=call,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )

    async def arequire(
        self,
        *,
        call: Callable[[AsyncBindingT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        return await self.asynchronous.arequire(
            call=call,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
        )


@dataclass(frozen=True, slots=True)
class AsyncEndpointDispatch(Generic[AsyncBindingT]):
    asynchronous: EndpointBinding[AsyncBindingT]

    @classmethod
    def native(
        cls,
        *,
        route: str,
        asynchronous: str,
        enabled: RustEnablement,
    ) -> AsyncEndpointDispatch[AsyncBindingT]:
        return AsyncEndpointDispatch(
            asynchronous=EndpointBinding.native(
                route=route,
                attribute=asynchronous,
                enabled=enabled,
            )
        )

    def override(self, value: AsyncBindingT | None) -> None:
        self.asynchronous.override(value)

    def reset(self) -> None:
        self.asynchronous.reset()

    async def ainvoke(
        self,
        *,
        call: Callable[[AsyncBindingT], Awaitable[NativeT]],
        fallback: Callable[[], Awaitable[ResultT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> ResultT:
        return await self.asynchronous.ainvoke(
            call=call,
            fallback=fallback,
            adapt=adapt,
            context=context,
            request_override=request_override,
            eligible=eligible,
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


def always_enabled(*, request_override: bool | None = None) -> bool:
    return True


def identity(value: ResultT) -> ResultT:
    return value


async def async_none() -> None:
    return None
