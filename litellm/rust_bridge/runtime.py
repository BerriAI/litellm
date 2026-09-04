from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Generic, NoReturn, Protocol, TypeAlias, TypeVar

from litellm.exceptions import APIError
from litellm.rust_bridge.bindings import (
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
    RUST_DISABLED = "rust_disabled"
    RUST_UNAVAILABLE = "rust_unavailable"
    RUST_DECLINED = "rust_declined"


@dataclass(frozen=True, slots=True)
class RustHandled(Generic[ResultT]):
    value: ResultT


@dataclass(frozen=True, slots=True)
class PythonFallback:
    reason: PythonFallbackReason
    detail: str | None = None


RustDispatchResult: TypeAlias = RustHandled[ResultT] | PythonFallback


@dataclass(frozen=True, slots=True)
class BridgeErrorContext:
    provider: str
    model: str


class RustEnablement(Protocol):
    def __call__(self, *, request_override: bool | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class RustBridge(Generic[BindingT]):
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
    ) -> RustBridge[BindingT]:
        binding: NativeBinding[BindingT] = NativeBinding.callable(attribute)
        return cls(
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

    def _attempt(
        self,
        *,
        call: Callable[[BindingT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> RustDispatchResult[ResultT]:
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
    ) -> RustDispatchResult[ResultT]:
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
            case RustHandled(value=value):
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
            case RustHandled(value=value):
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
            case RustHandled(value=value):
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
            case RustHandled(value=value):
                return value
            case PythonFallback():
                self._raise_required(result)

    def accepts(
        self,
        *,
        check: Callable[[BindingT], str | None],
        request_override: bool | None = None,
        eligible: bool = True,
    ) -> bool:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            request_override=request_override,
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return False
        try:
            reason: Final = check(binding_or_fallback)
        except Exception:  # noqa: BLE001  # preflight performs no provider I/O, so Python handoff is safe
            return False
        return reason is None

    def _binding_or_python_fallback(
        self,
        *,
        request_override: bool | None,
        eligible: bool,
    ) -> BindingT | PythonFallback:
        if not eligible or not self.enabled(request_override=request_override):
            return PythonFallback(PythonFallbackReason.RUST_DISABLED)
        binding: Final = self.load()
        if binding is None:
            return PythonFallback(PythonFallbackReason.RUST_UNAVAILABLE)
        return binding

    def _attempt_call(
        self,
        *,
        call: Callable[[], NativeT],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
    ) -> RustDispatchResult[ResultT]:
        exceptions: Final = native_exception_types()
        if exceptions is None:
            return RustHandled(adapt(call()))
        declined, upstream = exceptions
        try:
            value: Final = call()
        except declined as error:
            return PythonFallback(PythonFallbackReason.RUST_DECLINED, _error_message(error))
        except upstream as error:
            self._raise_upstream(error, context)
        return RustHandled(adapt(value))

    async def _attempt_acall(
        self,
        *,
        call: Callable[[], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        context: BridgeErrorContext,
    ) -> RustDispatchResult[ResultT]:
        exceptions: Final = native_exception_types()
        if exceptions is None:
            return RustHandled(adapt(await call()))
        declined, upstream = exceptions
        try:
            value: Final = await call()
        except declined as error:
            return PythonFallback(PythonFallbackReason.RUST_DECLINED, _error_message(error))
        except upstream as error:
            self._raise_upstream(error, context)
        return RustHandled(adapt(value))

    def _raise_required(self, fallback: PythonFallback) -> NoReturn:
        detail: Final = f": {fallback.detail}" if fallback.detail else ""
        reason: Final = _required_reason(fallback.reason)
        raise RuntimeError(f"Rust {self.route} bridge {reason}{detail}")

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
class RustEndpoint(Generic[SyncBindingT, AsyncBindingT]):
    sync: RustBridge[SyncBindingT]
    asynchronous: RustBridge[AsyncBindingT]

    @classmethod
    def native(
        cls,
        *,
        route: str,
        sync: str,
        asynchronous: str,
        enabled: RustEnablement,
    ) -> RustEndpoint[SyncBindingT, AsyncBindingT]:
        return cls(
            sync=RustBridge.native(route=route, attribute=sync, enabled=enabled),
            asynchronous=RustBridge.native(
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


def _error_message(error: BaseException) -> str:
    reason: Final[object] = error.args[0] if error.args else str(error)
    return reason if isinstance(reason, str) else str(reason)


def _required_reason(reason: PythonFallbackReason) -> str:
    match reason:
        case PythonFallbackReason.RUST_DISABLED:
            return "is disabled"
        case PythonFallbackReason.RUST_UNAVAILABLE:
            return "is unavailable"
        case PythonFallbackReason.RUST_DECLINED:
            return "declined the request"


def always_enabled(*, request_override: bool | None = None) -> bool:
    return True


def identity(value: ResultT) -> ResultT:
    return value


async def async_none() -> None:
    return None
