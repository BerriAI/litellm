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
from litellm.rust_bridge.protocols import NativeModule

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


class NativeErrorPolicy(Enum):
    TRANSLATE = "translate"
    PROPAGATE = "propagate"


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
    def __call__(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EndpointBinding(Generic[BindingT]):
    route: str
    load: Callable[[], BindingT | None]
    enabled: RustEnablement
    error_policy: NativeErrorPolicy = NativeErrorPolicy.TRANSLATE
    _native_binding: NativeBinding[BindingT] | None = field(default=None, repr=False)

    @staticmethod
    def native(
        *,
        route: str,
        select: Callable[[NativeModule], SelectedT],
        enabled: RustEnablement,
        error_policy: NativeErrorPolicy = NativeErrorPolicy.TRANSLATE,
    ) -> EndpointBinding[SelectedT]:
        binding: Final = NativeBinding(select)
        return EndpointBinding(
            route=route,
            load=binding.load,
            enabled=enabled,
            error_policy=error_policy,
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
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
    ) -> DispatchResult[ResultT]:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return binding_or_fallback
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
    ) -> DispatchResult[ResultT]:
        binding_or_fallback: Final = self._binding_or_python_fallback(
            eligible=eligible,
        )
        if isinstance(binding_or_fallback, PythonFallback):
            return binding_or_fallback
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
    ) -> ResultT:
        result: Final = self._attempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
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
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], Awaitable[NativeT]],
        fallback: Callable[[], Awaitable[ResultT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
    ) -> ResultT:
        result: Final = await self._aattempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
        )
        match result:
            case Handled(value=value):
                return value
            case PythonFallback():
                return await fallback()

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
    ) -> ResultT:
        result: Final = self._attempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
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
        prepare: Callable[[], RequestT],
        call: Callable[[BindingT, RequestT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
    ) -> ResultT:
        result: Final = await self._aattempt(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
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
        if self.error_policy is NativeErrorPolicy.PROPAGATE:
            return Handled(adapt(call()))
        exceptions: Final = native_exception_types()
        if exceptions is None:
            try:
                value_without_exceptions: Final = call()
            except Exception as error:  # noqa: BLE001  # preserve chat fallback when native exception classes are absent
                return PythonFallback(PythonFallbackReason.NATIVE_DECLINED, _error_message(error))
            return Handled(adapt(value_without_exceptions))
        declined, upstream = exceptions
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
        if self.error_policy is NativeErrorPolicy.PROPAGATE:
            return Handled(adapt(await call()))
        exceptions: Final = native_exception_types()
        if exceptions is None:
            try:
                value_without_exceptions: Final = await call()
            except Exception as error:  # noqa: BLE001  # preserve chat fallback when native exception classes are absent
                return PythonFallback(PythonFallbackReason.NATIVE_DECLINED, _error_message(error))
            return Handled(adapt(value_without_exceptions))
        declined, upstream = exceptions
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
        raise APIError(
            status_code=status or 500,
            message=f"litellm rust {self.route}: {message}",
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
        error_policy: NativeErrorPolicy = NativeErrorPolicy.TRANSLATE,
    ) -> EndpointDispatch[SelectedSyncT, SelectedAsyncT]:
        return EndpointDispatch(
            sync=EndpointBinding.native(route=route, select=sync, enabled=enabled, error_policy=error_policy),
            asynchronous=EndpointBinding.native(
                route=route,
                select=asynchronous,
                enabled=enabled,
                error_policy=error_policy,
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
    ) -> ResultT:
        return self.sync.invoke(
            prepare=prepare,
            call=call,
            fallback=fallback,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
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
    ) -> ResultT:
        return await self.asynchronous.ainvoke(
            prepare=prepare,
            call=call,
            fallback=fallback,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
        )

    def require(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[SyncBindingT, RequestT], NativeT],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
    ) -> ResultT:
        return self.sync.require(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
        )

    async def arequire(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[AsyncBindingT, RequestT], Awaitable[NativeT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
    ) -> ResultT:
        return await self.asynchronous.arequire(
            prepare=prepare,
            call=call,
            adapt=adapt,
            error_context=error_context,
            eligible=eligible,
        )


@dataclass(frozen=True, slots=True)
class AsyncEndpointDispatch(Generic[AsyncBindingT]):
    asynchronous: EndpointBinding[AsyncBindingT]

    @staticmethod
    def native(
        *,
        route: str,
        asynchronous: Callable[[NativeModule], SelectedAsyncT],
        enabled: RustEnablement,
    ) -> AsyncEndpointDispatch[SelectedAsyncT]:
        return AsyncEndpointDispatch(
            asynchronous=EndpointBinding.native(route=route, select=asynchronous, enabled=enabled)
        )

    def override(self, value: AsyncBindingT | None) -> None:
        self.asynchronous.override(value)

    def reset(self) -> None:
        self.asynchronous.reset()

    async def ainvoke(
        self,
        *,
        prepare: Callable[[], RequestT],
        call: Callable[[AsyncBindingT, RequestT], Awaitable[NativeT]],
        fallback: Callable[[], Awaitable[ResultT]],
        adapt: Callable[[NativeT], ResultT],
        error_context: BridgeErrorContext,
        eligible: bool = True,
    ) -> ResultT:
        return await self.asynchronous.ainvoke(
            prepare=prepare,
            call=call,
            fallback=fallback,
            adapt=adapt,
            error_context=error_context,
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


def always_enabled() -> bool:
    return True


def identity(value: ResultT) -> ResultT:
    return value


async def async_none() -> None:
    return None
