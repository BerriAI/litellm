from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias, TypeVar

from litellm.rust_bridge import runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Decision, Policy, Route, RouteContext
from litellm.rust_bridge.public_call import Bind, NativeCall, native_call, optional_str

NativeT: Final = TypeVar("NativeT")
ResultT: Final = TypeVar("ResultT")

Fields: TypeAlias = Mapping[str, object]


def provider_kwarg(fields: Fields) -> str | None:
    return optional_str(fields.get("custom_llm_provider"))


def model_is_named(fields: Fields) -> bool:
    return isinstance(fields.get("model"), str)


def accept(fields: Fields) -> bool:
    return True


NativeHook: TypeAlias = Callable[[NativeT, NativeCall, tuple[object, ...], Mapping[str, object]], ResultT]
Provider: TypeAlias = Callable[[Fields], str | None]


def _context(route: Route, provider: Provider, request: NativeCall) -> RouteContext:
    return RouteContext(route, provider=provider(request.bound), model=optional_str(request.bound.get("model")))


@dataclass(frozen=True, slots=True)
class PublicDispatch:
    """A public entrypoint with a Python implementation whose calls may run natively.

    ``bind`` maps the public ``(*args, **kwargs)`` onto the Python implementation's named
    arguments, or ``None`` when they do not fit, so Python raises its own error.
    ``internal_hop`` names the kwarg Python's async entrypoint sets when it re-enters the sync
    one; that call has already been dispatched and never reaches the bridge again.
    ``accepts`` keeps a bound call on Python when it lacks what the native route needs.
    ``provider`` reads the provider the catalog decides on from the bound fields."""

    route: Route
    bind: Bind
    internal_hop: str | None = None
    accepts: Callable[[Fields], bool] = accept
    provider: Provider = provider_kwarg

    def request(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall | None:
        if self.internal_hop is not None and kwargs.get(self.internal_hop) is True:
            return None
        fields: Final = self.bind(args, kwargs)
        if fields is None or not self.accepts(fields):
            return None
        return native_call(args, kwargs, fields)

    def context(self, request: NativeCall) -> RouteContext:
        return _context(self.route, self.provider, request)

    def run(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        python: Callable[..., ResultT],
        binding: NativeBinding[NativeT],
        native: NativeHook[NativeT, ResultT],
        policy: Policy | Decision | None = None,
    ) -> ResultT:
        request: Final = self.request(args, kwargs)
        if request is None:
            return python(*args, **kwargs)
        return runtime.run(
            self.context(request),
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            python=lambda: python(*args, **kwargs),
            policy=policy,
        )

    async def arun(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        python: Callable[..., Awaitable[ResultT]],
        binding: NativeBinding[NativeT],
        native: NativeHook[NativeT, Awaitable[ResultT]],
        policy: Policy | Decision | None = None,
    ) -> ResultT:
        request: Final = self.request(args, kwargs)
        if request is None:
            return await python(*args, **kwargs)
        return await runtime.arun(
            self.context(request),
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            python=lambda: python(*args, **kwargs),
            policy=policy,
        )


@dataclass(frozen=True, slots=True)
class NativeDispatch:
    """A public entrypoint with no Python implementation: every call is bound and handed to
    native, which validates it. ``bind`` raises the public ``TypeError`` for a call that does
    not fit the entrypoint's signature."""

    route: Route
    bind: Callable[[tuple[object, ...], Mapping[str, object]], Fields]
    provider: Provider = provider_kwarg

    def request(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall:
        return native_call(args, kwargs, self.bind(args, kwargs))

    def context(self, request: NativeCall) -> RouteContext:
        return _context(self.route, self.provider, request)

    def run(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        binding: NativeBinding[NativeT],
        native: NativeHook[NativeT, ResultT],
        policy: Policy | Decision | None = None,
    ) -> ResultT:
        request: Final = self.request(args, kwargs)
        return runtime.run_native(
            self.context(request),
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            policy=policy,
        )

    async def arun(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        binding: NativeBinding[NativeT],
        native: NativeHook[NativeT, Awaitable[ResultT]],
        policy: Policy | Decision | None = None,
    ) -> ResultT:
        request: Final = self.request(args, kwargs)
        return await runtime.arun_native(
            self.context(request),
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            policy=policy,
        )
