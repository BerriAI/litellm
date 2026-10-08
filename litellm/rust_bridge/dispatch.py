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
NativeHook: TypeAlias = Callable[[NativeT, NativeCall, tuple[object, ...], Mapping[str, object]], ResultT]


def provider_kwarg(fields: Fields) -> str | None:
    return optional_str(fields.get("custom_llm_provider"))


def model_is_named(fields: Fields) -> bool:
    return isinstance(fields.get("model"), str)


def accept(fields: Fields) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class PublicDispatch:
    """One public entrypoint whose calls may run natively.

    ``bind`` maps the public ``(*args, **kwargs)`` onto the Python implementation's named
    arguments, or ``None`` when they do not fit, so Python raises its own error.
    ``internal_hop`` names the kwarg Python's async entrypoint sets when it re-enters the sync
    one; that call has already been dispatched and never reaches the bridge again.
    ``accepts`` keeps a bound call on Python when it lacks what the native route needs; a route
    without a Python implementation accepts everything and lets native validation reject it.
    ``provider`` reads the provider the catalog decides on from the bound fields."""

    route: Route
    bind: Bind
    internal_hop: str | None = None
    accepts: Callable[[Fields], bool] = accept
    provider: Callable[[Fields], str | None] = provider_kwarg

    def request(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall | None:
        if self.internal_hop is not None and kwargs.get(self.internal_hop) is True:
            return None
        fields: Final = self.bind(args, kwargs)
        if fields is None or not self.accepts(fields):
            return None
        return native_call(args, kwargs, fields)

    def context(self, request: NativeCall) -> RouteContext:
        return RouteContext(
            self.route, provider=self.provider(request.bound), model=optional_str(request.bound.get("model"))
        )

    def _native_request(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall:
        request: Final = self.request(args, kwargs)
        if request is None:
            raise runtime.NoPythonImplementationError(
                f"{self.route.value} has no Python implementation, so every call must project to a native request"
            )
        return request

    def run(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        python: Callable[..., ResultT] | runtime.NoPythonImplementation,
        binding: NativeBinding[NativeT],
        native: NativeHook[NativeT, ResultT],
        policy: Policy | Decision | None = None,
    ) -> ResultT:
        if isinstance(python, runtime.NoPythonImplementation):
            native_request: Final = self._native_request(args, kwargs)
            return runtime.run(
                self.context(native_request),
                binding=binding,
                native=lambda hook: native(hook, native_request, args, kwargs),
                python=python,
                policy=policy,
            )
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
        python: Callable[..., Awaitable[ResultT]] | runtime.NoPythonImplementation,
        binding: NativeBinding[NativeT],
        native: NativeHook[NativeT, Awaitable[ResultT]],
        policy: Policy | Decision | None = None,
    ) -> ResultT:
        if isinstance(python, runtime.NoPythonImplementation):
            native_request: Final = self._native_request(args, kwargs)
            return await runtime.arun(
                self.context(native_request),
                binding=binding,
                native=lambda hook: native(hook, native_request, args, kwargs),
                python=python,
                policy=policy,
            )
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
