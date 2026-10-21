from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Generic, TypeAlias, TypeVar

from litellm._logging import verbose_logger
from litellm.rust_bridge import catalog, runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, Rules

RequestT: Final = TypeVar("RequestT")
NativeT: Final = TypeVar("NativeT")
ResultT: Final = TypeVar("ResultT")

NativeHook: TypeAlias = Callable[[RequestT, tuple[object, ...], Mapping[str, object]], ResultT]


def call_hook(
    hook: NativeHook[RequestT, ResultT],
    request: RequestT,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> ResultT:
    return hook(request, args, kwargs)


@dataclass(frozen=True, slots=True)
class PublicDispatch(Generic[RequestT]):
    route: Route
    request: Callable[[tuple[object, ...], Mapping[str, object]], RequestT | None]
    context: Callable[[RequestT], RouteContext]

    def _native_request(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> RequestT:
        request: Final = self.request(args, kwargs)
        if request is None:
            raise runtime.NoPythonImplementationError(
                f"{self.route.value} has no Python implementation, so every call must project to a native request"
            )
        return request

    def _guarded_context(self, request: RequestT) -> RouteContext | None:
        try:
            return self.context(request)
        except Exception:  # noqa: BLE001  # detectors call legacy handlers, so any error must fall back to Python
            verbose_logger.warning(
                "%s route detection failed, serving the call on Python", self.route.value, exc_info=True
            )
            return None

    def run(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        python: Callable[..., ResultT] | runtime.NoPythonImplementation,
        binding: NativeBinding[NativeT],
        native: Callable[[NativeT, RequestT, tuple[object, ...], Mapping[str, object]], ResultT],
        rules: Rules | None = None,
    ) -> ResultT:
        selected_rules: Final = catalog.RULES if rules is None else rules
        if isinstance(python, runtime.NoPythonImplementation):
            native_request: Final = self._native_request(args, kwargs)
            return runtime.run(
                self.context(native_request),
                binding=binding,
                native=lambda hook: native(hook, native_request, args, kwargs),
                python=python,
                rules=selected_rules,
            )
        if not catalog.rust_might_handle(self.route, selected_rules):
            return python(*args, **kwargs)
        request: Final = self.request(args, kwargs)
        if request is None:
            return python(*args, **kwargs)
        context: Final = self._guarded_context(request)
        if context is None:
            return python(*args, **kwargs)
        return runtime.run(
            context,
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )

    async def arun(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        python: Callable[..., Awaitable[ResultT]] | runtime.NoPythonImplementation,
        binding: NativeBinding[NativeT],
        native: Callable[[NativeT, RequestT, tuple[object, ...], Mapping[str, object]], Awaitable[ResultT]],
        rules: Rules | None = None,
    ) -> ResultT:
        selected_rules: Final = catalog.RULES if rules is None else rules
        if isinstance(python, runtime.NoPythonImplementation):
            native_request: Final = self._native_request(args, kwargs)
            return await runtime.arun(
                self.context(native_request),
                binding=binding,
                native=lambda hook: native(hook, native_request, args, kwargs),
                python=python,
                rules=selected_rules,
            )
        if not catalog.rust_might_handle(self.route, selected_rules):
            return await python(*args, **kwargs)
        request: Final = self.request(args, kwargs)
        if request is None:
            return await python(*args, **kwargs)
        context: Final = self._guarded_context(request)
        if context is None:
            return await python(*args, **kwargs)
        return await runtime.arun(
            context,
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )
