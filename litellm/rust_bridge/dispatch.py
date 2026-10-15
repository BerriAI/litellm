from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Generic, TypeAlias, TypeVar

from litellm.rust_bridge import catalog, runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, Rules
from litellm.rust_bridge.configuration import Decision
from litellm.rust_bridge.configuration import decision as rollout_decision

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

    def _requires_projection(self, rules: Rules) -> bool:
        for rule in rules:
            if rule.route is not self.route:
                continue
            if rule.constrained:
                if rollout_decision(rule.rollout) is not Decision.PYTHON:
                    return True
                continue
            return rollout_decision(rule.rollout) is not Decision.PYTHON
        return False

    def _native_request(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> RequestT:
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
        if not self._requires_projection(selected_rules):
            return python(*args, **kwargs)
        request: Final = self.request(args, kwargs)
        if request is None:
            return python(*args, **kwargs)
        return runtime.run(
            self.context(request),
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
        if not self._requires_projection(selected_rules):
            return await python(*args, **kwargs)
        request: Final = self.request(args, kwargs)
        if request is None:
            return await python(*args, **kwargs)
        return await runtime.arun(
            self.context(request),
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )

    def defer(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        flag: str,
        python: Callable[..., ResultT | Coroutine[object, object, ResultT]] | runtime.NoPythonImplementation,
        binding: NativeBinding[NativeT],
        native: Callable[[NativeT, RequestT, tuple[object, ...], Mapping[str, object]], Awaitable[ResultT]],
        rules: Rules | None = None,
    ) -> ResultT | Coroutine[object, object, ResultT]:
        selected_rules: Final = catalog.RULES if rules is None else rules
        native_kwargs: Final = MappingProxyType({name: value for name, value in kwargs.items() if name != flag})
        if isinstance(python, runtime.NoPythonImplementation):
            native_request: Final = self._native_request(args, native_kwargs)
            return runtime.arun(
                self.context(native_request),
                binding=binding,
                native=lambda hook: native(hook, native_request, args, native_kwargs),
                python=python,
                rules=selected_rules,
            )
        if not self._requires_projection(selected_rules):
            return python(*args, **kwargs)
        request: Final = self.request(args, native_kwargs)
        if request is None:
            return python(*args, **kwargs)
        context: Final = self.context(request)
        if catalog.decision(context, selected_rules) is Decision.PYTHON:
            return python(*args, **kwargs)
        return runtime.arun(
            context,
            binding=binding,
            native=lambda hook: native(hook, request, args, native_kwargs),
            python=lambda: _settled(python(*args, **kwargs)),
            rules=selected_rules,
        )


async def _settled(value: ResultT | Coroutine[object, object, ResultT]) -> ResultT:
    if isinstance(value, Coroutine):
        return await value  # pyright: ignore[reportUnknownVariableType]  # isinstance cannot split a bare TypeVar from Coroutine
    return value
