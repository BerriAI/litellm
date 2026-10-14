from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Generic, Protocol, TypeAlias

from typing_extensions import Never, TypeVar

from litellm.rust_bridge import catalog, runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule, Rules
from litellm.rust_bridge.configuration import Decision
from litellm.rust_bridge.configuration import decision as rollout_decision

RequestT: Final = TypeVar("RequestT")
NativeResultT: Final = TypeVar("NativeResultT")
ResultT: Final = TypeVar("ResultT")
AsyncResultT: Final = TypeVar("AsyncResultT", default=Never)
AsyncResultT_co: Final = TypeVar("AsyncResultT_co", covariant=True)

Project: TypeAlias = Callable[[tuple[object, ...], Mapping[str, object]], RequestT | None]


def _requires_projection(route: Route, rules: Rules) -> bool:
    for rule in rules:
        if not isinstance(rule, RouteRule) or rule.route is not route:
            continue
        if rule.providers is not None or rule.models is not None:
            if rollout_decision(rule.rollout) is not Decision.PYTHON:
                return True
            continue
        return rollout_decision(rule.rollout) is not Decision.PYTHON
    return False


def _native_request(route: Route, request: RequestT | None) -> RequestT:
    if request is None:
        raise runtime.NoPythonImplementationError(
            f"{route.value} has no Python implementation, so every call must project to a native request"
        )
    return request


def _projected(
    route: Route, project: Project[RequestT], args: tuple[object, ...], kwargs: Mapping[str, object], rules: Rules
) -> RequestT | None:
    return project(args, kwargs) if _requires_projection(route, rules) else None


@dataclass(frozen=True, slots=True)
class AsyncPublicRoute(Generic[RequestT, NativeResultT, ResultT]):
    route: Route
    request: Project[RequestT]
    context: Callable[[RequestT], RouteContext]
    binding: NativeBinding[Callable[[RequestT], Awaitable[NativeResultT]]]
    python: Callable[..., Awaitable[ResultT]] | runtime.NoPythonImplementation

    async def arun(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> ResultT | NativeResultT:
        selected_rules: Final = catalog.active_rules() if rules is None else rules
        with catalog.scoped_rules(selected_rules):
            return await self._arun(args, kwargs, selected_rules)

    async def _arun(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], selected_rules: Rules
    ) -> ResultT | NativeResultT:
        python: Final = self.python
        if isinstance(python, runtime.NoPythonImplementation):
            required: Final = _native_request(self.route, self.request(args, kwargs))
            return await runtime.arun(
                self.context(required),
                binding=self.binding,
                native=lambda hook: hook(required),
                python=python,
                rules=selected_rules,
            )
        request: Final = _projected(self.route, self.request, args, kwargs, selected_rules)
        if request is None:
            return await python(*args, **kwargs)
        return await runtime.arun(
            self.context(request),
            binding=self.binding,
            native=lambda hook: hook(request),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )


class AsyncRoute(Protocol[AsyncResultT_co]):
    def arun(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> Coroutine[object, object, AsyncResultT_co]: ...


@dataclass(frozen=True, slots=True)
class AsyncVariant(Generic[AsyncResultT]):
    flag: str
    route: AsyncRoute[AsyncResultT]


@dataclass(frozen=True, slots=True)
class PublicRoute(Generic[RequestT, NativeResultT, ResultT, AsyncResultT]):
    route: Route
    request: Project[RequestT]
    context: Callable[[RequestT], RouteContext]
    binding: NativeBinding[Callable[[RequestT], NativeResultT]]
    python: Callable[..., ResultT] | runtime.NoPythonImplementation
    asynchronous: AsyncVariant[AsyncResultT] | None = None

    def run(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> ResultT | NativeResultT | Coroutine[object, object, AsyncResultT]:
        selected_rules: Final = catalog.active_rules() if rules is None else rules
        variant: Final = self.asynchronous
        if variant is not None and kwargs.get(variant.flag) is True:
            unflagged: Final = MappingProxyType({name: value for name, value in kwargs.items() if name != variant.flag})
            return variant.route.arun(args, unflagged, rules=selected_rules)
        with catalog.scoped_rules(selected_rules):
            return self._run(args, kwargs, selected_rules)

    def _run(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], selected_rules: Rules
    ) -> ResultT | NativeResultT:
        python: Final = self.python
        if isinstance(python, runtime.NoPythonImplementation):
            required: Final = _native_request(self.route, self.request(args, kwargs))
            return runtime.run(
                self.context(required),
                binding=self.binding,
                native=lambda hook: hook(required),
                python=python,
                rules=selected_rules,
            )
        request: Final = _projected(self.route, self.request, args, kwargs, selected_rules)
        if request is None:
            return python(*args, **kwargs)
        return runtime.run(
            self.context(request),
            binding=self.binding,
            native=lambda hook: hook(request),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )
