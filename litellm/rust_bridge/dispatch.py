from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import (  # noqa: TID251  # a legacy sync entrypoint returns a value or its coroutine
    Final,
    Generic,
    TypeAlias,
    cast,
)

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
SettledT: Final = TypeVar("SettledT")

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


@dataclass(frozen=True, slots=True)
class AsyncVariant(Generic[RequestT, AsyncResultT]):
    flag: str
    binding: NativeBinding[Callable[[RequestT], Awaitable[AsyncResultT]]]


async def _settled(value: Awaitable[SettledT] | SettledT) -> SettledT:
    if inspect.isawaitable(value):
        return await cast(
            "Awaitable[SettledT]", value
        )  # cast-ok: legacy sync entrypoints return a value or its coroutine
    return value


@dataclass(frozen=True, slots=True)
class PublicRoute(Generic[RequestT, NativeResultT, ResultT, AsyncResultT]):
    route: Route
    request: Project[RequestT]
    context: Callable[[RequestT], RouteContext]
    binding: NativeBinding[Callable[[RequestT], NativeResultT]]
    python: Callable[..., ResultT | Coroutine[object, object, ResultT]] | runtime.NoPythonImplementation
    asynchronous: AsyncVariant[RequestT, AsyncResultT] | None = None

    def run(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> (
        ResultT | NativeResultT | Coroutine[object, object, ResultT] | Coroutine[object, object, AsyncResultT | ResultT]
    ):
        selected_rules: Final = catalog.active_rules() if rules is None else rules
        with catalog.scoped_rules(selected_rules):
            variant: Final = self.asynchronous
            if variant is not None and kwargs.get(variant.flag) is True:
                return self._run_flagged(variant, args, kwargs, selected_rules)
            return self._run(args, kwargs, selected_rules)

    def _run_flagged(
        self,
        variant: AsyncVariant[RequestT, AsyncResultT],
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        selected_rules: Rules,
    ) -> ResultT | Coroutine[object, object, ResultT] | Coroutine[object, object, AsyncResultT | ResultT]:
        unflagged: Final = MappingProxyType({name: value for name, value in kwargs.items() if name != variant.flag})
        python: Final = self.python
        if isinstance(python, runtime.NoPythonImplementation):
            required: Final = _native_request(self.route, self.request(args, unflagged))
            return self._anative(self.context(required), variant, required, python, selected_rules)
        request: Final = _projected(self.route, self.request, args, unflagged, selected_rules)
        if request is None:
            return python(*args, **kwargs)
        context: Final = self.context(request)
        if catalog.decision(context, selected_rules) is Decision.PYTHON:
            return python(*args, **kwargs)
        return self._anative(context, variant, request, lambda: _settled(python(*args, **kwargs)), selected_rules)

    async def _anative(
        self,
        context: RouteContext,
        variant: AsyncVariant[RequestT, AsyncResultT],
        request: RequestT,
        python: Callable[[], Awaitable[ResultT]] | runtime.NoPythonImplementation,
        selected_rules: Rules,
    ) -> AsyncResultT | ResultT:
        with catalog.scoped_rules(selected_rules):
            return await runtime.arun(
                context,
                binding=variant.binding,
                native=lambda hook: hook(request),
                python=python,
                rules=selected_rules,
            )

    def _run(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], selected_rules: Rules
    ) -> ResultT | NativeResultT | Coroutine[object, object, ResultT]:
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
