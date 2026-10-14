from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Generic, TypeAlias, TypeVar

from litellm.rust_bridge import catalog, runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule, Rules
from litellm.rust_bridge.configuration import Decision
from litellm.rust_bridge.configuration import decision as rollout_decision

RequestT: Final = TypeVar("RequestT")
NativeResultT: Final = TypeVar("NativeResultT")
ResultT: Final = TypeVar("ResultT")

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


def _native_request(
    route: Route,
    request: RequestT | None,
    bypass: Callable[[RequestT], bool] | None,
) -> RequestT:
    if request is None:
        raise runtime.NoPythonImplementationError(
            f"{route.value} has no Python implementation, so every call must project to a native request"
        )
    if bypass is not None and bypass(request):
        raise runtime.NoPythonImplementationError(
            f"{route.value} has no Python implementation, so a call its bypass predicate matches cannot be served"
        )
    return request


def _projected(
    route: Route,
    project: Project[RequestT],
    bypass: Callable[[RequestT], bool] | None,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    rules: Rules,
) -> RequestT | None:
    if not _requires_projection(route, rules):
        return None
    request: Final = project(args, kwargs)
    if request is None or (bypass is not None and bypass(request)):
        return None
    return request


@dataclass(frozen=True, slots=True)
class PublicRoute(Generic[RequestT, NativeResultT, ResultT]):
    route: Route
    request: Project[RequestT]
    context: Callable[[RequestT], RouteContext]
    binding: NativeBinding[Callable[[RequestT], NativeResultT]]
    python: Callable[..., ResultT] | runtime.NoPythonImplementation
    bypass: Callable[[RequestT], bool] | None = None

    def run(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> ResultT | NativeResultT:
        selected_rules: Final = catalog.RULES if rules is None else rules
        python: Final = self.python
        if isinstance(python, runtime.NoPythonImplementation):
            required: Final = _native_request(self.route, self.request(args, kwargs), self.bypass)
            return runtime.run(
                self.context(required),
                binding=self.binding,
                native=lambda hook: hook(required),
                python=python,
                rules=selected_rules,
            )
        request: Final = _projected(self.route, self.request, self.bypass, args, kwargs, selected_rules)
        if request is None:
            return python(*args, **kwargs)
        return runtime.run(
            self.context(request),
            binding=self.binding,
            native=lambda hook: hook(request),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )


@dataclass(frozen=True, slots=True)
class AsyncPublicRoute(Generic[RequestT, NativeResultT, ResultT]):
    route: Route
    request: Project[RequestT]
    context: Callable[[RequestT], RouteContext]
    binding: NativeBinding[Callable[[RequestT], Awaitable[NativeResultT]]]
    python: Callable[..., Awaitable[ResultT]] | runtime.NoPythonImplementation
    bypass: Callable[[RequestT], bool] | None = None

    async def arun(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], *, rules: Rules | None = None
    ) -> ResultT | NativeResultT:
        selected_rules: Final = catalog.RULES if rules is None else rules
        python: Final = self.python
        if isinstance(python, runtime.NoPythonImplementation):
            required: Final = _native_request(self.route, self.request(args, kwargs), self.bypass)
            return await runtime.arun(
                self.context(required),
                binding=self.binding,
                native=lambda hook: hook(required),
                python=python,
                rules=selected_rules,
            )
        request: Final = _projected(self.route, self.request, self.bypass, args, kwargs, selected_rules)
        if request is None:
            return await python(*args, **kwargs)
        return await runtime.arun(
            self.context(request),
            binding=self.binding,
            native=lambda hook: hook(request),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )
