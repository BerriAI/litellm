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
    bypass: Callable[[RequestT], bool] | None = None

    def _requires_projection(self, rules: Rules) -> bool:
        for rule in rules:
            if not isinstance(rule, RouteRule) or rule.route is not self.route:
                continue
            if rule.providers is not None or rule.models is not None or rule.deliveries is not None:
                if rollout_decision(rule.rollout) is not Decision.PYTHON:
                    return True
                continue
            return rollout_decision(rule.rollout) is not Decision.PYTHON
        return False

    def run(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        python: Callable[..., ResultT],
        binding: NativeBinding[NativeT],
        native: Callable[[NativeT, RequestT, tuple[object, ...], Mapping[str, object]], ResultT],
        rules: Rules | None = None,
    ) -> ResultT:
        selected_rules: Final = catalog.RULES if rules is None else rules
        if not self._requires_projection(selected_rules):
            return python(*args, **kwargs)
        request: Final = self.request(args, kwargs)
        if request is None or (self.bypass is not None and self.bypass(request)):
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
        python: Callable[..., Awaitable[ResultT]],
        binding: NativeBinding[NativeT],
        native: Callable[[NativeT, RequestT, tuple[object, ...], Mapping[str, object]], Awaitable[ResultT]],
        rules: Rules | None = None,
    ) -> ResultT:
        selected_rules: Final = catalog.RULES if rules is None else rules
        if not self._requires_projection(selected_rules):
            return await python(*args, **kwargs)
        request: Final = self.request(args, kwargs)
        if request is None or (self.bypass is not None and self.bypass(request)):
            return await python(*args, **kwargs)
        return await runtime.arun(
            self.context(request),
            binding=binding,
            native=lambda hook: native(hook, request, args, kwargs),
            python=lambda: python(*args, **kwargs),
            rules=selected_rules,
        )
