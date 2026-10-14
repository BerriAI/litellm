from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, NoReturn, TypeAlias

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import AsyncPublicRoute, PublicRoute
from litellm.rust_bridge.runtime import NO_PYTHON, NoPythonImplementationError


@dataclass(frozen=True, slots=True)
class Request:
    model: str


NativeRoute: TypeAlias = Callable[[Request], object]
AsyncNativeRoute: TypeAlias = Callable[[Request], Awaitable[object]]


def failing(reason: str) -> NativeBinding[NativeRoute]:
    bound: Final[NativeBinding[NativeRoute]] = NativeBinding("failing", validate=lambda _: None)
    bound.override(lambda _: pytest.fail(reason))
    return bound


def afailing(reason: str) -> NativeBinding[AsyncNativeRoute]:
    bound: Final[NativeBinding[AsyncNativeRoute]] = NativeBinding("failing", validate=lambda _: None)
    bound.override(lambda _: pytest.fail(reason))
    return bound


def reject_request(reason: str) -> Callable[[tuple[object, ...], Mapping[str, object]], NoReturn]:
    return lambda args, kwargs: pytest.fail(reason)


@pytest.mark.parametrize("rules", ((), (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)))
def test_route_without_rules_forwards_before_request_projection(rules: Rules) -> None:
    stream: Final[Iterator[int]] = iter((1, 2))
    route: Final = PublicRoute(
        route=Route.CHAT_COMPLETIONS,
        request=reject_request("Python-only routes must not project the request"),
        context=lambda _: RouteContext(Route.CHAT_COMPLETIONS),
        binding=failing("Python-only routes must not call native"),
        python=lambda *args, **kwargs: stream,
    )
    assert route.run(("model",), {"stream": True}, rules=rules) is stream


def test_unconditional_python_rule_prevents_later_rust_rule_projection() -> None:
    rules: Final[Rules] = (
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),
    )
    expected: Final = object()
    route: Final = PublicRoute(
        route=Route.CHAT_COMPLETIONS,
        request=reject_request("First-match Python rule must prevent request projection"),
        context=lambda _: RouteContext(Route.CHAT_COMPLETIONS),
        binding=failing("First-match Python rule must prevent native"),
        python=lambda *args, **kwargs: expected,
    )
    assert route.run(("model",), {}, rules=rules) is expected


def test_disabled_optional_rust_rule_forwards_before_projection() -> None:
    rules: Final[Rules] = (RouteRule(Route.OCR, Rollout.RUST_OPT_OUT),)
    expected: Final = object()
    route: Final = PublicRoute(
        route=Route.OCR,
        request=reject_request("Disabled optional Rust must not project the request"),
        context=lambda _: RouteContext(Route.OCR),
        binding=failing("Disabled optional Rust must not call native"),
        python=lambda *args, **kwargs: expected,
    )
    configuration.rust(False)
    try:
        result: Final = route.run(("model",), {}, rules=rules)
    finally:
        configuration.rust(None)
    assert result is expected


def test_native_stream_result_is_not_consumed_or_wrapped() -> None:
    request: Final = Request(model="streaming-model")
    stream: Final[Iterator[int]] = iter((1, 2))
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),
    )
    native_binding: Final[NativeBinding[Callable[[Request], Iterator[int]]]] = NativeBinding(
        "stream", validate=lambda _: None
    )
    native_binding.override(lambda value: stream if value is request else pytest.fail("native got another request"))
    route: Final = PublicRoute(
        route=Route.CHAT_COMPLETIONS,
        request=lambda args, kwargs: request,
        context=lambda value: RouteContext(Route.CHAT_COMPLETIONS, model=value.model),
        binding=native_binding,
        python=lambda *args, **kwargs: pytest.fail("Required native stream dispatch must not call Python"),
    )
    assert route.run(("streaming-model",), {"stream": True}, rules=rules) is stream


@pytest.mark.asyncio
@pytest.mark.parametrize("rules", ((), (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)))
async def test_async_route_without_rules_preserves_async_iterator_result(rules: Rules) -> None:
    async def chunks() -> AsyncGenerator[int, None]:
        yield 1

    stream: Final = chunks()

    async def python(*args: object, **kwargs: object) -> AsyncGenerator[int, None]:  # kwargs-ok: pass-through shape
        return stream

    route: Final = AsyncPublicRoute(
        route=Route.RESPONSES,
        request=reject_request("Python-only routes must not project the request"),
        context=lambda _: RouteContext(Route.RESPONSES),
        binding=afailing("Python-only routes must not call native"),
        python=python,
    )
    assert await route.arun(("model",), {"stream": True}, rules=rules) is stream
    await stream.aclose()


@pytest.mark.asyncio
async def test_async_dispatch_accepts_websocket_style_none_result() -> None:
    request: Final = Request(model="realtime-model")
    rules: Final[Rules] = (RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),)

    async def python(*args: object, **kwargs: object) -> None:  # kwargs-ok: public pass-through shape
        pytest.fail("Required native WebSocket dispatch must not call Python")

    async def native(value: Request) -> None:
        assert value is request

    native_binding: Final[NativeBinding[Callable[[Request], Awaitable[None]]]] = NativeBinding(
        "websocket", validate=lambda _: None
    )
    native_binding.override(native)
    route: Final = AsyncPublicRoute(
        route=Route.RESPONSES,
        request=lambda args, kwargs: request,
        context=lambda value: RouteContext(Route.RESPONSES, model=value.model),
        binding=native_binding,
        python=python,
    )
    assert await route.arun(("realtime-model",), {}, rules=rules) is None


def test_rules_for_other_routes_and_constrained_python_rules_skip_projection() -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY, providers=frozenset({"mistral"})),
    )
    expected: Final = object()
    route: Final = PublicRoute(
        route=Route.OCR,
        request=reject_request("Rules that cannot select Rust must not project the request"),
        context=lambda _: RouteContext(Route.OCR),
        binding=failing("Rules that cannot select Rust must not call native"),
        python=lambda *args, **kwargs: expected,
    )
    assert route.run(("model",), {}, rules=rules) is expected


@pytest.mark.asyncio
async def test_async_bypass_forwards_to_python_without_native() -> None:
    request: Final = Request(model="bypassed-model")
    rules: Final[Rules] = (RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),)
    expected: Final = object()

    async def python(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        return expected

    route: Final = AsyncPublicRoute(
        route=Route.RESPONSES,
        request=lambda args, kwargs: request,
        context=lambda value: RouteContext(Route.RESPONSES, model=value.model),
        binding=afailing("Bypassed requests must not call native"),
        python=python,
        bypass=lambda value: value.model == "bypassed-model",
    )
    assert await route.arun(("bypassed-model",), {}, rules=rules) is expected


async def run_without_python(
    request: Request | None, native: NativeRoute, rules: Rules, *, bypass: bool = False, asynchronous: bool
) -> object:
    def project(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request | None:
        return request

    if not asynchronous:
        bound: Final[NativeBinding[NativeRoute]] = NativeBinding("no_python", validate=lambda _: None)
        bound.override(native)
        route: Final[PublicRoute[Request, object, NoReturn]] = PublicRoute(
            route=Route.OCR,
            request=project,
            context=lambda value: RouteContext(Route.OCR, model=value.model),
            binding=bound,
            python=NO_PYTHON,
            bypass=lambda _: bypass,
        )
        return route.run(("model",), {"page": 1}, rules=rules)

    async def anative(value: Request) -> object:
        return native(value)

    abound: Final[NativeBinding[AsyncNativeRoute]] = NativeBinding("no_python", validate=lambda _: None)
    abound.override(anative)
    aroute: Final[AsyncPublicRoute[Request, object, NoReturn]] = AsyncPublicRoute(
        route=Route.OCR,
        request=project,
        context=lambda value: RouteContext(Route.OCR, model=value.model),
        binding=abound,
        python=NO_PYTHON,
        bypass=lambda _: bypass,
    )
    return await aroute.arun(("model",), {"page": 1}, rules=rules)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("switch", (None, False))
async def test_dispatch_without_python_hands_every_call_to_native(asynchronous: bool, switch: bool | None) -> None:
    request: Final = Request(model="model")
    result: Final = object()
    configuration.rust(switch)
    try:
        dispatched: Final = await run_without_python(
            request,
            lambda value: (result, value),
            (RouteRule(Route.OCR, Rollout.RUST_REQUIRED),),
            asynchronous=asynchronous,
        )
    finally:
        configuration.rust(None)

    assert dispatched == (result, request)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize(
    ("request_value", "bypass", "rules", "reason"),
    (
        (Request(model="model"), False, (), "must resolve to RUST_REQUIRED"),
        (Request(model="model"), False, (RouteRule(Route.OCR, Rollout.RUST_OPT_OUT),), "must resolve to RUST_REQUIRED"),
        (None, False, (RouteRule(Route.OCR, Rollout.RUST_REQUIRED),), "must project to a native request"),
        (Request(model="model"), True, (RouteRule(Route.OCR, Rollout.RUST_REQUIRED),), "bypass predicate matches"),
    ),
    ids=("no-rule", "opt-out-rule", "unprojectable-call", "bypassed-call"),
)
async def test_dispatch_without_python_never_falls_back(
    asynchronous: bool, request_value: Request | None, bypass: bool, rules: Rules, reason: str
) -> None:
    with pytest.raises(NoPythonImplementationError, match=f"ocr has no Python implementation, so .*{reason}"):
        await run_without_python(
            request_value,
            lambda _: pytest.fail("a misdeclared route must not reach native"),
            rules,
            bypass=bypass,
            asynchronous=asynchronous,
        )


async def run_messages_route(rules: Rules, *, asynchronous: bool) -> object:
    def project(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        return Request(model=str(args[0]))

    def context(value: Request) -> RouteContext:
        return RouteContext(Route.MESSAGES, model=value.model)

    if not asynchronous:
        bound: Final[NativeBinding[NativeRoute]] = NativeBinding("messages", validate=lambda _: None)
        bound.override(lambda value: ("rust", value.model))
        return PublicRoute(
            route=Route.MESSAGES,
            request=project,
            context=context,
            binding=bound,
            python=lambda *args, **kwargs: ("python", args, dict(kwargs)),
        ).run(("bound-model",), {"page": 1}, rules=rules)

    async def native(value: Request) -> object:
        return ("rust", value.model)

    async def python(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        return ("python", args, dict(kwargs))

    abound: Final[NativeBinding[AsyncNativeRoute]] = NativeBinding("messages", validate=lambda _: None)
    abound.override(native)
    return await AsyncPublicRoute(
        route=Route.MESSAGES, request=project, context=context, binding=abound, python=python
    ).arun(("bound-model",), {"page": 1}, rules=rules)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize(
    ("rules", "expected"),
    (
        ((RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),), ("rust", "bound-model")),
        ((RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),), ("python", ("bound-model",), {"page": 1})),
    ),
    ids=("rust", "python"),
)
async def test_public_route_selects_a_backend_with_the_given_rules(
    rules: Rules, expected: object, asynchronous: bool
) -> None:
    assert await run_messages_route(rules, asynchronous=asynchronous) == expected
