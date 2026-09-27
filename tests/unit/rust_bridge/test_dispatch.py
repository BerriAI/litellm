from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import CacheRule, Delivery, Route, RouteContext, RouteRule, Rules, SecretManagerRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.runtime import NO_PYTHON, NoPythonImplementationError


@dataclass(frozen=True, slots=True)
class Request:
    model: str


def binding() -> NativeBinding[object]:
    bound: Final[NativeBinding[object]] = NativeBinding("unused", validate=lambda value: value)
    bound.override(None)
    return bound


@pytest.mark.parametrize("rules", ((), (CacheRule(Rollout.RUST_REQUIRED), SecretManagerRule(Rollout.RUST_REQUIRED))))
def test_route_without_rules_forwards_before_request_projection(rules: Rules) -> None:
    stream: Final[Iterator[int]] = iter((1, 2))

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("Python-only routes must not project the request")

    dispatch: Final = PublicDispatch(
        route=Route.CHAT_COMPLETIONS, request=reject_request, context=lambda _: RouteContext(Route.CHAT_COMPLETIONS)
    )
    result: Final = dispatch.run(
        ("model",),
        {"stream": True},
        python=lambda *args, **kwargs: stream,
        binding=binding(),
        native=lambda hook, request, args, kwargs: pytest.fail("Python-only routes must not call native"),
        rules=rules,
    )
    assert result is stream


def test_unconditional_python_rule_prevents_later_rust_rule_projection() -> None:
    rules: Final[Rules] = (
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),
    )

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("First-match Python rule must prevent request projection")

    dispatch: Final = PublicDispatch(
        route=Route.CHAT_COMPLETIONS,
        request=reject_request,
        context=lambda _: RouteContext(Route.CHAT_COMPLETIONS),
    )
    expected: Final = object()
    result: Final = dispatch.run(
        ("model",),
        {},
        python=lambda *args, **kwargs: expected,
        binding=binding(),
        native=lambda hook, request, args, kwargs: pytest.fail("First-match Python rule must prevent native"),
        rules=rules,
    )
    assert result is expected


def test_disabled_optional_rust_rule_forwards_before_projection() -> None:
    rules: Final[Rules] = (RouteRule(Route.OCR, Rollout.RUST_OPT_OUT),)

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("Disabled optional Rust must not project the request")

    dispatch: Final = PublicDispatch(route=Route.OCR, request=reject_request, context=lambda _: RouteContext(Route.OCR))
    expected: Final = object()
    configuration.rust(False)
    try:
        result: Final = dispatch.run(
            ("model",),
            {},
            python=lambda *args, **kwargs: expected,
            binding=binding(),
            native=lambda hook, request, args, kwargs: pytest.fail("Disabled optional Rust must not call native"),
            rules=rules,
        )
    finally:
        configuration.rust(None)
    assert result is expected


def test_native_stream_result_is_not_consumed_or_wrapped() -> None:
    request: Final = Request(model="streaming-model")
    stream: Final[Iterator[int]] = iter((1, 2))
    rules: Final[Rules] = (
        CacheRule(Rollout.PYTHON_ONLY),
        SecretManagerRule(Rollout.PYTHON_ONLY),
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED, deliveries=frozenset({Delivery.STREAMING})),
    )
    dispatch: Final = PublicDispatch(
        route=Route.CHAT_COMPLETIONS,
        request=lambda args, kwargs: request,
        context=lambda value: RouteContext(Route.CHAT_COMPLETIONS, model=value.model, delivery=Delivery.STREAMING),
    )

    def native(request: Request, args: tuple[object, ...], kwargs: Mapping[str, object]) -> Iterator[int]:
        return stream

    native_binding: Final[
        NativeBinding[Callable[[Request, tuple[object, ...], Mapping[str, object]], Iterator[int]]]
    ] = NativeBinding("stream", validate=lambda _: None)
    native_binding.override(native)
    result: Final = dispatch.run(
        ("streaming-model",),
        {"stream": True},
        python=lambda *args, **kwargs: pytest.fail("Required native stream dispatch must not call Python"),
        binding=native_binding,
        native=lambda hook, value, args, kwargs: hook(value, args, kwargs),
        rules=rules,
    )
    assert result is stream


@pytest.mark.asyncio
@pytest.mark.parametrize("rules", ((), (CacheRule(Rollout.RUST_REQUIRED), SecretManagerRule(Rollout.RUST_REQUIRED))))
async def test_async_route_without_rules_preserves_async_iterator_result(rules: Rules) -> None:
    async def chunks() -> AsyncGenerator[int, None]:
        yield 1

    stream: Final = chunks()

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("Python-only routes must not project the request")

    async def python(*args: object, **kwargs: object) -> AsyncGenerator[int, None]:  # kwargs-ok: pass-through shape
        return stream

    dispatch: Final = PublicDispatch(
        route=Route.RESPONSES, request=reject_request, context=lambda _: RouteContext(Route.RESPONSES)
    )
    result: Final = await dispatch.arun(
        ("model",),
        {"stream": True},
        python=python,
        binding=binding(),
        native=lambda hook, request, args, kwargs: pytest.fail("Python-only routes must not call native"),
        rules=rules,
    )
    assert result is stream
    await stream.aclose()


@pytest.mark.asyncio
async def test_async_dispatch_accepts_websocket_style_none_result() -> None:
    request: Final = Request(model="realtime-model")
    rules: Final[Rules] = (
        RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED, deliveries=frozenset({Delivery.WEBSOCKET})),
    )
    dispatch: Final = PublicDispatch(
        route=Route.RESPONSES,
        request=lambda args, kwargs: request,
        context=lambda value: RouteContext(Route.RESPONSES, model=value.model, delivery=Delivery.WEBSOCKET),
    )

    async def python(*args: object, **kwargs: object) -> None:  # kwargs-ok: public pass-through shape
        pytest.fail("Required native WebSocket dispatch must not call Python")

    async def native(request: Request, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        return None

    native_binding: Final[
        NativeBinding[Callable[[Request, tuple[object, ...], Mapping[str, object]], Awaitable[None]]]
    ] = NativeBinding("websocket", validate=lambda _: None)
    native_binding.override(native)

    result: Final = await dispatch.arun(
        ("realtime-model",),
        {},
        python=python,
        binding=native_binding,
        native=lambda hook, value, args, kwargs: hook(value, args, kwargs),
        rules=rules,
    )
    assert result is None


def test_rules_for_other_routes_and_constrained_python_rules_skip_projection() -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
        RouteRule(Route.OCR, Rollout.PYTHON_ONLY, providers=frozenset({"mistral"})),
    )

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("Rules that cannot select Rust must not project the request")

    dispatch: Final = PublicDispatch(route=Route.OCR, request=reject_request, context=lambda _: RouteContext(Route.OCR))
    expected: Final = object()
    result: Final = dispatch.run(
        ("model",),
        {},
        python=lambda *args, **kwargs: expected,
        binding=binding(),
        native=lambda hook, request, args, kwargs: pytest.fail("Rules that cannot select Rust must not call native"),
        rules=rules,
    )
    assert result is expected


@pytest.mark.asyncio
async def test_async_bypass_forwards_to_python_without_native() -> None:
    request: Final = Request(model="bypassed-model")
    rules: Final[Rules] = (RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),)
    dispatch: Final = PublicDispatch(
        route=Route.RESPONSES,
        request=lambda args, kwargs: request,
        context=lambda value: RouteContext(Route.RESPONSES, model=value.model),
        bypass=lambda value: value.model == "bypassed-model",
    )
    expected: Final = object()

    async def python(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        return expected

    result: Final = await dispatch.arun(
        ("bypassed-model",),
        {},
        python=python,
        binding=binding(),
        native=lambda hook, value, args, kwargs: pytest.fail("Bypassed requests must not call native"),
        rules=rules,
    )
    assert result is expected


NativeRoute: TypeAlias = Callable[[Request, tuple[object, ...], Mapping[str, object]], object]


def native_route(result: object) -> NativeBinding[NativeRoute]:
    bound: Final[NativeBinding[NativeRoute]] = NativeBinding("no_python", validate=lambda _: None)
    bound.override(lambda request, args, kwargs: (result, request, args, dict(kwargs)))
    return bound


async def dispatch_without_python(
    dispatch: PublicDispatch[Request], bound: NativeBinding[NativeRoute], rules: Rules, *, asynchronous: bool
) -> object:
    if not asynchronous:
        return dispatch.run(
            ("model",),
            {"page": 1},
            python=NO_PYTHON,
            binding=bound,
            native=lambda hook, value, args, kwargs: hook(value, args, kwargs),
            rules=rules,
        )

    async def native(
        hook: NativeRoute, value: Request, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> object:
        return hook(value, args, kwargs)

    return await dispatch.arun(("model",), {"page": 1}, python=NO_PYTHON, binding=bound, native=native, rules=rules)


def ocr_dispatch(request: Request | None, *, bypass: bool = False) -> PublicDispatch[Request]:
    return PublicDispatch(
        route=Route.OCR,
        request=lambda args, kwargs: request,
        context=lambda value: RouteContext(Route.OCR, model=value.model),
        bypass=lambda _: bypass,
    )


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("switch", (None, False))
async def test_dispatch_without_python_hands_every_call_to_native(asynchronous: bool, switch: bool | None) -> None:
    request: Final = Request(model="model")
    result: Final = object()
    configuration.rust(switch)
    try:
        dispatched: Final = await dispatch_without_python(
            ocr_dispatch(request),
            native_route(result),
            (RouteRule(Route.OCR, Rollout.RUST_REQUIRED),),
            asynchronous=asynchronous,
        )
    finally:
        configuration.rust(None)

    assert dispatched == (result, request, ("model",), {"page": 1})


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
    bound: Final[NativeBinding[NativeRoute]] = NativeBinding("no_python", validate=lambda _: None)
    bound.override(lambda request, args, kwargs: pytest.fail("a misdeclared route must not reach native"))

    with pytest.raises(NoPythonImplementationError, match=f"ocr has no Python implementation, so .*{reason}"):
        await dispatch_without_python(
            ocr_dispatch(request_value, bypass=bypass), bound, rules, asynchronous=asynchronous
        )
