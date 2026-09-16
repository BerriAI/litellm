from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Context, Delivery, Route, Rule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import PublicDispatch


@dataclass(frozen=True, slots=True)
class Request:
    model: str


def binding() -> NativeBinding[object]:
    bound: Final[NativeBinding[object]] = NativeBinding("unused", validate=lambda value: value)
    bound.override(None)
    return bound


def test_route_without_rules_forwards_before_request_projection() -> None:
    stream: Final[Iterator[int]] = iter((1, 2))

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("Python-only routes must not project the request")

    dispatch: Final = PublicDispatch(route=Route.CHAT_COMPLETIONS, request=reject_request, context=lambda _: Context(Route.CHAT_COMPLETIONS))
    result: Final = dispatch.run(
        ("model",),
        {"stream": True},
        python=lambda *args, **kwargs: stream,
        binding=binding(),
        native=lambda hook, request, args, kwargs: pytest.fail("Python-only routes must not call native"),
        rules=(),
    )
    assert result is stream


def test_unconditional_python_rule_prevents_later_rust_rule_projection() -> None:
    rules: Final[Rules] = (
        Rule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
        Rule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),
    )

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("First-match Python rule must prevent request projection")

    dispatch: Final = PublicDispatch(
        route=Route.CHAT_COMPLETIONS,
        request=reject_request,
        context=lambda _: Context(Route.CHAT_COMPLETIONS),
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
    rules: Final[Rules] = (Rule(Route.OCR, Rollout.RUST_OPT_OUT),)

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("Disabled optional Rust must not project the request")

    dispatch: Final = PublicDispatch(route=Route.OCR, request=reject_request, context=lambda _: Context(Route.OCR))
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
        Rule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED, deliveries=frozenset({Delivery.STREAMING})),
    )
    dispatch: Final = PublicDispatch(
        route=Route.CHAT_COMPLETIONS,
        request=lambda args, kwargs: request,
        context=lambda value: Context(Route.CHAT_COMPLETIONS, model=value.model, delivery=Delivery.STREAMING),
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
async def test_async_route_without_rules_preserves_async_iterator_result() -> None:
    async def chunks() -> AsyncGenerator[int, None]:
        yield 1

    stream: Final = chunks()

    def reject_request(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request:
        pytest.fail("Python-only routes must not project the request")

    async def python(*args: object, **kwargs: object) -> AsyncGenerator[int, None]:  # kwargs-ok: pass-through shape
        return stream

    dispatch: Final = PublicDispatch(route=Route.RESPONSES, request=reject_request, context=lambda _: Context(Route.RESPONSES))
    result: Final = await dispatch.arun(
        ("model",),
        {"stream": True},
        python=python,
        binding=binding(),
        native=lambda hook, request, args, kwargs: pytest.fail("Python-only routes must not call native"),
        rules=(),
    )
    assert result is stream
    await stream.aclose()


@pytest.mark.asyncio
async def test_async_dispatch_accepts_websocket_style_none_result() -> None:
    request: Final = Request(model="realtime-model")
    rules: Final[Rules] = (
        Rule(Route.RESPONSES, Rollout.RUST_REQUIRED, deliveries=frozenset({Delivery.WEBSOCKET})),
    )
    dispatch: Final = PublicDispatch(
        route=Route.RESPONSES,
        request=lambda args, kwargs: request,
        context=lambda value: Context(Route.RESPONSES, model=value.model, delivery=Delivery.WEBSOCKET),
    )

    async def python(*args: object, **kwargs: object) -> None:  # kwargs-ok: public pass-through shape
        pytest.fail("Required native WebSocket dispatch must not call Python")

    async def native(request: Request, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        return None

    native_binding: Final[NativeBinding[Callable[[Request, tuple[object, ...], Mapping[str, object]], Awaitable[None]]]] = NativeBinding(
        "websocket", validate=lambda _: None
    )
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
