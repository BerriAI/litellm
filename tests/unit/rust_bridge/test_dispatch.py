from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from typing import Final, NoReturn, TypeAlias

import pytest

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Decision, Python, Route, RouteContext, Rust
from litellm.rust_bridge.dispatch import Fields, PublicDispatch
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.runtime import NO_PYTHON, NoPythonImplementationError

PYTHON: Final = Python("test keeps the call on Python")
REQUIRED: Final = Rust(required=True)
Args: TypeAlias = tuple[object, ...]
Kwargs: TypeAlias = Mapping[str, object]


def bind_model(args: Args, kwargs: Kwargs) -> Fields:
    return {"model": args[0], **kwargs}


def unbindable(args: Args, kwargs: Kwargs) -> None:
    return None


def unused_binding() -> NativeBinding[object]:
    bound: Final[NativeBinding[object]] = NativeBinding("unused", validate=lambda value: value)
    bound.override(None)
    return bound


NativeRoute: TypeAlias = Callable[[NativeCall, Args, Kwargs], object]


def native_route(result: object) -> NativeBinding[NativeRoute]:
    bound: Final[NativeBinding[NativeRoute]] = NativeBinding("native", validate=lambda _: None)
    bound.override(lambda request, args, kwargs: (result, request.bound, args, dict(kwargs)))
    return bound


def reject_native(hook: object, request: NativeCall, args: Args, kwargs: Kwargs) -> NoReturn:
    pytest.fail("this call must not reach native")


def test_projects_the_bound_fields_into_the_native_call() -> None:
    dispatch: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=bind_model)
    result: Final = object()

    dispatched: Final = dispatch.run(
        ("model",),
        {"stream": True},
        python=lambda *args, **kwargs: pytest.fail("a required Rust decision must not call Python"),
        binding=native_route(result),
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        policy=REQUIRED,
    )
    assert dispatched == (result, {"model": "model", "stream": True}, ("model",), {"stream": True})


@pytest.mark.parametrize("policy", (PYTHON, Rust()))
def test_unbindable_call_forwards_the_original_call_to_python(policy: Decision) -> None:
    seen: Final[list[tuple[Args, dict[str, object]]]] = []
    dispatch: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=unbindable)
    expected: Final = object()

    def python(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        seen.append((args, dict(kwargs)))
        return expected

    result: Final = dispatch.run(
        ("model",), {"stream": True}, python=python, binding=unused_binding(), native=reject_native, policy=policy
    )
    assert result is expected
    assert seen == [(("model",), {"stream": True})]


def test_internal_hop_marker_stays_on_python_without_binding() -> None:
    def reject_bind(args: Args, kwargs: Kwargs) -> Fields:
        pytest.fail("Python's internal hop must not be bound")

    dispatch: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=reject_bind, internal_hop="acompletion")
    expected: Final = object()

    result: Final = dispatch.run(
        ("model",),
        {"acompletion": True},
        python=lambda *args, **kwargs: expected,
        binding=unused_binding(),
        native=reject_native,
        policy=REQUIRED,
    )
    assert result is expected


def test_context_leaves_a_non_string_model_unnamed() -> None:
    dispatch: Final = PublicDispatch(Route.OCR, bind=bind_model)
    request: Final = dispatch.request((None,), {})

    assert request is not None
    assert dispatch.context(request) == RouteContext(Route.OCR, provider=None, model=None)


def test_rejected_fields_stay_on_python() -> None:
    dispatch: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=bind_model, accepts=lambda fields: False)
    expected: Final = object()

    result: Final = dispatch.run(
        ("model",),
        {},
        python=lambda *args, **kwargs: expected,
        binding=unused_binding(),
        native=reject_native,
        policy=REQUIRED,
    )
    assert result is expected


def test_policy_sees_the_route_provider_and_model_of_the_bound_call() -> None:
    seen: Final[list[RouteContext]] = []
    dispatch: Final = PublicDispatch(Route.MESSAGES, bind=bind_model, provider=lambda fields: "anthropic")

    def policy(context: RouteContext) -> Decision:
        seen.append(context)
        return PYTHON

    dispatch.run(
        ("claude",),
        {},
        python=lambda *args, **kwargs: None,
        binding=unused_binding(),
        native=reject_native,
        policy=policy,
    )
    assert seen == [RouteContext(Route.MESSAGES, provider="anthropic", model="claude")]


def test_native_stream_result_is_not_consumed_or_wrapped() -> None:
    stream: Final[Iterator[int]] = iter((1, 2))
    dispatch: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=bind_model)
    native_binding: Final[NativeBinding[NativeRoute]] = NativeBinding("stream", validate=lambda _: None)
    native_binding.override(lambda request, args, kwargs: stream)

    result: Final = dispatch.run(
        ("streaming-model",),
        {"stream": True},
        python=lambda *args, **kwargs: pytest.fail("Required native stream dispatch must not call Python"),
        binding=native_binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        policy=REQUIRED,
    )
    assert result is stream


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", (PYTHON, Rust()))
async def test_async_unbindable_call_preserves_async_iterator_result(policy: Decision) -> None:
    async def chunks() -> AsyncGenerator[int, None]:
        yield 1

    stream: Final = chunks()

    async def python(*args: object, **kwargs: object) -> AsyncGenerator[int, None]:  # kwargs-ok: pass-through shape
        return stream

    dispatch: Final = PublicDispatch(Route.RESPONSES, bind=unbindable)
    result: Final = await dispatch.arun(
        ("model",), {"stream": True}, python=python, binding=unused_binding(), native=reject_native, policy=policy
    )
    assert result is stream
    await stream.aclose()


@pytest.mark.asyncio
async def test_async_dispatch_accepts_websocket_style_none_result() -> None:
    dispatch: Final = PublicDispatch(Route.RESPONSES, bind=bind_model)

    async def python(*args: object, **kwargs: object) -> None:  # kwargs-ok: public pass-through shape
        pytest.fail("Required native WebSocket dispatch must not call Python")

    async def native(request: NativeCall, args: Args, kwargs: Kwargs) -> None:
        return None

    native_binding: Final[NativeBinding[Callable[[NativeCall, Args, Kwargs], Awaitable[None]]]] = NativeBinding(
        "websocket", validate=lambda _: None
    )
    native_binding.override(native)

    result: Final = await dispatch.arun(
        ("realtime-model",),
        {},
        python=python,
        binding=native_binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        policy=REQUIRED,
    )
    assert result is None


async def dispatch_without_python(
    dispatch: PublicDispatch, bound: NativeBinding[NativeRoute], policy: Decision, *, asynchronous: bool
) -> object:
    if not asynchronous:
        return dispatch.run(
            ("model",),
            {"page": True},
            python=NO_PYTHON,
            binding=bound,
            native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
            policy=policy,
        )

    async def native(hook: NativeRoute, request: NativeCall, args: Args, kwargs: Kwargs) -> object:
        return hook(request, args, kwargs)

    return await dispatch.arun(
        ("model",), {"page": True}, python=NO_PYTHON, binding=bound, native=native, policy=policy
    )


@pytest.mark.parametrize("asynchronous", (False, True))
async def test_dispatch_without_python_hands_every_call_to_native(asynchronous: bool) -> None:
    result: Final = object()
    dispatched: Final = await dispatch_without_python(
        PublicDispatch(Route.OCR, bind=bind_model), native_route(result), REQUIRED, asynchronous=asynchronous
    )

    assert dispatched == (result, {"model": "model", "page": True}, ("model",), {"page": True})


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize(
    ("dispatch", "policy", "reason"),
    (
        (PublicDispatch(Route.OCR, bind=bind_model), PYTHON, "must select required Rust"),
        (PublicDispatch(Route.OCR, bind=bind_model), Rust(), "must select required Rust"),
        (PublicDispatch(Route.OCR, bind=unbindable), REQUIRED, "must project to a native request"),
        (PublicDispatch(Route.OCR, bind=bind_model, internal_hop="page"), REQUIRED, "must project to a native request"),
    ),
    ids=("python-decision", "optional-decision", "unbindable-call", "internal-hop"),
)
async def test_dispatch_without_python_never_falls_back(
    asynchronous: bool, dispatch: PublicDispatch, policy: Decision, reason: str
) -> None:
    bound: Final[NativeBinding[NativeRoute]] = NativeBinding("no_python", validate=lambda _: None)
    bound.override(lambda request, args, kwargs: pytest.fail("a misdeclared route must not reach native"))

    with pytest.raises(NoPythonImplementationError, match=f"ocr has no Python implementation, so .*{reason}"):
        await dispatch_without_python(dispatch, bound, policy, asynchronous=asynchronous)
