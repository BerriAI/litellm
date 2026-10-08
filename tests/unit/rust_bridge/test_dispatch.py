from collections.abc import Callable, Mapping
from typing import Final, NoReturn, TypeAlias

import pytest

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Decision, Python, Route, RouteContext, Rust
from litellm.rust_bridge.dispatch import Fields, NativeDispatch, PublicDispatch
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.runtime import NoPythonImplementationError

PYTHON: Final = Python("test keeps the call on Python")
REQUIRED: Final = Rust(required=True)
Args: TypeAlias = tuple[object, ...]
Kwargs: TypeAlias = Mapping[str, object]
NativeRoute: TypeAlias = Callable[[NativeCall, Args, Kwargs], object]


def bind_model(args: Args, kwargs: Kwargs) -> Fields:
    return {"model": args[0], **kwargs}


def unbindable(args: Args, kwargs: Kwargs) -> None:
    return None


def reject_bind(args: Args, kwargs: Kwargs) -> Fields:
    pytest.fail("this call must not be bound")


def native_route(result: object) -> NativeBinding[NativeRoute]:
    bound: Final[NativeBinding[NativeRoute]] = NativeBinding("native", validate=lambda _: None)
    bound.override(lambda request, args, kwargs: (result, request.bound, args, dict(kwargs)))
    return bound


def reject_native() -> NativeBinding[NativeRoute]:
    bound: Final[NativeBinding[NativeRoute]] = NativeBinding("native", validate=lambda _: None)
    bound.override(lambda request, args, kwargs: pytest.fail("this call must not reach native"))
    return bound


def reject_python(*args: object, **kwargs: object) -> NoReturn:  # kwargs-ok: public pass-through shape
    pytest.fail("this call must not reach Python")


class PythonCalls:
    def __init__(self, result: object) -> None:
        self.result: Final = result
        self.seen: tuple[tuple[Args, Kwargs], ...] = ()

    def __call__(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        self.seen = (*self.seen, (args, kwargs))
        return self.result


async def run_public(
    dispatch: PublicDispatch,
    args: Args,
    kwargs: Kwargs,
    *,
    python: Callable[..., object],
    binding: NativeBinding[NativeRoute],
    policy: Decision | Callable[[RouteContext], Decision],
    asynchronous: bool,
) -> object:
    if not asynchronous:
        return dispatch.run(
            args,
            kwargs,
            python=python,
            binding=binding,
            native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
            policy=policy,
        )

    async def apython(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        return python(*args, **kwargs)

    async def anative(hook: NativeRoute, request: NativeCall, args: Args, kwargs: Kwargs) -> object:
        return hook(request, args, kwargs)

    return await dispatch.arun(args, kwargs, python=apython, binding=binding, native=anative, policy=policy)


async def run_native(
    dispatch: NativeDispatch, binding: NativeBinding[NativeRoute], policy: Decision, *, asynchronous: bool
) -> object:
    if not asynchronous:
        return dispatch.run(
            ("model",),
            {"page": True},
            binding=binding,
            native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
            policy=policy,
        )

    async def anative(hook: NativeRoute, request: NativeCall, args: Args, kwargs: Kwargs) -> object:
        return hook(request, args, kwargs)

    return await dispatch.arun(("model",), {"page": True}, binding=binding, native=anative, policy=policy)


@pytest.mark.parametrize("asynchronous", (False, True))
async def test_bound_call_reaches_native_with_the_bound_fields_and_the_original_call(asynchronous: bool) -> None:
    result: Final = object()

    dispatched: Final = await run_public(
        PublicDispatch(Route.CHAT_COMPLETIONS, bind=bind_model),
        ("model",),
        {"stream": True},
        python=reject_python,
        binding=native_route(result),
        policy=REQUIRED,
        asynchronous=asynchronous,
    )

    assert dispatched == (result, {"model": "model", "stream": True}, ("model",), {"stream": True})


@pytest.mark.parametrize("asynchronous", (False, True))
async def test_python_decision_forwards_the_original_call_to_python(asynchronous: bool) -> None:
    messages: Final = [{"role": "user", "content": "hi"}]
    python: Final = PythonCalls(object())

    result: Final = await run_public(
        PublicDispatch(Route.CHAT_COMPLETIONS, bind=bind_model),
        ("model", messages),
        {"stream": True},
        python=python,
        binding=reject_native(),
        policy=PYTHON,
        asynchronous=asynchronous,
    )

    assert result is python.result
    assert python.seen == ((("model", messages), {"stream": True}),)
    assert python.seen[0][0][1] is messages


@pytest.mark.parametrize("asynchronous", (False, True))
async def test_unbindable_call_forwards_the_original_call_to_python_even_when_rust_is_required(
    asynchronous: bool,
) -> None:
    python: Final = PythonCalls(object())

    result: Final = await run_public(
        PublicDispatch(Route.CHAT_COMPLETIONS, bind=unbindable),
        ("model",),
        {"stream": True},
        python=python,
        binding=reject_native(),
        policy=REQUIRED,
        asynchronous=asynchronous,
    )

    assert result is python.result
    assert python.seen == ((("model",), {"stream": True}),)


def test_internal_hop_marker_stays_on_python_without_binding() -> None:
    python: Final = PythonCalls(object())
    dispatch: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=reject_bind, internal_hop="acompletion")

    result: Final = dispatch.run(
        ("model",),
        {"acompletion": True},
        python=python,
        binding=reject_native(),
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        policy=REQUIRED,
    )

    assert result is python.result
    assert python.seen == ((("model",), {"acompletion": True}),)


def test_rejected_fields_stay_on_python() -> None:
    python: Final = PythonCalls(object())
    dispatch: Final = PublicDispatch(Route.CHAT_COMPLETIONS, bind=bind_model, accepts=lambda fields: False)

    result: Final = dispatch.run(
        ("model",),
        {},
        python=python,
        binding=reject_native(),
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        policy=REQUIRED,
    )

    assert result is python.result


@pytest.mark.parametrize("model", ("claude", None, 7))
def test_policy_sees_the_route_provider_and_named_model_of_the_bound_call(model: object) -> None:
    seen: Final[list[RouteContext]] = []
    dispatch: Final = PublicDispatch(Route.MESSAGES, bind=bind_model, provider=lambda fields: "anthropic")

    def policy(context: RouteContext) -> Decision:
        seen.append(context)
        return PYTHON

    dispatch.run(
        (model,),
        {},
        python=lambda *args, **kwargs: None,
        binding=reject_native(),
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        policy=policy,
    )

    assert seen == [RouteContext(Route.MESSAGES, provider="anthropic", model=model if isinstance(model, str) else None)]


@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_dispatch_hands_every_bound_call_to_native(asynchronous: bool) -> None:
    result: Final = object()

    dispatched: Final = await run_native(
        NativeDispatch(Route.OCR, bind=bind_model), native_route(result), REQUIRED, asynchronous=asynchronous
    )

    assert dispatched == (result, {"model": "model", "page": True}, ("model",), {"page": True})


@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_dispatch_raises_the_binder_error_before_native(asynchronous: bool) -> None:
    def reject(args: Args, kwargs: Kwargs) -> Fields:
        raise TypeError("ocr() takes a document")

    with pytest.raises(TypeError, match="takes a document"):
        await run_native(NativeDispatch(Route.OCR, bind=reject), reject_native(), REQUIRED, asynchronous=asynchronous)


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("policy", (PYTHON, Rust()), ids=("python-decision", "optional-decision"))
async def test_native_dispatch_rejects_a_policy_that_could_select_python(asynchronous: bool, policy: Decision) -> None:
    with pytest.raises(NoPythonImplementationError, match="ocr has no Python implementation, so .*required Rust"):
        await run_native(NativeDispatch(Route.OCR, bind=bind_model), reject_native(), policy, asynchronous=asynchronous)
