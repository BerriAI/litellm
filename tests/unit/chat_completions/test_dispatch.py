from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures

import pytest

import litellm
from litellm.chat_completions import dispatch
from litellm.chat_completions.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.chat_completions.entrypoints import (
    NATIVE_ACOMPLETION,
    NATIVE_COMPLETION,
    NativeAcompletion,
    NativeCompletion,
)
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.public_call import NativeCall, native_call_hook
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]
PYTHON_RULES: Final = ()
RUST_RULES: Final = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)


def completion_binding(native: NativeCompletion | None) -> NativeBinding[NativeCompletion]:
    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    return binding


def acompletion_binding(native: NativeAcompletion | None) -> NativeBinding[NativeAcompletion]:
    binding: Final[NativeBinding[NativeAcompletion]] = NativeBinding("acompletion", validate=lambda _: None)
    binding.override(native)
    return binding


def test_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = ("gpt-4o", MESSAGES)
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = ModelResponse()

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records public call shape
        captured.append((call_args, call_kwargs))
        return response

    def native(request: NativeCall) -> ModelResponse:
        pytest.fail("Python-only dispatch must not call native")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=completion_binding(native),
            native=native_call_hook,
            rules=PYTHON_RULES,
        )
        is response
    )
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "metadata": metadata}


@pytest.mark.asyncio
async def test_async_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = ("gpt-4o", MESSAGES)
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = ModelResponse()

    async def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return response

    async def native(request: NativeCall) -> ModelResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=acompletion_binding(native),
        native=native_call_hook,
        rules=PYTHON_RULES,
    )
    assert result is response
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "metadata": metadata}


def test_native_receives_bound_request_and_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "base_url": "https://example.invalid",
        "extra_headers": {"x-test": "1"},
        "custom_llm_provider": "anthropic",
        "metadata": metadata,
    }
    captured: Final[list[tuple[NativeCall, tuple[object, ...], Mapping[str, object]]]] = []

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: rejected Rust fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(request: NativeCall) -> ModelResponse:
        captured.append((request, request.args, request.kwargs))
        return ModelResponse()

    args: Final[tuple[object, ...]] = ("anthropic/claude-sonnet-4-5", MESSAGES)
    _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=completion_binding(native),
        native=native_call_hook,
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert request.bound["model"] == "anthropic/claude-sonnet-4-5"
    assert request.bound["messages"] is MESSAGES
    assert request.bound["stream"] is True
    assert request.bound["api_key"] == "sk-test"
    assert request.bound["base_url"] == "https://example.invalid"
    assert request.bound["custom_llm_provider"] == "anthropic"
    assert request.bound["extra_headers"] == {"x-test": "1"}
    assert request.kwargs is kwargs
    assert call_args == args
    assert call_kwargs == kwargs
    assert call_kwargs["metadata"] is metadata


def test_internal_async_marker_bypasses_native() -> None:
    response: Final = ModelResponse()
    called: Final[list[bool]] = []

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records public call shape
        called.append(True)
        return response

    def native(request: NativeCall) -> ModelResponse:
        pytest.fail("acompletion's inner completion call must stay on Python")

    result: Final = _DISPATCH.run(
        ("gpt-4o", MESSAGES),
        {"acompletion": True},
        python=python,
        binding=completion_binding(native),
        native=native_call_hook,
        rules=RUST_RULES,
    )
    assert result is response
    assert called == [True]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        (("gpt-4o", MESSAGES), {"model": "duplicate"}),
        ((), {}),
    ),
)
def test_binding_errors_delegate_to_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = ModelResponse()

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records invalid call shape
        captured.append((call_args, call_kwargs))
        return response

    def native(request: NativeCall) -> ModelResponse:
        pytest.fail("Binding failures must be delegated to Python")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=completion_binding(native),
            native=native_call_hook,
            rules=RUST_RULES,
        )
        is response
    )
    assert captured == [(args, kwargs)]


def test_public_completion_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = ModelResponse()

    def native(request: NativeCall) -> ModelResponse:
        captured.append(request)
        return expected

    NATIVE_COMPLETION.override(native)
    monkeypatch.setattr(catalog, "RULES", RUST_RULES)
    public_completion: Final = cast(Callable[..., ModelResponse], litellm.completion)
    try:
        result: Final = public_completion(model="gpt-4o", messages=MESSAGES)
    finally:
        NATIVE_COMPLETION.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_public_acompletion_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = ModelResponse()

    async def native(request: NativeCall) -> ModelResponse:
        captured.append(request)
        return expected

    NATIVE_ACOMPLETION.override(native)
    monkeypatch.setattr(catalog, "RULES", RUST_RULES)
    public_acompletion: Final = cast(Callable[..., Awaitable[ModelResponse]], litellm.acompletion)
    try:
        result: Final = await public_acompletion(model="gpt-4o", messages=MESSAGES)
    finally:
        NATIVE_ACOMPLETION.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_public_completion_calls_keep_the_python_result() -> None:
    sync_response: Final = litellm.completion(model="openai/test-model", messages=MESSAGES, mock_response="ok")
    async_response: Final = await litellm.acompletion(model="openai/test-model", messages=MESSAGES, mock_response="ok")

    assert isinstance(sync_response, ModelResponse)
    assert isinstance(async_response, ModelResponse)
    assert sync_response.choices[0].message.content == "ok"
    assert async_response.choices[0].message.content == "ok"


def test_sync_completion_request_projects_public_arguments() -> None:
    rules: Final[Rules] = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)
    expected: Final = ModelResponse()

    def native(request: NativeCall) -> ModelResponse:
        assert request.bound["model"] == "test-model"
        assert request.bound["messages"] == MESSAGES
        assert request.bound["custom_llm_provider"] == "openai"
        assert request.bound["stream"] is True
        return expected

    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", MESSAGES),
        {"custom_llm_provider": "openai", "stream": True},
        python=lambda *args, **kwargs: pytest.fail("required native route must handle this call"),
        binding=binding,
        native=native_call_hook,
        rules=rules,
    )

    assert response is expected


@pytest.mark.asyncio
async def test_async_completion_falls_back_after_native_declines() -> None:
    from litellm.rust_bridge.bindings import native_exception_types

    native_types: Final = native_exception_types()
    if native_types is None:
        pytest.skip("native bridge is unavailable")
    declined, _ = native_types
    expected: Final = ModelResponse()
    rules: Final[Rules] = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_OPT_OUT),)

    async def native(request: NativeCall) -> ModelResponse:
        raise declined("unsupported")

    async def python(*args: object, **kwargs: object) -> ModelResponse:
        return expected

    binding: Final[NativeBinding[NativeAcompletion]] = NativeBinding("acompletion", validate=lambda _: None)
    binding.override(native)
    response: Final = await dispatch._ADISPATCH.arun(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", MESSAGES),
        {},
        python=python,
        binding=binding,
        native=native_call_hook,
        rules=rules,
    )

    assert response is expected


def test_internal_acompletion_marker_bypasses_native() -> None:
    rules: Final[Rules] = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)
    expected: Final = ModelResponse()

    def python(*args: object, **kwargs: object) -> ModelResponse:
        return expected

    def native(request: NativeCall) -> ModelResponse:
        pytest.fail("acompletion's inner completion call must stay on Python")

    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", MESSAGES),
        {"custom_llm_provider": "openai", "acompletion": True},
        python=python,
        binding=binding,
        native=native_call_hook,
        rules=rules,
    )

    assert response is expected


def test_positional_parameters_remain_available_to_native_projection() -> None:
    request: Final = _DISPATCH.request(("anthropic/test-model", MESSAGES, 12.0, 0.25), {})
    assert request is not None
    assert request.bound["timeout"] == 12.0
    assert request.bound["temperature"] == 0.25
    assert request.bound["messages"] is MESSAGES
