import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest

import litellm
from litellm import main as python_chat
from litellm.chat_completions.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, Rule
from litellm.rust_bridge.chat_completions.entrypoints import (
    NATIVE_ACOMPLETION,
    NATIVE_COMPLETION,
    LiteLLMChatCompletionsRequest,
    NativeAcompletion,
    NativeCompletion,
)
from litellm.rust_bridge.configuration import Rollout
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]
PYTHON_RULES: Final = ()
RUST_RULES: Final = (Rule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)


def completion_binding(native: NativeCompletion | None) -> NativeBinding[NativeCompletion]:
    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    return binding


def acompletion_binding(native: NativeAcompletion | None) -> NativeBinding[NativeAcompletion]:
    binding: Final[NativeBinding[NativeAcompletion]] = NativeBinding("acompletion", validate=lambda _: None)
    binding.override(native)
    return binding


def test_public_signature_is_the_legacy_signature() -> None:
    public_completion: Final = cast(Callable[..., object], litellm.completion)
    legacy_completion: Final = cast(Callable[..., object], python_chat.completion)
    public_acompletion: Final = cast(Callable[..., object], litellm.acompletion)
    legacy_acompletion: Final = cast(Callable[..., object], python_chat.acompletion)
    assert inspect.signature(public_completion) == inspect.signature(legacy_completion)
    assert inspect.signature(public_acompletion) == inspect.signature(legacy_acompletion)


def test_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = ("gpt-4o", MESSAGES)
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = ModelResponse()

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records public call shape
        captured.append((call_args, call_kwargs))
        return response

    def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
        pytest.fail("Python-only dispatch must not call native")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=completion_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
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

    async def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=acompletion_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
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
    captured: Final[
        list[tuple[LiteLLMChatCompletionsRequest, tuple[object, ...], Mapping[str, object]]]
    ] = []

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: rejected Rust fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
        captured.append((request, args, kwargs))
        return ModelResponse()

    args: Final[tuple[object, ...]] = ("anthropic/claude-sonnet-4-5", MESSAGES)
    _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=completion_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert request.model == "anthropic/claude-sonnet-4-5"
    assert request.messages is MESSAGES
    assert request.stream is True
    assert request.api_key == "sk-test"
    assert request.api_base == "https://example.invalid"
    assert request.custom_llm_provider == "anthropic"
    assert request.extra_headers == {"x-test": "1"}
    assert request.kwargs == {"custom_llm_provider": "anthropic", "metadata": metadata}
    assert call_args == args
    assert call_kwargs == kwargs
    assert call_kwargs["metadata"] is metadata


def test_internal_async_marker_bypasses_native() -> None:
    response: Final = ModelResponse()
    called: Final[list[bool]] = []

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records public call shape
        called.append(True)
        return response

    def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
        pytest.fail("acompletion's inner completion call must stay on Python")

    result: Final = _DISPATCH.run(
        ("gpt-4o", MESSAGES),
        {"acompletion": True},
        python=python,
        binding=completion_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
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

    def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
        pytest.fail("Binding failures must be delegated to Python")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=completion_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
            rules=RUST_RULES,
        )
        is response
    )
    assert captured == [(args, kwargs)]


def test_public_completion_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[LiteLLMChatCompletionsRequest]] = []
    expected: Final = ModelResponse()

    def native(
        request: LiteLLMChatCompletionsRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ModelResponse:
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
    assert [request.model for request in captured] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_public_acompletion_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[LiteLLMChatCompletionsRequest]] = []
    expected: Final = ModelResponse()

    async def native(
        request: LiteLLMChatCompletionsRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ModelResponse:
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
    assert [request.model for request in captured] == ["gpt-4o"]
