import inspect
from collections.abc import Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages import handler as python_messages
from litellm.messages.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, Rule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import (
    LiteLLMMessagesRequest,
    NativeAmessages,
    NativeMessages,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]
PYTHON_RULES: Final[Rules] = ()
RUST_RULES: Final[Rules] = (Rule(Route.MESSAGES, Rollout.RUST_REQUIRED),)


def messages_binding(native: NativeMessages | None) -> NativeBinding[NativeMessages]:
    binding: Final[NativeBinding[NativeMessages]] = NativeBinding(
        "anthropic_messages_handler", validate=lambda _: None
    )
    binding.override(native)
    return binding


def amessages_binding(native: NativeAmessages | None) -> NativeBinding[NativeAmessages]:
    binding: Final[NativeBinding[NativeAmessages]] = NativeBinding("anthropic_messages", validate=lambda _: None)
    binding.override(native)
    return binding


def response(model: str = "claude-sonnet-4-5") -> AnthropicMessagesResponse:
    return AnthropicMessagesResponse(id="msg_test", type="message", role="assistant", model=model, content=[])


def test_public_signature_is_the_legacy_signature() -> None:
    public_messages: Final = cast(Callable[..., object], litellm.anthropic_messages_handler)
    legacy_messages: Final = cast(Callable[..., object], python_messages.anthropic_messages_handler)
    public_amessages: Final = cast(Callable[..., object], litellm.anthropic_messages)
    legacy_amessages: Final = cast(Callable[..., object], python_messages.anthropic_messages)
    assert inspect.signature(public_messages) == inspect.signature(legacy_messages)
    assert inspect.signature(public_amessages) == inspect.signature(legacy_amessages)


def test_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (16, MESSAGES, "claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "litellm_metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: LiteLLMMessagesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> AnthropicMessagesResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=PYTHON_RULES,
    )
    assert result is expected
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "litellm_metadata": metadata}


@pytest.mark.asyncio
async def test_async_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (16, MESSAGES, "claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "litellm_metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    async def python(
        *call_args: object, **call_kwargs: object  # kwargs-ok: records call shape
    ) -> AnthropicMessagesResponse:
        captured.append((call_args, call_kwargs))
        return expected

    async def native(
        request: LiteLLMMessagesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> AnthropicMessagesResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=PYTHON_RULES,
    )
    assert result is expected
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "litellm_metadata": metadata}


def test_native_receives_normalized_request_and_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (16, MESSAGES, "anthropic/claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "api_base": "https://example.invalid",
        "custom_llm_provider": "anthropic",
        "litellm_metadata": metadata,
    }
    captured: Final[list[tuple[LiteLLMMessagesRequest, tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response("anthropic/claude-sonnet-4-5")

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(
        request: LiteLLMMessagesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> AnthropicMessagesResponse:
        captured.append((request, args, kwargs))
        return expected

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )
    assert result is expected
    request, call_args, call_kwargs = captured[0]
    assert request.model == "anthropic/claude-sonnet-4-5"
    assert request.messages is MESSAGES
    assert request.max_tokens == 16
    assert request.stream is True
    assert request.api_key == "sk-test"
    assert request.api_base == "https://example.invalid"
    assert request.custom_llm_provider == "anthropic"
    assert request.kwargs == {"litellm_metadata": metadata}
    assert request.kwargs["litellm_metadata"] is metadata
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata


def test_internal_async_marker_bypasses_native() -> None:
    args: Final[tuple[object, ...]] = (16, MESSAGES, "claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {"is_async": True}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: LiteLLMMessagesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> AnthropicMessagesResponse:
        pytest.fail("The async handler's inner sync call must stay on Python")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )
    assert result is expected
    assert captured == [(args, kwargs)]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        ((16, MESSAGES, "claude-sonnet-4-5"), {"model": "duplicate"}),
        ((), {}),
    ),
)
def test_binding_errors_delegate_to_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: records invalid call
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: LiteLLMMessagesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> AnthropicMessagesResponse:
        pytest.fail("Binding failures must be delegated to Python")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )
    assert result is expected
    assert captured == [(args, kwargs)]
