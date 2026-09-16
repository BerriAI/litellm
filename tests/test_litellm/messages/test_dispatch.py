import inspect
from collections.abc import Generator, Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages import handler as python_messages
from litellm.rust_bridge import configuration, runtime
from litellm.rust_bridge.catalog import Route, Rule, decision
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import (
    NATIVE_AMESSAGES,
    NATIVE_MESSAGES,
    LiteLLMMessagesRequest,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]
RUST_RULES: Final = (Rule(Route.MESSAGES, Rollout.RUST_OPT_OUT),)


def _response(model: str = "claude-sonnet-4-5") -> AnthropicMessagesResponse:
    return AnthropicMessagesResponse(id="msg_test", type="message", role="assistant", model=model, content=[])


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    NATIVE_MESSAGES.reset()
    NATIVE_AMESSAGES.reset()
    configuration.reset_rust_configuration()


@pytest.fixture
def rust_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "decision", lambda context, rules=RUST_RULES: decision(context, RUST_RULES))


def test_public_signature_is_the_legacy_signature() -> None:
    assert inspect.signature(litellm.anthropic_messages_handler) == inspect.signature(
        python_messages.anthropic_messages_handler
    )
    assert inspect.signature(litellm.anthropic_messages) == inspect.signature(python_messages.anthropic_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_python_only_route_never_loads_native(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    response: Final = _response()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(
        python_messages, "anthropic_messages" if asynchronous else "anthropic_messages_handler", fallback
    )
    monkeypatch.setattr(
        NATIVE_AMESSAGES if asynchronous else NATIVE_MESSAGES,
        "load",
        Mock(side_effect=AssertionError("native must not be loaded")),
    )
    litellm.rust(True)

    result: Final = (
        await litellm.anthropic_messages(16, MESSAGES, "claude-sonnet-4-5", temperature=0.1)
        if asynchronous
        else litellm.anthropic_messages_handler(16, MESSAGES, "claude-sonnet-4-5", temperature=0.1)
    )

    assert result is response
    fallback.assert_called_once_with(16, MESSAGES, "claude-sonnet-4-5", temperature=0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unavailable_native_uses_python(
    monkeypatch: pytest.MonkeyPatch, rust_route: None, asynchronous: bool
) -> None:
    response: Final = _response()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(
        python_messages, "anthropic_messages" if asynchronous else "anthropic_messages_handler", fallback
    )
    (NATIVE_AMESSAGES if asynchronous else NATIVE_MESSAGES).override(None)

    result: Final = (
        await litellm.anthropic_messages(16, MESSAGES, "claude-sonnet-4-5", temperature=0.1)
        if asynchronous
        else litellm.anthropic_messages_handler(16, MESSAGES, "claude-sonnet-4-5", temperature=0.1)
    )

    assert result is response
    fallback.assert_called_once_with(16, MESSAGES, "claude-sonnet-4-5", temperature=0.1)


def test_native_receives_the_bound_request_and_original_call_shape(rust_route: None) -> None:
    captured: Final[list[tuple[LiteLLMMessagesRequest, tuple[object, ...], Mapping[str, object]]]] = []

    def native(
        request: LiteLLMMessagesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> AnthropicMessagesResponse:
        captured.append((request, args, kwargs))
        return _response(request.model)

    NATIVE_MESSAGES.override(native)

    response: Final = litellm.anthropic_messages_handler(
        16,
        MESSAGES,
        "anthropic/claude-sonnet-4-5",
        stream=True,
        api_key="sk-test",
        api_base="https://example.invalid",
        custom_llm_provider="anthropic",
        litellm_metadata={"user_id": "u"},
    )

    request, call_args, hook_kwargs = captured[0]
    assert isinstance(response, dict)
    assert response["model"] == "anthropic/claude-sonnet-4-5"
    assert request.model == "anthropic/claude-sonnet-4-5"
    assert request.messages is MESSAGES
    assert request.max_tokens == 16
    assert request.stream is True
    assert request.api_key == "sk-test"
    assert request.api_base == "https://example.invalid"
    assert request.custom_llm_provider == "anthropic"
    assert request.kwargs == {"litellm_metadata": {"user_id": "u"}}
    assert call_args == (16, MESSAGES, "anthropic/claude-sonnet-4-5")
    assert hook_kwargs["litellm_metadata"] == {"user_id": "u"}
    assert "temperature" not in hook_kwargs


def test_internal_async_dispatch_marker_stays_on_python(monkeypatch: pytest.MonkeyPatch, rust_route: None) -> None:
    native: Final = Mock(side_effect=AssertionError("the async handler's inner sync call must stay on Python"))
    NATIVE_MESSAGES.override(native)
    response: Final = _response()
    fallback: Final = Mock(return_value=response)
    monkeypatch.setattr(python_messages, "anthropic_messages_handler", fallback)

    assert litellm.anthropic_messages_handler(16, MESSAGES, "claude-sonnet-4-5", is_async=True) is response
    native.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-disabled", "flag-enabled"])
def test_public_binding_errors_do_not_depend_on_native_selection(rust_route: None, enabled: bool) -> None:
    native: Final = Mock(side_effect=AssertionError("binding errors precede admission"))
    litellm.rust(enabled)
    NATIVE_MESSAGES.override(native)

    with pytest.raises(TypeError, match=r"anthropic_messages_handler\(\) got multiple values for argument 'model'"):
        litellm.anthropic_messages_handler(16, MESSAGES, "claude-sonnet-4-5", model="duplicate")
    with pytest.raises(TypeError, match=r"anthropic_messages_handler\(\) missing 3 required positional arguments"):
        litellm.anthropic_messages_handler()
    native.assert_not_called()


class Declined(Exception):
    pass


class Upstream(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("declined", [False, True])
async def test_only_native_declines_replay_on_python(
    monkeypatch: pytest.MonkeyPatch, rust_route: None, asynchronous: bool, declined: bool
) -> None:
    failure: Final = Declined("unsupported") if declined else RuntimeError("provider already called")
    native: Final = AsyncMock(side_effect=failure) if asynchronous else Mock(side_effect=failure)
    (NATIVE_AMESSAGES if asynchronous else NATIVE_MESSAGES).override(native)
    monkeypatch.setattr(runtime, "native_exception_types", lambda: (Declined, Upstream))
    response: Final = _response()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(
        python_messages, "anthropic_messages" if asynchronous else "anthropic_messages_handler", fallback
    )

    async def call() -> object:
        if asynchronous:
            return await litellm.anthropic_messages(16, MESSAGES, "claude-sonnet-4-5")
        return litellm.anthropic_messages_handler(16, MESSAGES, "claude-sonnet-4-5")

    if declined:
        assert await call() is response
        fallback.assert_called_once_with(16, MESSAGES, "claude-sonnet-4-5")
    else:
        with pytest.raises(RuntimeError) as caught:
            await call()
        assert caught.value is failure
        fallback.assert_not_called()
    assert native.call_count == 1
