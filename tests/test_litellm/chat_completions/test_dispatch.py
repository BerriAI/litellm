import inspect
from collections.abc import Generator, Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm import main as python_chat
from litellm.rust_bridge import configuration, runtime
from litellm.rust_bridge.catalog import Route, Rule, decision
from litellm.rust_bridge.chat_completions.entrypoints import (
    NATIVE_ACOMPLETION,
    NATIVE_COMPLETION,
    LiteLLMChatCompletionsRequest,
)
from litellm.rust_bridge.configuration import Rollout
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]
RUST_RULES: Final = (Rule(Route.CHAT_COMPLETIONS, Rollout.RUST_OPT_OUT),)


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    NATIVE_COMPLETION.reset()
    NATIVE_ACOMPLETION.reset()
    configuration.reset_rust_configuration()


@pytest.fixture
def rust_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "decision", lambda context, rules=RUST_RULES: decision(context, RUST_RULES))


def test_public_signature_is_the_legacy_signature() -> None:
    assert inspect.signature(litellm.completion) == inspect.signature(python_chat.completion)
    assert inspect.signature(litellm.acompletion) == inspect.signature(python_chat.acompletion)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_python_only_route_never_loads_native(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    response: Final = ModelResponse()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(python_chat, "acompletion" if asynchronous else "completion", fallback)
    monkeypatch.setattr(
        NATIVE_ACOMPLETION if asynchronous else NATIVE_COMPLETION,
        "load",
        Mock(side_effect=AssertionError("native must not be loaded")),
    )
    litellm.rust(True)

    result: Final = (
        await litellm.acompletion("gpt-4o", MESSAGES, temperature=0.1)
        if asynchronous
        else litellm.completion("gpt-4o", MESSAGES, temperature=0.1)
    )

    assert result is response
    fallback.assert_called_once_with("gpt-4o", MESSAGES, temperature=0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unavailable_native_uses_python(
    monkeypatch: pytest.MonkeyPatch, rust_route: None, asynchronous: bool
) -> None:
    response: Final = ModelResponse()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(python_chat, "acompletion" if asynchronous else "completion", fallback)
    (NATIVE_ACOMPLETION if asynchronous else NATIVE_COMPLETION).override(None)

    result: Final = (
        await litellm.acompletion("gpt-4o", MESSAGES, temperature=0.1)
        if asynchronous
        else litellm.completion("gpt-4o", MESSAGES, temperature=0.1)
    )

    assert result is response
    fallback.assert_called_once_with("gpt-4o", MESSAGES, temperature=0.1)


def test_native_receives_the_bound_request_and_original_call_shape(rust_route: None) -> None:
    captured: Final[list[tuple[LiteLLMChatCompletionsRequest, tuple[object, ...], Mapping[str, object]]]] = []

    def native(
        request: LiteLLMChatCompletionsRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ModelResponse:
        captured.append((request, args, kwargs))
        return ModelResponse(model=request.model)

    NATIVE_COMPLETION.override(native)

    response: Final = litellm.completion(
        "anthropic/claude-sonnet-4-5",
        MESSAGES,
        stream=True,
        api_key="sk-test",
        base_url="https://example.invalid",
        extra_headers={"x-test": "1"},
        custom_llm_provider="anthropic",
        metadata={"user_id": "u"},
    )

    request, call_args, hook_kwargs = captured[0]
    assert isinstance(response, ModelResponse)
    assert response.model == "anthropic/claude-sonnet-4-5"
    assert request.model == "anthropic/claude-sonnet-4-5"
    assert request.messages is MESSAGES
    assert request.stream is True
    assert request.api_key == "sk-test"
    assert request.api_base == "https://example.invalid"
    assert request.custom_llm_provider == "anthropic"
    assert request.extra_headers == {"x-test": "1"}
    assert request.kwargs == {"custom_llm_provider": "anthropic", "metadata": {"user_id": "u"}}
    assert call_args == ("anthropic/claude-sonnet-4-5", MESSAGES)
    assert hook_kwargs["metadata"] == {"user_id": "u"}
    assert "temperature" not in hook_kwargs


def test_internal_async_dispatch_marker_stays_on_python(monkeypatch: pytest.MonkeyPatch, rust_route: None) -> None:
    native: Final = Mock(side_effect=AssertionError("acompletion's inner completion() call must stay on Python"))
    NATIVE_COMPLETION.override(native)
    response: Final = ModelResponse()
    fallback: Final = Mock(return_value=response)
    monkeypatch.setattr(python_chat, "completion", fallback)

    assert litellm.completion("gpt-4o", MESSAGES, acompletion=True) is response
    native.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-disabled", "flag-enabled"])
def test_public_binding_errors_do_not_depend_on_native_selection(rust_route: None, enabled: bool) -> None:
    native: Final = Mock(side_effect=AssertionError("binding errors precede admission"))
    litellm.rust(enabled)
    NATIVE_COMPLETION.override(native)

    with pytest.raises(TypeError, match=r"completion\(\) got multiple values for argument 'model'"):
        litellm.completion("gpt-4o", MESSAGES, model="duplicate")
    with pytest.raises(TypeError, match=r"completion\(\) missing 1 required positional argument: 'model'"):
        litellm.completion()
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
    (NATIVE_ACOMPLETION if asynchronous else NATIVE_COMPLETION).override(native)
    monkeypatch.setattr(runtime, "native_exception_types", lambda: (Declined, Upstream))
    response: Final = ModelResponse()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(python_chat, "acompletion" if asynchronous else "completion", fallback)

    async def call() -> object:
        if asynchronous:
            return await litellm.acompletion("gpt-4o", MESSAGES)
        return litellm.completion("gpt-4o", MESSAGES)

    if declined:
        assert await call() is response
        fallback.assert_called_once_with("gpt-4o", MESSAGES)
    else:
        with pytest.raises(RuntimeError) as caught:
            await call()
        assert caught.value is failure
        fallback.assert_not_called()
    assert native.call_count == 1
