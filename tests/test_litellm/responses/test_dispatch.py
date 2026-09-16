import inspect
from collections.abc import Generator, Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm.responses import main as python_responses
from litellm.rust_bridge import configuration, runtime
from litellm.rust_bridge.catalog import Route, Rule, decision
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.responses.entrypoints import (
    NATIVE_ARESPONSES,
    NATIVE_RESPONSES,
    LiteLLMResponsesRequest,
)
from litellm.types.llms.openai import ResponsesAPIResponse

RUST_RULES: Final = (Rule(Route.RESPONSES, Rollout.RUST_OPT_OUT),)


def _response(model: str = "gpt-4o") -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_test", object="response", created_at=0, model=model, output=[], status="completed"
    )


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    NATIVE_RESPONSES.reset()
    NATIVE_ARESPONSES.reset()
    configuration.reset_rust_configuration()


@pytest.fixture
def rust_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "decision", lambda context, rules=RUST_RULES: decision(context, RUST_RULES))


def test_public_signature_is_the_legacy_signature() -> None:
    assert inspect.signature(litellm.responses) == inspect.signature(python_responses.responses)
    assert inspect.signature(litellm.aresponses) == inspect.signature(python_responses.aresponses)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_python_only_route_never_loads_native(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    response: Final = _response()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(python_responses, "aresponses" if asynchronous else "responses", fallback)
    monkeypatch.setattr(
        NATIVE_ARESPONSES if asynchronous else NATIVE_RESPONSES,
        "load",
        Mock(side_effect=AssertionError("native must not be loaded")),
    )
    litellm.rust(True)

    result: Final = (
        await litellm.aresponses("hi", "gpt-4o", temperature=0.1)
        if asynchronous
        else litellm.responses("hi", "gpt-4o", temperature=0.1)
    )

    assert result is response
    fallback.assert_called_once_with("hi", "gpt-4o", temperature=0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unavailable_native_uses_python(
    monkeypatch: pytest.MonkeyPatch, rust_route: None, asynchronous: bool
) -> None:
    response: Final = _response()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(python_responses, "aresponses" if asynchronous else "responses", fallback)
    (NATIVE_ARESPONSES if asynchronous else NATIVE_RESPONSES).override(None)

    result: Final = (
        await litellm.aresponses("hi", "gpt-4o", temperature=0.1)
        if asynchronous
        else litellm.responses("hi", "gpt-4o", temperature=0.1)
    )

    assert result is response
    fallback.assert_called_once_with("hi", "gpt-4o", temperature=0.1)


def test_native_receives_the_bound_request_and_original_call_shape(rust_route: None) -> None:
    captured: Final[list[tuple[LiteLLMResponsesRequest, tuple[object, ...], Mapping[str, object]]]] = []

    def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        captured.append((request, args, kwargs))
        return _response(request.model)

    NATIVE_RESPONSES.override(native)

    response: Final = litellm.responses(
        "hi",
        "anthropic/claude-sonnet-4-5",
        stream=True,
        api_key="sk-test",
        api_base="https://example.invalid",
        extra_headers={"x-test": "1"},
        custom_llm_provider="anthropic",
        litellm_metadata={"user_id": "u"},
    )

    request, call_args, hook_kwargs = captured[0]
    assert isinstance(response, ResponsesAPIResponse)
    assert response.model == "anthropic/claude-sonnet-4-5"
    assert request.model == "anthropic/claude-sonnet-4-5"
    assert request.input == "hi"
    assert request.stream is True
    assert request.api_key == "sk-test"
    assert request.api_base == "https://example.invalid"
    assert request.custom_llm_provider == "anthropic"
    assert request.extra_headers == {"x-test": "1"}
    assert request.kwargs == {
        "api_key": "sk-test",
        "api_base": "https://example.invalid",
        "litellm_metadata": {"user_id": "u"},
    }
    assert call_args == ("hi", "anthropic/claude-sonnet-4-5")
    assert hook_kwargs["litellm_metadata"] == {"user_id": "u"}
    assert "temperature" not in hook_kwargs


def test_internal_async_dispatch_marker_stays_on_python(monkeypatch: pytest.MonkeyPatch, rust_route: None) -> None:
    native: Final = Mock(side_effect=AssertionError("aresponses's inner responses() call must stay on Python"))
    NATIVE_RESPONSES.override(native)
    response: Final = _response()
    fallback: Final = Mock(return_value=response)
    monkeypatch.setattr(python_responses, "responses", fallback)

    assert litellm.responses("hi", "gpt-4o", aresponses=True) is response
    native.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-disabled", "flag-enabled"])
def test_public_binding_errors_do_not_depend_on_native_selection(rust_route: None, enabled: bool) -> None:
    native: Final = Mock(side_effect=AssertionError("binding errors precede admission"))
    litellm.rust(enabled)
    NATIVE_RESPONSES.override(native)

    with pytest.raises(TypeError, match=r"responses\(\) got multiple values for argument 'model'"):
        litellm.responses("hi", "gpt-4o", model="duplicate")
    with pytest.raises(TypeError, match=r"responses\(\) missing 2 required positional arguments: 'input' and 'model'"):
        litellm.responses()
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
    (NATIVE_ARESPONSES if asynchronous else NATIVE_RESPONSES).override(native)
    monkeypatch.setattr(runtime, "native_exception_types", lambda: (Declined, Upstream))
    response: Final = _response()
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(python_responses, "aresponses" if asynchronous else "responses", fallback)

    async def call() -> object:
        if asynchronous:
            return await litellm.aresponses("hi", "gpt-4o")
        return litellm.responses("hi", "gpt-4o")

    if declined:
        assert await call() is response
        fallback.assert_called_once_with("hi", "gpt-4o")
    else:
        with pytest.raises(RuntimeError) as caught:
            await call()
        assert caught.value is failure
        fallback.assert_not_called()
    assert native.call_count == 1
