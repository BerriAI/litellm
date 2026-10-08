import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures

import pytest

import litellm
from litellm import main as python_chat
from litellm.chat_completions.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Decision, Route, RouteContext, Rust
from litellm.rust_bridge.chat_completions.entrypoints import NATIVE_ACOMPLETION, NATIVE_COMPLETION
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]


def required_everywhere(_context: RouteContext) -> Decision:
    return Rust(required=True)


def test_public_signature_is_the_legacy_signature() -> None:
    public_completion: Final = cast(Callable[..., object], litellm.completion)
    legacy_completion: Final = cast(Callable[..., object], python_chat.completion)
    public_acompletion: Final = cast(Callable[..., object], litellm.acompletion)
    legacy_acompletion: Final = cast(Callable[..., object], python_chat.acompletion)
    assert inspect.signature(public_completion) == inspect.signature(legacy_completion)
    assert inspect.signature(public_acompletion) == inspect.signature(legacy_acompletion)


@pytest.mark.parametrize("dispatch", (_DISPATCH, _ADISPATCH), ids=("completion", "acompletion"))
def test_request_binds_positional_and_keyword_arguments_onto_the_legacy_signature(dispatch: PublicDispatch) -> None:
    metadata: Final = {"user_id": "u"}
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "base_url": "https://example.invalid",
        "custom_llm_provider": "anthropic",
        "metadata": metadata,
    }

    request: Final = dispatch.request(("anthropic/claude-sonnet-4-5", MESSAGES), kwargs)

    assert request is not None
    assert request.bound["model"] == "anthropic/claude-sonnet-4-5"
    assert request.bound["messages"] is MESSAGES
    assert request.bound["stream"] is True
    assert request.bound["api_key"] == "sk-test"
    assert request.bound["base_url"] == "https://example.invalid"
    assert request.bound["metadata"] is metadata
    assert request.kwargs is kwargs
    assert dispatch.context(request) == RouteContext(
        Route.CHAT_COMPLETIONS, provider="anthropic", model="anthropic/claude-sonnet-4-5"
    )


def test_sync_request_binds_the_trailing_positional_parameters() -> None:
    request: Final = _DISPATCH.request(("gpt-4o", MESSAGES, 12.0, 0.25), {})

    assert request is not None
    assert request.bound["timeout"] == 12.0
    assert request.bound["temperature"] == 0.25


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        pytest.param((None, MESSAGES), {}, id="unnamed-model"),
        pytest.param(("gpt-4o", "hi"), {}, id="messages-not-a-sequence"),
        pytest.param(("gpt-4o", MESSAGES), {"model": "duplicate"}, id="does-not-bind"),
        pytest.param(("gpt-4o", MESSAGES), {"acompletion": True}, id="internal-async-hop"),
    ),
)
def test_request_stays_on_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    assert _DISPATCH.request(args, kwargs) is None


def test_public_completion_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = ModelResponse()

    def native(call: NativeCall) -> ModelResponse:
        captured.append(call)
        return expected

    NATIVE_COMPLETION.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
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

    async def native(call: NativeCall) -> ModelResponse:
        captured.append(call)
        return expected

    NATIVE_ACOMPLETION.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
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
