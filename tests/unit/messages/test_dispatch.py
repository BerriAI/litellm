import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.anthropic.pass_through.messages import handler as python_messages
from litellm.messages import dispatch
from litellm.messages.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Decision, Route, RouteContext, Rust
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]


def required_everywhere(_context: RouteContext) -> Decision:
    return Rust(required=True)


def response(model: str = "claude-sonnet-4-5") -> AnthropicMessagesResponse:
    return AnthropicMessagesResponse(id="msg_test", type="message", role="assistant", model=model, content=[])


def test_public_signature_is_the_legacy_signature() -> None:
    public_messages: Final = cast(Callable[..., object], litellm.anthropic_messages_handler)
    legacy_messages: Final = cast(Callable[..., object], python_messages.anthropic_messages_handler)
    public_amessages: Final = cast(Callable[..., object], litellm.anthropic_messages)
    legacy_amessages: Final = cast(Callable[..., object], python_messages.anthropic_messages)
    assert inspect.signature(public_messages) == inspect.signature(legacy_messages)
    assert inspect.signature(public_amessages) == inspect.signature(legacy_amessages)


@pytest.mark.parametrize("dispatch", (_DISPATCH, _ADISPATCH), ids=("handler", "anthropic_messages"))
def test_request_binds_positional_and_keyword_arguments_onto_the_legacy_signature(dispatch: PublicDispatch) -> None:
    metadata: Final = {"user_id": "u"}
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "api_base": "https://example.invalid",
        "custom_llm_provider": "anthropic",
        "litellm_metadata": metadata,
    }

    request: Final = dispatch.request((16, MESSAGES, "anthropic/claude-sonnet-4-5"), kwargs)

    assert request is not None
    assert request.bound["max_tokens"] == 16
    assert request.bound["messages"] is MESSAGES
    assert request.bound["model"] == "anthropic/claude-sonnet-4-5"
    assert request.bound["stream"] is True
    assert request.bound["api_key"] == "sk-test"
    assert request.bound["api_base"] == "https://example.invalid"
    assert request.bound["litellm_metadata"] is metadata
    assert request.kwargs is kwargs


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        pytest.param((16, MESSAGES, None), {}, id="unnamed-model"),
        pytest.param((16, "hi", "claude-sonnet-4-5"), {}, id="messages-not-a-sequence"),
        pytest.param((None, MESSAGES, "claude-sonnet-4-5"), {}, id="max-tokens-not-an-int"),
        pytest.param((16, MESSAGES, "claude-sonnet-4-5"), {"model": "duplicate"}, id="does-not-bind"),
        pytest.param((16, MESSAGES, "claude-sonnet-4-5"), {"is_async": True}, id="internal-async-hop"),
    ),
)
def test_request_stays_on_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    assert _DISPATCH.request(args, kwargs) is None


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "provider"),
    (
        pytest.param("anthropic/claude-sonnet-4-5", None, "anthropic", id="model-prefix"),
        pytest.param("claude-sonnet-4-5", None, "anthropic", id="known-model"),
        pytest.param("claude-sonnet-4-5", "bedrock", "bedrock", id="declared-provider"),
        pytest.param("unknown-model", None, None, id="unresolvable"),
        pytest.param("unknown-model", "custom", "custom", id="unresolvable-keeps-declared"),
    ),
)
def test_context_resolves_the_provider_like_the_python_route(
    model: str, custom_llm_provider: str | None, provider: str | None
) -> None:
    request: Final = _DISPATCH.request(
        (16, MESSAGES, model), {} if custom_llm_provider is None else {"custom_llm_provider": custom_llm_provider}
    )
    assert request is not None

    assert _DISPATCH.context(request) == RouteContext(Route.MESSAGES, provider=provider, model=model)


@pytest.mark.parametrize(
    ("model", "custom_llm_provider"),
    (("github_copilot/gpt-5.5", None), ("gpt-5.5", "github_copilot"), ("chatgpt/gpt-5.5", None)),
)
def test_context_takes_an_authenticating_provider_from_the_declaration(
    monkeypatch: pytest.MonkeyPatch, model: str, custom_llm_provider: str | None
) -> None:
    def resolver_runs_the_oauth_flow(*args: object, **kwargs: object) -> object:  # kwargs-ok: resolver call shape
        pytest.fail("selecting an implementation must not run get_llm_provider for an authenticating provider")

    monkeypatch.setattr(dispatch, "get_llm_provider", resolver_runs_the_oauth_flow)
    request: Final = _DISPATCH.request(
        (), {"model": model, "messages": MESSAGES, "max_tokens": 10, "custom_llm_provider": custom_llm_provider}
    )
    assert request is not None

    assert _DISPATCH.context(request) == RouteContext(
        Route.MESSAGES, provider=custom_llm_provider or model.split("/")[0], model=model
    )


def test_anthropic_create_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    def native(request: NativeCall) -> AnthropicMessagesResponse:
        captured.append(request)
        return expected

    NATIVE_MESSAGES.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_create: Final = cast(Callable[..., AnthropicMessagesResponse], litellm.anthropic.create)
    try:
        result: Final = public_create(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5")
    finally:
        NATIVE_MESSAGES.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_anthropic_acreate_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        captured.append(request)
        return expected

    NATIVE_AMESSAGES.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_acreate: Final = cast(Callable[..., Awaitable[AnthropicMessagesResponse]], litellm.anthropic.acreate)
    try:
        result: Final = await public_acreate(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5")
    finally:
        NATIVE_AMESSAGES.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_public_anthropic_messages_keeps_the_python_result() -> None:
    result: Final = await litellm.anthropic_messages(
        model="anthropic/claude-sonnet-4-5", messages=MESSAGES, max_tokens=10, mock_response="ok"
    )

    assert isinstance(result, dict)
    content: Final = TypeAdapter(list[dict[str, object]]).validate_python(result.get("content", []))
    assert content[0]["text"] == "ok"
