from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.messages import dispatch
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import (
    LiteLLMMessagesRequest,
    NativeAmessages,
    NativeMessages,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_public_anthropic_messages_keeps_the_python_result() -> None:
    response: Final = await litellm.anthropic_messages(
        model="anthropic/claude-sonnet-4-5", messages=MESSAGES, max_tokens=10, mock_response="ok"
    )

    assert isinstance(response, dict)
    content: Final = TypeAdapter(list[dict[str, object]]).validate_python(response.get("content", []))
    assert content[0]["text"] == "ok"


def test_sync_messages_request_projects_public_arguments() -> None:
    rules: Final[Rules] = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)
    expected: Final = AnthropicMessagesResponse(model="claude-test")

    def native(
        request: LiteLLMMessagesRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> AnthropicMessagesResponse:
        assert request.model == "claude-test"
        assert request.messages == MESSAGES
        assert request.max_tokens == 10
        assert request.custom_llm_provider == "anthropic"
        return expected

    binding: Final[NativeBinding[NativeMessages]] = NativeBinding("messages", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {
            "model": "claude-test",
            "messages": MESSAGES,
            "max_tokens": 10,
            "custom_llm_provider": "anthropic",
        },
        python=lambda *args, **kwargs: pytest.fail("required native route must handle this call"),
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        rules=rules,
    )

    assert response is expected


def test_messages_binding_error_delegates_unchanged_to_python() -> None:
    rules: Final[Rules] = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)
    expected: Final = AnthropicMessagesResponse(model="claude-test")

    def python(*args: object, **kwargs: object) -> AnthropicMessagesResponse:
        return expected

    def native(
        request: LiteLLMMessagesRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> AnthropicMessagesResponse:
        pytest.fail("a call without max_tokens cannot project a request and must stay on Python")

    binding: Final[NativeBinding[NativeMessages]] = NativeBinding("messages", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {"model": "claude-test", "messages": MESSAGES, "custom_llm_provider": "anthropic"},
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        rules=rules,
    )

    assert response is expected


@pytest.mark.asyncio
async def test_async_messages_falls_back_after_native_declines() -> None:
    from litellm.rust_bridge.bindings import native_exception_types

    native_types: Final = native_exception_types()
    if native_types is None:
        pytest.skip("native bridge is unavailable")
    declined, _ = native_types
    expected: Final = AnthropicMessagesResponse(model="claude-test")
    rules: Final[Rules] = (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_OUT),)

    async def native(
        request: LiteLLMMessagesRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> AnthropicMessagesResponse:
        raise declined("unsupported")

    async def python(*args: object, **kwargs: object) -> AnthropicMessagesResponse:
        return expected

    binding: Final[NativeBinding[NativeAmessages]] = NativeBinding("amessages", validate=lambda _: None)
    binding.override(native)
    response: Final = await dispatch._ADISPATCH.arun(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {"model": "claude-test", "messages": MESSAGES, "max_tokens": 10},
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        rules=rules,
    )

    assert response is expected


def test_internal_is_async_marker_bypasses_native() -> None:
    rules: Final[Rules] = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)
    expected: Final = AnthropicMessagesResponse(model="claude-test")

    def python(*args: object, **kwargs: object) -> AnthropicMessagesResponse:
        return expected

    def native(
        request: LiteLLMMessagesRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> AnthropicMessagesResponse:
        pytest.fail("anthropic_messages' inner handler call must stay on Python")

    binding: Final[NativeBinding[NativeMessages]] = NativeBinding("messages", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {
            "model": "claude-test",
            "messages": MESSAGES,
            "max_tokens": 10,
            "custom_llm_provider": "anthropic",
            "is_async": True,
        },
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        rules=rules,
    )

    assert response is expected
