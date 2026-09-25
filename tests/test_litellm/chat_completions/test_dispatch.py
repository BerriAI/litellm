from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.chat_completions import dispatch
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.chat_completions.entrypoints import (
    LiteLLMChatCompletionsRequest,
    NativeAcompletion,
    NativeCompletion,
)
from litellm.rust_bridge.configuration import Rollout
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]


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

    def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
        assert request.model == "test-model"
        assert request.messages == MESSAGES
        assert request.custom_llm_provider == "openai"
        assert request.stream is True
        return expected

    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", MESSAGES),
        {"custom_llm_provider": "openai", "stream": True},
        python=lambda *args, **kwargs: pytest.fail("required native route must handle this call"),
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
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

    async def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
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
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        rules=rules,
    )

    assert response is expected


def test_internal_acompletion_marker_bypasses_native() -> None:
    rules: Final[Rules] = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)
    expected: Final = ModelResponse()

    def python(*args: object, **kwargs: object) -> ModelResponse:
        return expected

    def native(
        request: LiteLLMChatCompletionsRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> ModelResponse:
        pytest.fail("acompletion's inner completion call must stay on Python")

    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", MESSAGES),
        {"custom_llm_provider": "openai", "acompletion": True},
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        rules=rules,
    )

    assert response is expected
