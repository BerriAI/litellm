from __future__ import annotations

from typing import Final, Literal
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import Router
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.router_utils.fallback_event_handlers import get_fallback_model_group


@pytest.mark.asyncio
async def test_async_fallbacks_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that router.acompletion with stream=True and mock_response works correctly."""
    monkeypatch.setattr(litellm, "set_verbose", False)
    router: Final = Router(
        model_list=[
            {
                "model_name": "azure/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "fake-key",
                    "api_version": "2024-01-01",
                    "api_base": "https://fake.openai.azure.com",
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_key": "fake-key",
                },
                "tpm": 1000000,
                "rpm": 9000,
            },
        ],
        fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-4o-mini"]}],
        set_verbose=False,
    )
    response: Final = await router.acompletion(
        model="azure/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=True,
        mock_response="This is a mock streaming response",
    )
    chunks: Final = tuple([chunk async for chunk in response])
    assert chunks


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("litellm_module_fallbacks", [True, False])
@pytest.mark.asyncio
async def test_default_model_fallbacks(
    sync_mode: bool,
    litellm_module_fallbacks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Related issue - https://github.com/BerriAI/litellm/issues/3623

    If model misconfigured, setup a default model for generic fallback
    """
    if litellm_module_fallbacks:
        monkeypatch.setattr(litellm, "default_fallbacks", ["my-good-model"])
    router: Final = Router(
        model_list=[
            {
                "model_name": "bad-model",
                "litellm_params": {
                    "model": "openai/my-bad-model",
                    "api_key": "my-bad-api-key",
                },
            },
            {
                "model_name": "my-good-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "test-key",
                },
            },
        ],
        default_fallbacks=(["my-good-model"] if litellm_module_fallbacks is False else None),
    )

    response: Final = (
        router.completion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
        if sync_mode
        else await router.acompletion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
    )

    assert isinstance(response, litellm.ModelResponse)
    assert response.model is not None and response.model == "gpt-4o"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_client_side_fallbacks_list(sync_mode: bool) -> None:
    """

    Tests Client Side Fallbacks

    User can pass "fallbacks": ["gpt-3.5-turbo"] and this should work

    """
    router: Final = Router(
        model_list=[
            {
                "model_name": "bad-model",
                "litellm_params": {
                    "model": "openai/my-bad-model",
                    "api_key": "my-bad-api-key",
                },
            },
            {
                "model_name": "my-good-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "test-key",
                },
            },
        ],
    )

    response: Final = (
        router.completion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            fallbacks=["my-good-model"],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
        if sync_mode
        else await router.acompletion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            fallbacks=["my-good-model"],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
    )

    assert isinstance(response, litellm.ModelResponse)
    assert response.model is not None and response.model == "gpt-4o"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("content_filter_response_exception", [True, False])
@pytest.mark.parametrize("fallback_type", ["model-specific", "default"])
@pytest.mark.asyncio
async def test_router_content_policy_fallbacks(
    sync_mode: bool,
    content_filter_response_exception: bool,
    fallback_type: Literal["model-specific", "default"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")

    if content_filter_response_exception:
        mock_response: Final = Exception("content filtering policy")
    else:
        mock_response: Final = litellm.ModelResponse(
            choices=[litellm.Choices(finish_reason="content_filter")],
            model="gpt-3.5-turbo",
            usage=litellm.Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": mock_response,
                },
            },
            {
                "model_name": "my-fallback-model",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_key": "",
                    "mock_response": "This works!",
                },
            },
            {
                "model_name": "my-default-fallback-model",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_key": "",
                    "mock_response": "This works 2!",
                },
            },
            {
                "model_name": "my-general-model",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": Exception("Should not have called this."),
                },
            },
            {
                "model_name": "my-context-window-model",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": Exception("Should not have called this."),
                },
            },
        ],
        content_policy_fallbacks=(
            [{"claude-sonnet-4-5-20250929": ["my-fallback-model"]}] if fallback_type == "model-specific" else None
        ),
        default_fallbacks=(["my-default-fallback-model"] if fallback_type == "default" else None),
    )

    response: Final = (
        router.completion(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
        if sync_mode
        else await router.acompletion(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    )

    assert response.model == "my-fake-model"


def mock_post_streaming(url: str, **kwargs: object) -> MagicMock:
    response: Final = MagicMock()
    response.status_code = 529
    response.headers = {"Content-Type": "application/json"}
    response.return_value = {"detail": "Overloaded!"}
    return response


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_anthropic_streaming_fallbacks(
    sync_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "set_verbose", True)
    client: Final = HTTPHandler(concurrent_limit=1) if sync_mode else AsyncHTTPHandler(concurrent_limit=1)
    router: Final = Router(
        model_list=[
            {
                "model_name": "anthropic/claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "test-key",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "mock_response": "Hey, how's it going?",
                },
            },
        ],
        fallbacks=[{"anthropic/claude-sonnet-4-5-20250929": ["gpt-3.5-turbo"]}],
        num_retries=0,
    )

    with patch.object(client, "post", side_effect=mock_post_streaming) as mock_client:
        if sync_mode:
            chunks: Final = tuple(
                router.completion(
                    model="anthropic/claude-sonnet-4-5-20250929",
                    messages=[{"role": "user", "content": "Hey, how's it going?"}],
                    stream=True,
                    client=client,
                )
            )
        else:
            response: Final = await router.acompletion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                stream=True,
                client=client,
            )
            chunks: Final = tuple([chunk async for chunk in response])

        mock_client.assert_called_once()
        assert chunks


def test_router_fallbacks_with_custom_model_costs() -> None:
    """
    Tests prod use-case where a custom model is registered with a different provider + custom costs.

    Goal: make sure custom model doesn't override default model costs.
    """

    default_model_info: Final = litellm.get_model_info(model="claude-sonnet-4-5-20250929")
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "api_key": "test-key",
                    "input_cost_per_token": 30,
                    "output_cost_per_token": 60,
                    "mock_response": "Hello! How can I help you today?",
                },
            },
            {
                "model_name": "claude-3-5-sonnet-aihubmix",
                "litellm_params": {
                    "model": "openai/claude-sonnet-4-5-20250929",
                    "input_cost_per_token": 0.000003,
                    "output_cost_per_token": 0.000015,
                    "api_key": "my-fake-key",
                    "mock_response": "Hello! How can I help you today?",
                },
            },
        ],
        fallbacks=[{"claude-sonnet-4-5-20250929": ["claude-3-5-sonnet-aihubmix"]}],
    )

    router.completion(
        model="claude-3-5-sonnet-aihubmix",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    fallback_model_info: Final = litellm.get_model_info(model="claude-sonnet-4-5-20250929")

    assert fallback_model_info["litellm_provider"] == "anthropic"

    response: Final = router.completion(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    assert response._hidden_params["response_cost"] > 10

    final_model_info: Final = litellm.get_model_info(model="claude-sonnet-4-5-20250929")

    assert final_model_info["input_cost_per_token"] == default_model_info["input_cost_per_token"]
    assert final_model_info["output_cost_per_token"] == default_model_info["output_cost_per_token"]


def test_router_fallbacks_with_wildcard_model_name() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "test-key",
                },
            },
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": "claude-haiku-4-5-20251001",
                    "api_key": "test-key",
                    "mock_response": "Hi this is claude!",
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["claude-3-haiku"]}],
    )

    response: Final = router.completion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        mock_testing_fallbacks=True,
    )

    assert response["choices"][0]["message"]["content"] == "Hi this is claude!"


def test_get_fallback_model_group() -> None:
    fallback_model_group, _ = get_fallback_model_group(
        fallbacks=[
            {"gpt-3.5-turbo": ["claude-3-haiku"]},
            {"*": ["claude-3-sonnet"]},
        ],
        model_group="openai/gpt-3.5-turbo",
    )
    assert fallback_model_group == ["claude-3-haiku"]


@pytest.mark.parametrize("expected_attempted_fallbacks", [0])
@pytest.mark.asyncio
async def test_router_attempted_fallbacks_in_response(
    expected_attempted_fallbacks: int,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "working-fake-endpoint",
                "litellm_params": {
                    "model": "openai/working-fake-endpoint",
                    "api_key": "test-key",
                    "mock_response": "Hello",
                },
            },
            {
                "model_name": "badly-configured-openai-endpoint",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_base": "https://example.invalid",
                },
            },
        ],
        fallbacks=[{"badly-configured-openai-endpoint": ["working-fake-endpoint"]}],
    )

    response: Final = router.completion(
        model="working-fake-endpoint",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    assert (
        response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == expected_attempted_fallbacks
    )
