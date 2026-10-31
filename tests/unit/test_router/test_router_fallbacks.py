from __future__ import annotations

import asyncio
import json
import logging
from typing import Final, Literal
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm import Router
import os
from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients


@pytest.mark.asyncio
async def test_async_fallbacks_streaming():
    """Test that router.acompletion with stream=True and mock_response works correctly."""
    litellm.set_verbose = False
    model_list = [
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
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-4o-mini"]}],
        set_verbose=False,
    )
    customHandler = MyCustomHandler()
    litellm.callbacks = [customHandler]
    user_message = "Hello, how are you?"
    try:
        response = await router.acompletion(
            model="azure/gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}],
            stream=True,
            mock_response="This is a mock streaming response",
        )
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
        assert len(chunks) > 0, "Expected at least one streaming chunk"
        router.reset()
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")
    finally:
        router.reset()


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("litellm_module_fallbacks", [True, False])
@pytest.mark.asyncio
async def test_default_model_fallbacks(sync_mode, litellm_module_fallbacks):
    """
    Related issue - https://github.com/BerriAI/litellm/issues/3623

    If model misconfigured, setup a default model for generic fallback
    """
    if litellm_module_fallbacks:
        litellm.default_fallbacks = ["my-good-model"]
    router = Router(
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
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        default_fallbacks=(
            ["my-good-model"] if litellm_module_fallbacks is False else None
        ),
    )

    if sync_mode:
        response = router.completion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
    else:
        response = await router.acompletion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )

    assert isinstance(response, litellm.ModelResponse)
    assert response.model is not None and response.model == "gpt-4o"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_client_side_fallbacks_list(sync_mode):
    """

    Tests Client Side Fallbacks

    User can pass "fallbacks": ["gpt-3.5-turbo"] and this should work

    """
    router = Router(
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
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
    )

    if sync_mode:
        response = router.completion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            fallbacks=["my-good-model"],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )
    else:
        response = await router.acompletion(
            model="bad-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            fallbacks=["my-good-model"],
            mock_testing_fallbacks=True,
            mock_response="Hey! nice day",
        )

    assert isinstance(response, litellm.ModelResponse)
    assert response.model is not None and response.model == "gpt-4o"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("content_filter_response_exception", [True, False])
@pytest.mark.parametrize("fallback_type", ["model-specific", "default"])
@pytest.mark.asyncio
async def test_router_content_policy_fallbacks(
    sync_mode, content_filter_response_exception, fallback_type
):
    os.environ["LITELLM_LOG"] = "DEBUG"

    if content_filter_response_exception:
        mock_response = Exception("content filtering policy")
    else:
        mock_response = litellm.ModelResponse(
            choices=[litellm.Choices(finish_reason="content_filter")],
            model="gpt-3.5-turbo",
            usage=litellm.Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )
    router = Router(
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
            [{"claude-sonnet-4-5-20250929": ["my-fallback-model"]}]
            if fallback_type == "model-specific"
            else None
        ),
        default_fallbacks=(
            ["my-default-fallback-model"] if fallback_type == "default" else None
        ),
    )

    if sync_mode is True:
        response = router.completion(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    else:
        response = await router.acompletion(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

    assert response.model == "my-fake-model"


def mock_post_streaming(url: str, **kwargs: object) -> MagicMock:
    response: Final = MagicMock()
    response.status_code = 529
    response.headers = {"Content-Type": "application/json"}
    response.return_value = {"detail": "Overloaded!"}
    return response


@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_anthropic_streaming_fallbacks(sync_mode):
    litellm.set_verbose = True
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    if sync_mode:
        client = HTTPHandler(concurrent_limit=1)
    else:
        client = AsyncHTTPHandler(concurrent_limit=1)

    router = Router(
        model_list=[
            {
                "model_name": "anthropic/claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Hey, how's it going?",
                },
            },
        ],
        fallbacks=[{"anthropic/claude-sonnet-4-5-20250929": ["gpt-3.5-turbo"]}],
        num_retries=0,
    )

    with patch.object(client, "post", side_effect=mock_post_streaming) as mock_client:
        chunks = []
        if sync_mode:
            response = router.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                stream=True,
                client=client,
            )
            for chunk in response:
                print(chunk)
                chunks.append(chunk)
        else:
            response = await router.acompletion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                stream=True,
                client=client,
            )
            async for chunk in response:
                print(chunk)
                chunks.append(chunk)
        print(f"RETURNED response: {response}")

        mock_client.assert_called_once()
        print(chunks)
        assert len(chunks) > 0


def test_router_fallbacks_with_custom_model_costs():
    """
    Tests prod use-case where a custom model is registered with a different provider + custom costs.

    Goal: make sure custom model doesn't override default model costs.
    """

    default_model_info = litellm.get_model_info(model="claude-sonnet-4-5-20250929")

    model_list = [
        {
            "model_name": "claude-sonnet-4-5-20250929",
            "litellm_params": {
                "model": "claude-sonnet-4-5-20250929",
                "api_key": os.environ.get("ANTHROPIC_API_KEY", "fake-key"),
                "input_cost_per_token": 30,
                "output_cost_per_token": 60,
                "mock_response": "Hello! How can I help you today?",
            },
        },
        {
            "model_name": "claude-3-5-sonnet-aihubmix",
            "litellm_params": {
                "model": "openai/claude-sonnet-4-5-20250929",
                "input_cost_per_token": 0.000003,  # 3$/M
                "output_cost_per_token": 0.000015,  # 15$/M
                "api_base": FAKE_OPENAI_API_BASE,
                "api_key": "my-fake-key",
                "mock_response": "Hello! How can I help you today?",
            },
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"claude-sonnet-4-5-20250929": ["claude-3-5-sonnet-aihubmix"]}],
    )

    router.completion(
        model="claude-3-5-sonnet-aihubmix",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    model_info = litellm.get_model_info(model="claude-sonnet-4-5-20250929")

    print(f"key: {model_info['key']}")

    assert model_info["litellm_provider"] == "anthropic"

    response = router.completion(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    print(f"response_cost: {response._hidden_params['response_cost']}")

    assert response._hidden_params["response_cost"] > 10

    model_info = litellm.get_model_info(model="claude-sonnet-4-5-20250929")

    print(f"key: {model_info['key']}")

    assert model_info["input_cost_per_token"] == default_model_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] == default_model_info["output_cost_per_token"]


def test_router_fallbacks_with_wildcard_model_name():
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": "claude-haiku-4-5-20251001",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "mock_response": "Hi this is claude!",
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["claude-3-haiku"]}],
    )

    response = router.completion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        mock_testing_fallbacks=True,
    )

    print(response)
    assert response["choices"][0]["message"]["content"] == "Hi this is claude!"


def test_get_fallback_model_group():
    from litellm.router_utils.fallback_event_handlers import get_fallback_model_group

    args = {
        "fallbacks": [
            {"gpt-3.5-turbo": ["claude-3-haiku"]},
            {"*": ["claude-3-sonnet"]},
        ],
        "model_group": "openai/gpt-3.5-turbo",
    }
    fallback_model_group, _ = get_fallback_model_group(**args)
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


class MyCustomHandler(CustomLogger):
    success: bool = False
    failure: bool = False
    previous_models: int = 0

    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API Call")
        print(
            f"previous_models: {kwargs['litellm_params']['metadata'].get('previous_models', None)}"
        )
        self.previous_models = len(
            kwargs["litellm_params"]["metadata"].get("previous_models", [])
        )  # {"previous_models": [{"model": litellm_model_name, "exception_type": AuthenticationError, "exception_string": <complete_traceback>}]}
        print(f"self.previous_models: {self.previous_models}")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print(
            f"Post-API Call - response object: {response_obj}; model: {kwargs['model']}"
        )

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")

    def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Failure")


def test_async_fallbacks(caplog, respx_mock: respx.MockRouter, monkeypatch):
    monkeypatch.setattr(litellm, "set_verbose", False)
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided: bad-key.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
    )
    verbose_router_logger.setLevel(level=logging.INFO)
    verbose_logger.setLevel(logging.CRITICAL + 1)
    verbose_proxy_logger.setLevel(logging.CRITICAL + 1)
    model_list = [
        {
            "model_name": "azure/gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/gpt-4.1-mini",
                "api_key": os.getenv("AZURE_AI_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_AI_API_BASE"),
                "mock_response": "Hello world",
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "bad-key",
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"gpt-3.5-turbo": ["azure/gpt-3.5-turbo"]}],
        num_retries=1,
    )

    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]

    async def _make_request():
        try:
            await router.acompletion(model="gpt-3.5-turbo", messages=messages, max_tokens=1)
            router.reset()
        except litellm.Timeout:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        finally:
            router.reset()
            await close_litellm_async_clients()

    asyncio.run(_make_request())
    captured_logs = [rec.message for rec in caplog.records]

    captured_logs = [
        log
        for log in captured_logs
        if "Task exception was never retrieved" not in log
        and "Task was destroyed but it is pending" not in log
        and "get_available_deployment" not in log
        and "Selected deployment for model" not in log
        and "in the Langfuse queue" not in log
        and "Unclosed client session" not in log
        and "Unclosed connector" not in log
    ]

    print("\n Captured caplog records - ", captured_logs)

    expected_logs = [
        "Falling back to model_group = azure/gpt-3.5-turbo",
        "litellm.acompletion(model=azure/gpt-4.1-mini)\x1b[32m 200 OK\x1b[0m",
        "Successful fallback b/w models.",
    ]

    assert captured_logs[-3:] == expected_logs


def _fallback_router(
    fallbacks: list[dict[str, list[str]]] | None = None,
    default_fallbacks: list[str] | None = None,
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/primary",
                    "api_key": "test-key",
                    "api_base": "https://fallback-migration.local/v1",
                },
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "openai/backup",
                    "api_key": "test-key",
                    "api_base": "https://fallback-migration.local/v1",
                },
            },
        ],
        fallbacks=fallbacks if fallbacks is not None else [{"primary": ["backup"]}],
        default_fallbacks=default_fallbacks,
        num_retries=0,
    )


def _fallback_chat_response(model: str, content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-fallback-migration",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _fallback_unavailable_response(status_code: int = 503) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "error": {
                "message": "primary unavailable",
                "type": "server_error",
                "code": "service_unavailable",
            }
        },
    )


@pytest.mark.asyncio
async def test_static_fallback_routes_after_service_unavailable(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router()
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            _fallback_unavailable_response(),
            _fallback_chat_response("backup", "served by backup"),
        )
    )

    response: Final = await router.acompletion(
        model="primary", messages=[{"role": "user", "content": "fallback prompt"}]
    )

    outbound_models: Final = tuple(json.loads(call.request.content)["model"] for call in route.calls)
    assert outbound_models == ("primary", "backup")
    assert response.choices[0].message.content == "served by backup"


def test_dynamic_fallback_routes_sync_request(respx_mock: respx.MockRouter) -> None:
    router: Final = _fallback_router(fallbacks=[])
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            _fallback_unavailable_response(),
            _fallback_chat_response("backup", "served by backup"),
        )
    )

    response: Final = router.completion(
        model="primary",
        messages=[{"role": "user", "content": "fallback prompt"}],
        fallbacks=[{"primary": ["backup"]}],
    )

    outbound_models: Final = tuple(json.loads(call.request.content)["model"] for call in route.calls)
    assert outbound_models == ("primary", "backup")
    assert response.choices[0].message.content == "served by backup"


@pytest.mark.asyncio
async def test_dynamic_fallback_routes_async_request(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router(fallbacks=[])
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            _fallback_unavailable_response(),
            _fallback_chat_response("backup", "served by backup"),
        )
    )

    response: Final = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "fallback prompt"}],
        fallbacks=[{"primary": ["backup"]}],
    )

    outbound_models: Final = tuple(json.loads(call.request.content)["model"] for call in route.calls)
    assert outbound_models == ("primary", "backup")
    assert response.choices[0].message.content == "served by backup"


@pytest.mark.asyncio
async def test_disable_fallbacks_stops_after_primary_error(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router()
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        return_value=_fallback_unavailable_response()
    )

    with pytest.raises(litellm.ServiceUnavailableError):
        await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "fallback prompt"}],
            disable_fallbacks=True,
        )

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_fallback_preserves_original_messages(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router()
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            _fallback_unavailable_response(),
            _fallback_chat_response("backup", "served by backup"),
        )
    )
    messages: Final = [{"role": "user", "content": "preserve this prompt"}]

    await router.acompletion(model="primary", messages=messages)

    outbound_messages: Final = tuple(json.loads(call.request.content)["messages"] for call in route.calls)
    assert outbound_messages == (messages, messages)


@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_embedding_fallback_routes_after_primary_error(
    sync_mode: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary-embedding",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "test-key",
                    "api_base": "https://fallback-migration.local/v1",
                },
                "model_info": {"id": "primary-embedding-deployment"},
            },
            {
                "model_name": "backup-embedding",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "test-key",
                    "api_base": "https://fallback-migration.local/v1",
                },
                "model_info": {"id": "backup-embedding-deployment"},
            },
        ],
        fallbacks=[{"primary-embedding": ["backup-embedding"]}],
        num_retries=0,
    )
    route: Final = respx_mock.post("https://fallback-migration.local/v1/embeddings").mock(
        side_effect=(
            _fallback_unavailable_response(401),
            httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            ),
        )
    )

    if sync_mode:
        response: Final = router.embedding(model="primary-embedding", input="fallback prompt")
    else:
        response: Final = await router.aembedding(model="primary-embedding", input="fallback prompt")

    outbound_models: Final = tuple(json.loads(call.request.content)["model"] for call in route.calls)
    assert outbound_models == ("text-embedding-3-small", "text-embedding-3-small")
    assert len(response.data) == 1
    assert response._hidden_params["model_id"] == "backup-embedding-deployment"
    assert route.call_count == 2


def test_model_id_fallback_returns_selected_deployment(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/primary",
                    "api_key": "test-key",
                    "api_base": "https://fallback-migration.local/v1",
                },
                "model_info": {"id": "primary-deployment"},
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "openai/backup",
                    "api_key": "test-key",
                    "api_base": "https://fallback-migration.local/v1",
                },
                "model_info": {"id": "deployment-123"},
            }
        ],
        fallbacks=[{"primary": ["deployment-123"]}],
        num_retries=0,
    )
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            _fallback_unavailable_response(),
            _fallback_chat_response("backup", "fallback by deployment id"),
        )
    )

    response: Final = router.completion(
        model="primary",
        messages=[{"role": "user", "content": "fallback prompt"}],
    )
    outbound_models: Final = tuple(json.loads(call.request.content)["model"] for call in route.calls)

    assert response._hidden_params["model_id"] == "deployment-123"
    assert outbound_models == ("primary", "backup")


@pytest.mark.asyncio
async def test_default_fallback_serves_after_primary_error(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router(fallbacks=[], default_fallbacks=["backup"])
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            _fallback_unavailable_response(),
            _fallback_chat_response("backup", "served by default fallback"),
        )
    )

    response: Final = await router.acompletion(
        model="primary", messages=[{"role": "user", "content": "fallback prompt"}]
    )

    outbound_models: Final = tuple(json.loads(call.request.content)["model"] for call in route.calls)
    assert outbound_models == ("primary", "backup")
    assert response.choices[0].message.content == "served by default fallback"


def test_usage_based_routing_falls_back_after_rpm_exhaustion() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                },
                "model_info": {"id": deployment_id},
                "rpm": rpm,
            }
            for model_name, deployment_id, rpm in (
                ("primary", "1", 1),
                ("backup", "2", 1),
                ("limited", "3", 0),
                ("available", "4", 10),
            )
        ],
        fallbacks=[
            {"primary": ["backup"]},
            {"backup": ["limited"]},
            {"limited": ["available"]},
        ],
        routing_strategy="usage-based-routing-v2",
        num_retries=0,
    )
    responses: Final = tuple(
        router.completion(
            model="primary",
            messages=[{"role": "user", "content": "usage-based fallback"}],
            mock_response="fallback response",
        )
        for _ in range(11)
    )

    assert responses[0]._hidden_params["model_id"] == "1"
    assert responses[-1]._hidden_params["model_id"] == "4"


@pytest.mark.asyncio
async def test_fallback_request_does_not_retry_primary_when_retries_are_disabled(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/primary",
                    "api_key": "test-key",
                    "api_base": "https://fallback-migration.local/v1",
                },
            }
        ],
        num_retries=0,
    )
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        return_value=_fallback_unavailable_response()
    )

    with pytest.raises(litellm.ServiceUnavailableError):
        await router.acompletion(
            model="primary", messages=[{"role": "user", "content": "retry control"}]
        )

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_default_and_model_fallbacks_do_not_repeat_failed_models(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router(default_fallbacks=["primary"])
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            _fallback_unavailable_response(401),
            _fallback_unavailable_response(),
            _fallback_chat_response("primary", "unexpected repeat"),
        )
    )

    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(
            model="primary", messages=[{"role": "user", "content": "fallback prompt"}]
        )

    outbound_models: Final = tuple(json.loads(call.request.content)["model"] for call in route.calls)
    assert outbound_models == ("primary", "backup")


@pytest.mark.asyncio
async def test_unknown_default_fallback_raises_after_primary_failure(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "expose_router_debug_in_errors", True)
    router: Final = _fallback_router(fallbacks=[], default_fallbacks=["missing"])
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        return_value=_fallback_unavailable_response()
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        await router.acompletion(
            model="primary", messages=[{"role": "user", "content": "fallback prompt"}]
        )

    assert route.call_count == 1
    assert "missing" in str(exc_info.value)


def _context_error_response_stream(
    *, event_type: str = "error", preceding_event: dict[str, object] | None = None, completed: bool = False
) -> httpx.Response:
    response: Final = {
        "id": "resp_backup" if completed else "resp_primary",
        "object": "response",
        "created_at": 1700000000,
        "model": "backup" if completed else "primary",
        "status": "completed" if completed else "in_progress",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 1.0,
        "top_p": 1.0,
    }
    error: Final = {
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
        "message": "Your input exceeds the context window of this model. Please adjust your input and try again.",
    }
    terminal: Final = (
        {"type": "response.completed", "response": response, "sequence_number": 2}
        if completed
        else {"type": event_type, "error": error, "sequence_number": 2}
        if event_type == "error"
        else {"type": event_type, "response": {**response, "status": "failed", "error": error}, "sequence_number": 2}
    )
    events: Final = (
        {"type": "response.created", "response": response, "sequence_number": 0},
        *((preceding_event,) if preceding_event is not None else ()),
        terminal,
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ("error", "response.failed"))
@pytest.mark.parametrize("context_fallback", (False, True))
async def test_responses_context_error_uses_the_configured_fallback_before_output(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, event_type: str, context_fallback: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router(fallbacks=[{"primary": ["missing"]}] if context_fallback else None)
    route: Final = respx_mock.post("https://fallback-migration.local/v1/responses").mock(
        side_effect=(
            _context_error_response_stream(event_type=event_type),
            _context_error_response_stream(completed=True),
        )
    )
    stream: Final = await router.aresponses(
        model="primary",
        input="oversized prompt",
        stream=True,
        context_window_fallbacks=[{"primary": ["backup"]}] if context_fallback else None,
    )
    events: Final = [event async for event in stream]
    assert tuple(json.loads(call.request.content)["model"] for call in route.calls) == ("primary", "backup")
    assert tuple(event.type for event in events) == ("response.created", "response.completed")
    assert events[0].response.id == events[-1].response.id
    assert events[-1].response.model == "backup"


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", ("text", "tool", "disabled", "unconfigured"))
async def test_responses_context_error_does_not_restart_committed_or_disabled_streams(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, barrier: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router(fallbacks=[] if barrier == "unconfigured" else None)
    preceding: Final = {
        "text": {
            "type": "response.output_text.delta",
            "delta": "already visible",
            "item_id": "msg_primary",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
        },
        "tool": {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call", "id": "fc_primary", "call_id": "call_primary",
                "name": "lookup", "arguments": "",
            },
            "output_index": 0,
            "sequence_number": 1,
        },
    }.get(barrier)
    route: Final = respx_mock.post("https://fallback-migration.local/v1/responses").mock(
        return_value=_context_error_response_stream(preceding_event=preceding)
    )
    stream: Final = await router.aresponses(
        model="primary", input="oversized prompt", stream=True, disable_fallbacks=barrier == "disabled"
    )
    with pytest.raises(litellm.ContextWindowExceededError):
        async for _ in stream:
            pass
    assert route.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", (False, True))
@pytest.mark.parametrize("emission", ("role", "text", "tool"))
async def test_chat_context_error_falls_back_after_role_only_but_not_after_output(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, sync_mode: bool, emission: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _fallback_router(fallbacks=[{"primary": ["missing"]}])
    delta: Final = (
        {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_primary", "type": "function", "function": {"name": "lookup", "arguments": ""}
        }]}
        if emission == "tool"
        else {"role": "assistant", "content": "visible" if emission == "text" else ""}
    )
    first_chunk: Final = {
        "id": "chatcmpl-primary", "object": "chat.completion.chunk", "created": 1700000000, "model": "primary",
        "choices": [{"index": 0, "delta": delta}],
    }
    error: Final = {"error": {
        "type": "invalid_request_error", "code": "context_length_exceeded",
        "message": "This model's maximum context length is 128000 tokens. Your messages resulted in 130000 tokens.",
    }}
    backup: Final = {
        "id": "chatcmpl-backup", "object": "chat.completion.chunk", "created": 1700000000, "model": "backup",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "backup answer"}, "finish_reason": "stop"}],
    }
    route: Final = respx_mock.post("https://fallback-migration.local/v1/chat/completions").mock(
        side_effect=(
            httpx.Response(200, content=f"data: {json.dumps(first_chunk)}\n\ndata: {json.dumps(error)}\n\n".encode()),
            httpx.Response(200, content=f"data: {json.dumps(backup)}\n\ndata: [DONE]\n\n".encode()),
        )
    )
    request: Final = {
        "model": "primary", "messages": [{"role": "user", "content": "oversized prompt"}], "stream": True,
        "context_window_fallbacks": [{"primary": ["backup"]}],
    }
    stream: Final = router.completion(**request) if sync_mode else await router.acompletion(**request)
    if emission != "role":
        with pytest.raises(litellm.ContextWindowExceededError):
            tuple(stream) if sync_mode else tuple([chunk async for chunk in stream])
        assert route.call_count == 1
        return
    chunks: Final = tuple(stream) if sync_mode else tuple([chunk async for chunk in stream])
    assert tuple(json.loads(call.request.content)["model"] for call in route.calls) == ("primary", "backup")
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "backup answer"


@pytest.mark.parametrize(
    "wrapped,has_output,generated_content,disqualifying,recoverable",
    (
        (False, False, "", False, True),
        (False, True, "", False, False),
        (True, False, "", False, True),
        (True, True, "", False, False),
        (True, False, "already visible", False, False),
        (True, False, "", True, False),
    ),
)
def test_context_stream_fallback_obeys_all_output_receipts(
    wrapped: bool, has_output: bool, generated_content: str, disqualifying: bool, recoverable: bool
) -> None:
    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import _context_window_stream_fallback_error

    original: Final = litellm.ContextWindowExceededError(message="too long", model="primary", llm_provider="openai")
    error: Final = (
        MidStreamFallbackError(
            message="too long", model="primary", llm_provider="openai", original_exception=original,
            is_pre_first_chunk=True, generated_content=generated_content, emitted_disqualifying_content=disqualifying,
        )
        if wrapped
        else original
    )
    if not recoverable:
        with pytest.raises(litellm.ContextWindowExceededError) as raised:
            _context_window_stream_fallback_error(error, "primary", "openai", has_output)
        assert raised.value is original
        return
    recovered: Final = _context_window_stream_fallback_error(error, "primary", "openai", has_output)
    assert recovered.original_exception is original
    assert recovered.is_pre_first_chunk is True
    if wrapped:
        assert recovered is error


def test_context_stream_fallback_preserves_other_errors_for_existing_continuation_policy() -> None:
    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import _context_window_stream_fallback_error

    error: Final = MidStreamFallbackError(
        message="provider unavailable", model="primary", llm_provider="openai", generated_content="already visible"
    )
    assert _context_window_stream_fallback_error(error, "primary", "openai", True) is error
