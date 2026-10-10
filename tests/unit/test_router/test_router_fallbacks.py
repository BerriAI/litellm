from __future__ import annotations

import asyncio
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


@pytest.mark.parametrize(
    ("fallback_kind", "mock_trigger"),
    [
        ("fallbacks", "mock_testing_fallbacks"),
        ("context_window_fallbacks", "mock_testing_context_fallbacks"),
        ("content_policy_fallbacks", "mock_testing_content_policy_fallbacks"),
    ],
)
@pytest.mark.asyncio
async def test_a_string_valued_rule_falls_over_to_that_one_model(fallback_kind: str, mock_trigger: str):
    """Router.__init__ documents fallbacks=[{"primary": "backup"}]; the request must land on backup, never on
    the characters of its name."""
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "gpt-5.4-mini", "api_key": "fake", "mock_response": "primary answered"},
            },
            {
                "model_name": "backup",
                "litellm_params": {"model": "gpt-5.4-nano", "api_key": "fake", "mock_response": "backup answered"},
            },
        ],
        num_retries=0,
        **{fallback_kind: [{"primary": "backup"}]},
    )

    response: Final = await router.acompletion(
        model="primary", messages=[{"role": "user", "content": "hi"}], **{mock_trigger: True}
    )

    assert response.choices[0].message.content == "backup answered"


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
