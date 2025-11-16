import os
import sys
import traceback
from unittest import mock

from dotenv import load_dotenv

import litellm.proxy
import litellm.proxy.proxy_server

load_dotenv()
import io
import json
import os

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the desired logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import (  # Replace with the actual module where your FastAPI router is defined
    app,
    initialize,
    save_worker_config,
)
from litellm.proxy.utils import ProxyLogging

# Your bearer token
token = "sk-1234"

headers = {"Authorization": f"Bearer {token}"}

example_completion_result = {
    "choices": [
        {
            "message": {
                "content": "Whispers of the wind carry dreams to me.",
                "role": "assistant",
            }
        }
    ],
}
example_embedding_result = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [
                -0.006929283495992422,
                -0.005336422007530928,
                -4.547132266452536e-05,
                -0.024047505110502243,
                -0.006929283495992422,
                -0.005336422007530928,
                -4.547132266452536e-05,
                -0.024047505110502243,
                -0.006929283495992422,
                -0.005336422007530928,
                -4.547132266452536e-05,
                -0.024047505110502243,
            ],
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}
example_image_generation_result = {
    "created": 1589478378,
    "data": [{"url": "https://..."}, {"url": "https://..."}],
}


def mock_patch_acompletion():
    return mock.patch(
        "litellm.proxy.proxy_server.llm_router.acompletion",
        return_value=example_completion_result,
    )


def mock_patch_aembedding():
    return mock.patch(
        "litellm.proxy.proxy_server.llm_router.aembedding",
        return_value=example_embedding_result,
    )


def mock_patch_aimage_generation():
    return mock.patch(
        "litellm.proxy.proxy_server.llm_router.aimage_generation",
        return_value=example_image_generation_result,
    )


@pytest.fixture(scope="function")
def fake_env_vars(monkeypatch):
    # Set some fake environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_api_key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://fake-openai-api-base")
    monkeypatch.setenv("AZURE_API_BASE", "http://fake-azure-api-base")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake_azure_openai_api_key")
    monkeypatch.setenv("AZURE_SWEDEN_API_BASE", "http://fake-azure-sweden-api-base")
    monkeypatch.setenv("REDIS_HOST", "localhost")


@pytest.fixture(scope="function")
def client_no_auth(fake_env_vars):
    # Assuming litellm.proxy.proxy_server is an object
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
    # initialize can get run in parallel, it sets specific variables for the fast api app, sinc eit gets run in parallel different tests use the wrong variables
    asyncio.run(initialize(config=config_fp, debug=True))
    return TestClient(app)


@mock_patch_acompletion()
def test_chat_completion(mock_acompletion, client_no_auth):
    global headers
    try:
        # Your test data
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        print("testing proxy server with chat completions")
        response = client_no_auth.post("/v1/chat/completions", json=test_data)
        mock_acompletion.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "hi"},
            ],
            max_tokens=10,
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            specific_deployment=True,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        print(f"response - {response.text}")
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


def test_get_settings_request_timeout(client_no_auth):
    """
    When no timeout is set, it should use the litellm.request_timeout value
    """
    # Set a known value for litellm.request_timeout
    import litellm

    # Make a GET request to /settings
    response = client_no_auth.get("/settings")

    # Check if the request was successful
    assert response.status_code == 200

    # Parse the JSON response
    settings = response.json()
    print("settings", settings)

    assert settings["litellm.request_timeout"] == litellm.request_timeout


@pytest.mark.parametrize(
    "litellm_key_header_name",
    ["x-litellm-key", None],
)
def test_add_headers_to_request(litellm_key_header_name):
    from fastapi import Request
    from starlette.datastructures import URL
    import json
    from litellm.proxy.litellm_pre_call_utils import (
        clean_headers,
        LiteLLMProxyRequestSetup,
    )

    headers = {
        "Authorization": "Bearer 1234",
        "X-Custom-Header": "Custom-Value",
        "X-Stainless-Header": "Stainless-Value",
        "anthropic-beta": "beta-value",
    }
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")
    request._body = json.dumps({"model": "gpt-3.5-turbo"}).encode("utf-8")
    request_headers = clean_headers(headers, litellm_key_header_name)
    forwarded_headers = LiteLLMProxyRequestSetup._get_forwardable_headers(
        request_headers
    )
    assert forwarded_headers == {
        "X-Custom-Header": "Custom-Value",
        "anthropic-beta": "beta-value",
    }


@pytest.mark.parametrize(
    "litellm_key_header_name",
    ["x-litellm-key", None],
)
@pytest.mark.parametrize(
    "forward_headers",
    [True, False],
)
@mock_patch_acompletion()
def test_chat_completion_forward_headers(
    mock_acompletion, client_no_auth, litellm_key_header_name, forward_headers
):
    global headers
    try:
        if forward_headers:
            gs = getattr(litellm.proxy.proxy_server, "general_settings")
            gs["forward_client_headers_to_llm_api"] = True
            setattr(litellm.proxy.proxy_server, "general_settings", gs)
        if litellm_key_header_name is not None:
            gs = getattr(litellm.proxy.proxy_server, "general_settings")
            gs["litellm_key_header_name"] = litellm_key_header_name
            setattr(litellm.proxy.proxy_server, "general_settings", gs)
        # Your test data
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        headers_to_forward = {
            "X-Custom-Header": "Custom-Value",
            "X-Another-Header": "Another-Value",
        }

        if litellm_key_header_name is not None:
            headers_to_not_forward = {litellm_key_header_name: "Bearer 1234"}
        else:
            headers_to_not_forward = {"Authorization": "Bearer 1234"}

        received_headers = {**headers_to_forward, **headers_to_not_forward}

        print("testing proxy server with chat completions")
        response = client_no_auth.post(
            "/v1/chat/completions", json=test_data, headers=received_headers
        )
        if not forward_headers:
            assert "headers" not in mock_acompletion.call_args.kwargs
        else:
            assert mock_acompletion.call_args.kwargs["headers"] == {
                "x-custom-header": "Custom-Value",
                "x-another-header": "Another-Value",
            }

        print(f"response - {response.text}")
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


@mock_patch_acompletion()
@pytest.mark.asyncio
async def test_team_disable_guardrails(mock_acompletion, client_no_auth):
    """
    If team not allowed to turn on/off guardrails

    Raise 403 forbidden error, if request is made by team on `/key/generate` or `/chat/completions`.
    """
    import asyncio
    import json
    import time

    from fastapi import HTTPException, Request
    from starlette.datastructures import URL

    from litellm.proxy._types import (
        LiteLLM_TeamTable,
        LiteLLM_TeamTableCachedObj,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    _team_id = "1234"
    user_key = "sk-12345678"

    valid_token = UserAPIKeyAuth(
        team_id=_team_id,
        team_blocked=True,
        token=hash_token(user_key),
        last_refreshed_at=time.time(),
    )
    await asyncio.sleep(1)
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=_team_id,
        blocked=False,
        last_refreshed_at=time.time(),
        metadata={"guardrails": {"modify_guardrails": False}},
    )
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)
    user_api_key_cache.set_cache(key="team_id:{}".format(_team_id), value=team_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "hello-world")

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    body = {"metadata": {"guardrails": {"hide_secrets": False}}}
    json_bytes = json.dumps(body).encode("utf-8")

    request._body = json_bytes

    try:
        await user_api_key_auth(request=request, api_key="Bearer " + user_key)
        pytest.fail("Expected to raise 403 forbidden error.")
    except ProxyException as e:
        assert e.code == str(403)


from test_custom_callback_input import CompletionCustomHandler


@mock_patch_acompletion()
def test_custom_logger_failure_handler(mock_acompletion, client_no_auth):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    rpm_limit = 0

    mock_api_key = "sk-my-test-key"
    cache_value = UserAPIKeyAuth(token=hash_token(mock_api_key), rpm_limit=rpm_limit)

    user_api_key_cache.set_cache(key=hash_token(mock_api_key), value=cache_value)

    mock_logger = CustomLogger()
    mock_logger_unit_tests = CompletionCustomHandler()
    proxy_logging_obj: ProxyLogging = getattr(
        litellm.proxy.proxy_server, "proxy_logging_obj"
    )

    litellm.callbacks = [mock_logger, mock_logger_unit_tests]
    proxy_logging_obj._init_litellm_callbacks(llm_router=None)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "FAKE-VAR")
    setattr(litellm.proxy.proxy_server, "proxy_logging_obj", proxy_logging_obj)

    with patch.object(
        mock_logger, "async_log_failure_event", new=AsyncMock()
    ) as mock_failed_alert:
        # Your test data
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        print("testing proxy server with chat completions")
        response = client_no_auth.post(
            "/v1/chat/completions",
            json=test_data,
            headers={"Authorization": "Bearer {}".format(mock_api_key)},
        )
        assert response.status_code == 429

        # confirm async_log_failure_event is called
        mock_failed_alert.assert_called()

        assert len(mock_logger_unit_tests.errors) == 0


@mock_patch_acompletion()
def test_engines_model_chat_completions(mock_acompletion, client_no_auth):
    global headers
    try:
        # Your test data
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        print("testing proxy server with chat completions")
        response = client_no_auth.post(
            "/engines/gpt-3.5-turbo/chat/completions", json=test_data
        )
        mock_acompletion.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "hi"},
            ],
            max_tokens=10,
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            specific_deployment=True,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        print(f"response - {response.text}")
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


@mock_patch_acompletion()
def test_chat_completion_azure(mock_acompletion, client_no_auth):
    global headers
    try:
        # Your test data
        test_data = {
            "model": "azure/gpt-4.1-mini",
            "messages": [
                {"role": "user", "content": "write 1 sentence poem"},
            ],
            "max_tokens": 10,
        }

        print("testing proxy server with Azure Request /chat/completions")
        response = client_no_auth.post("/v1/chat/completions", json=test_data)

        mock_acompletion.assert_called_once_with(
            model="azure/gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "write 1 sentence poem"},
            ],
            max_tokens=10,
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            specific_deployment=True,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
        assert len(result["choices"][0]["message"]["content"]) > 0
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


# Run the test
# test_chat_completion_azure()


@mock_patch_acompletion()
def test_openai_deployments_model_chat_completions_azure(
    mock_acompletion, client_no_auth
):
    global headers
    try:
        # Your test data
        test_data = {
            "model": "azure/gpt-4.1-mini",
            "messages": [
                {"role": "user", "content": "write 1 sentence poem"},
            ],
            "max_tokens": 10,
        }

        url = "/openai/deployments/azure/gpt-4.1-mini/chat/completions"
        print(f"testing proxy server with Azure Request {url}")
        response = client_no_auth.post(url, json=test_data)

        mock_acompletion.assert_called_once_with(
            model="azure/gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "write 1 sentence poem"},
            ],
            max_tokens=10,
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            specific_deployment=True,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
        assert len(result["choices"][0]["message"]["content"]) > 0
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


# Run the test
# test_openai_deployments_model_chat_completions_azure()


### EMBEDDING
@mock_patch_aembedding()
def test_embedding(mock_aembedding, client_no_auth):
    global headers
    from litellm.proxy.proxy_server import user_custom_auth

    try:
        test_data = {
            "model": "azure/text-embedding-ada-002",
            "input": ["good morning from litellm"],
        }

        async def _pre_call_hook_side_effect(**kwargs):
            data = kwargs["data"]
            metadata = {**(data.get("metadata") or {}), "source": "unit-test"}
            data["metadata"] = metadata
            proxy_request = {**(data.get("proxy_server_request") or {})}
            proxy_request["path"] = "/v1/embeddings"
            data["proxy_server_request"] = proxy_request
            return data

        async def _post_call_success_side_effect(**kwargs):
            return kwargs["response"]

        with patch.object(
            litellm.proxy.proxy_server.proxy_logging_obj,
            "pre_call_hook",
            new=AsyncMock(side_effect=_pre_call_hook_side_effect),
        ) as mock_pre_call_hook, patch.object(
            litellm.proxy.proxy_server.proxy_logging_obj,
            "during_call_hook",
            new=AsyncMock(return_value=None),
        ) as mock_during_hook, patch.object(
            litellm.proxy.proxy_server.proxy_logging_obj,
            "post_call_success_hook",
            new=AsyncMock(side_effect=_post_call_success_side_effect),
        ):
            response = client_no_auth.post("/v1/embeddings", json=test_data)

        mock_aembedding.assert_called_once_with(
            model="azure/text-embedding-ada-002",
            input=["good morning from litellm"],
            specific_deployment=True,
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10  # this usually has len==1536 so

        call_metadata = mock_aembedding.call_args.kwargs["metadata"]
        assert call_metadata.get("source") == "unit-test"

        pre_call_kwargs = mock_pre_call_hook.await_args_list[0].kwargs
        assert (
            pre_call_kwargs.get("call_type") == "aembedding"
        ), f"expected pre_call_hook to receive call_type='aembedding', got {pre_call_kwargs.get('call_type')}"

        during_call_kwargs = mock_during_hook.await_args_list[0].kwargs
        assert (
            during_call_kwargs.get("call_type") == "embeddings"
        ), f"expected during_call_hook to receive call_type='embeddings', got {during_call_kwargs.get('call_type')}"
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


@mock_patch_aembedding()
def test_bedrock_embedding(mock_aembedding, client_no_auth):
    global headers
    from litellm.proxy.proxy_server import user_custom_auth

    try:
        test_data = {
            "model": "amazon-embeddings",
            "input": ["good morning from litellm"],
        }

        response = client_no_auth.post("/v1/embeddings", json=test_data)

        mock_aembedding.assert_called_once_with(
            model="amazon-embeddings",
            input=["good morning from litellm"],
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        assert response.status_code == 200
        print(response.status_code, response.text)
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10  # this usually has len==1536 so
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


@pytest.mark.skip(reason="AWS Suspended Account")
def test_sagemaker_embedding(client_no_auth):
    global headers
    from litellm.proxy.proxy_server import user_custom_auth

    try:
        test_data = {
            "model": "GPT-J 6B - Sagemaker Text Embedding (Internal)",
            "input": ["good morning from litellm"],
        }

        response = client_no_auth.post("/v1/embeddings", json=test_data)

        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10  # this usually has len==1536 so
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


# Run the test
# test_embedding()
#### IMAGE GENERATION


@mock_patch_aimage_generation()
def test_img_gen(mock_aimage_generation, client_no_auth):
    global headers
    from litellm.proxy.proxy_server import user_custom_auth

    try:
        test_data = {
            "model": "dall-e-3",
            "prompt": "A cute baby sea otter",
            "n": 1,
            "size": "1024x1024",
        }

        response = client_no_auth.post("/v1/images/generations", json=test_data)

        mock_aimage_generation.assert_called_once_with(
            model="dall-e-3",
            prompt="A cute baby sea otter",
            n=1,
            size="1024x1024",
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["url"]))
        assert len(result["data"][0]["url"]) > 10
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


#### ADDITIONAL
@pytest.mark.skip(reason="test via docker tests. Requires prisma client.")
def test_add_new_model(client_no_auth):
    global headers
    try:
        test_data = {
            "model_name": "test_openai_models",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
            },
            "model_info": {"description": "this is a test openai model"},
        }
        client_no_auth.post("/model/new", json=test_data, headers=headers)
        response = client_no_auth.get("/model/info", headers=headers)
        assert response.status_code == 200
        result = response.json()
        print(f"response: {result}")
        model_info = None
        for m in result["data"]:
            if m["model_name"] == "test_openai_models":
                model_info = m["model_info"]
        assert model_info["description"] == "this is a test openai model"
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


def test_health(client_no_auth):
    global headers
    import logging
    import time

    from litellm._logging import verbose_logger, verbose_proxy_logger

    verbose_proxy_logger.setLevel(logging.DEBUG)

    try:
        response = client_no_auth.get("/health")
        assert response.status_code == 200
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


# test_add_new_model()

from litellm.integrations.custom_logger import CustomLogger


class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API Call")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")
        assert kwargs["user"] == "proxy-user"
        assert kwargs["model"] == "gpt-3.5-turbo"
        assert kwargs["max_tokens"] == 10


customHandler = MyCustomHandler()


@mock_patch_acompletion()
def test_chat_completion_optional_params(mock_acompletion, client_no_auth):
    # [PROXY: PROD TEST] - DO NOT DELETE
    # This tests if all the /chat/completion params are passed to litellm
    try:
        # Your test data
        litellm.set_verbose = True
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
            "user": "proxy-user",
        }

        litellm.callbacks = [customHandler]
        print("testing proxy server: optional params")
        response = client_no_auth.post("/v1/chat/completions", json=test_data)
        mock_acompletion.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "hi"},
            ],
            max_tokens=10,
            user="proxy-user",
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            specific_deployment=True,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
            secret_fields=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)


# Run the test
# test_chat_completion_optional_params()


# Test Reading config.yaml file
from litellm.proxy.proxy_server import ProxyConfig


@pytest.mark.skip(reason="local variable conflicts. needs to be refactored.")
@mock.patch("litellm.proxy.proxy_server.litellm.Cache")
def test_load_router_config(mock_cache, fake_env_vars):
    mock_cache.return_value.cache.__dict__ = {"redis_client": None}
    mock_cache.return_value.supported_call_types = [
        "completion",
        "acompletion",
        "embedding",
        "aembedding",
        "atranscription",
        "transcription",
    ]

    try:
        import asyncio

        print("testing reading config")
        # this is a basic config.yaml with only a model
        filepath = os.path.dirname(os.path.abspath(__file__))
        proxy_config = ProxyConfig()
        result = asyncio.run(
            proxy_config.load_config(
                router=None,
                config_file_path=f"{filepath}/example_config_yaml/simple_config.yaml",
            )
        )
        print(result)
        assert len(result[1]) == 1

        # this is a load balancing config yaml
        result = asyncio.run(
            proxy_config.load_config(
                router=None,
                config_file_path=f"{filepath}/example_config_yaml/azure_config.yaml",
            )
        )
        print(result)
        assert len(result[1]) == 2

        # config with general settings - custom callbacks
        result = asyncio.run(
            proxy_config.load_config(
                router=None,
                config_file_path=f"{filepath}/example_config_yaml/azure_config.yaml",
            )
        )
        print(result)
        assert len(result[1]) == 2

        # tests for litellm.cache set from config
        print("testing reading proxy config for cache")
        litellm.cache = None
        asyncio.run(
            proxy_config.load_config(
                router=None,
                config_file_path=f"{filepath}/example_config_yaml/cache_no_params.yaml",
            )
        )
        assert litellm.cache is not None
        assert "redis_client" in vars(
            litellm.cache.cache
        )  # it should default to redis on proxy
        assert litellm.cache.supported_call_types == [
            "completion",
            "acompletion",
            "embedding",
            "aembedding",
            "atranscription",
            "transcription",
        ]  # init with all call types

        litellm.disable_cache()

        print("testing reading proxy config for cache with params")
        mock_cache.return_value.supported_call_types = [
            "embedding",
            "aembedding",
        ]
        asyncio.run(
            proxy_config.load_config(
                router=None,
                config_file_path=f"{filepath}/example_config_yaml/cache_with_params.yaml",
            )
        )
        assert litellm.cache is not None
        print(litellm.cache)
        print(litellm.cache.supported_call_types)
        print(vars(litellm.cache.cache))
        assert "redis_client" in vars(
            litellm.cache.cache
        )  # it should default to redis on proxy
        assert litellm.cache.supported_call_types == [
            "embedding",
            "aembedding",
        ]  # init with all call types

    except Exception as e:
        pytest.fail(
            f"Proxy: Got exception reading config: {str(e)}\n{traceback.format_exc()}"
        )


# test_load_router_config()


@pytest.mark.asyncio
async def test_team_update_redis():
    """
    Tests if team update, updates the redis cache if set
    """
    from litellm.caching.caching import DualCache, RedisCache
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.auth.auth_checks import _cache_team_object

    proxy_logging_obj: ProxyLogging = getattr(
        litellm.proxy.proxy_server, "proxy_logging_obj"
    )

    redis_cache = RedisCache()

    with patch.object(
        redis_cache,
        "async_set_cache",
        new=AsyncMock(),
    ) as mock_client:
        await _cache_team_object(
            team_id="1234",
            team_table=LiteLLM_TeamTableCachedObj(team_id="1234"),
            user_api_key_cache=DualCache(redis_cache=redis_cache),
            proxy_logging_obj=proxy_logging_obj,
        )

        mock_client.assert_called()


@pytest.mark.asyncio
async def test_get_team_redis(client_no_auth):
    """
    Tests if get_team_object gets value from redis cache, if set
    """
    from litellm.caching.caching import DualCache, RedisCache
    from litellm.proxy.auth.auth_checks import get_team_object

    proxy_logging_obj: ProxyLogging = getattr(
        litellm.proxy.proxy_server, "proxy_logging_obj"
    )

    redis_cache = RedisCache()

    with patch.object(
        redis_cache,
        "async_get_cache",
        new=AsyncMock(),
    ) as mock_client:
        try:
            await get_team_object(
                team_id="1234",
                user_api_key_cache=DualCache(redis_cache=redis_cache),
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
                prisma_client=AsyncMock(),
            )
        except Exception as e:
            pass

        mock_client.assert_called_once()


import random
from litellm._uuid import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from litellm.proxy._types import (
    LitellmUserRoles,
    NewUserRequest,
    TeamMemberAddRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.proxy.management_endpoints.team_endpoints import team_member_add
from test_key_generate_prisma import prisma_client


@pytest.mark.parametrize(
    "user_role",
    [LitellmUserRoles.INTERNAL_USER.value, LitellmUserRoles.PROXY_ADMIN.value],
)
@pytest.mark.asyncio
async def test_create_user_default_budget(prisma_client, user_role):

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm, "max_internal_user_budget", 10)
    setattr(litellm, "internal_user_budget_duration", "5m")
    await litellm.proxy.proxy_server.prisma_client.connect()
    user = f"ishaan {uuid.uuid4().hex}"
    request = NewUserRequest(
        user_id=user, user_role=user_role
    )  # create a key with no budget
    with patch.object(
        litellm.proxy.proxy_server.prisma_client, "insert_data", new=AsyncMock()
    ) as mock_client:
        await new_user(
            request,
        )

        mock_client.assert_called()

        print(f"mock_client.call_args: {mock_client.call_args}")
        print("mock_client.call_args.kwargs: {}".format(mock_client.call_args.kwargs))

        if user_role == LitellmUserRoles.INTERNAL_USER.value:
            assert (
                mock_client.call_args.kwargs["data"]["max_budget"]
                == litellm.max_internal_user_budget
            )
            assert (
                mock_client.call_args.kwargs["data"]["budget_duration"]
                == litellm.internal_user_budget_duration
            )

        else:
            assert mock_client.call_args.kwargs["data"]["max_budget"] is None
            assert mock_client.call_args.kwargs["data"]["budget_duration"] is None


@pytest.mark.parametrize("new_member_method", ["user_id", "user_email"])
@pytest.mark.asyncio
async def test_create_team_member_add(prisma_client, new_member_method):
    import time

    from fastapi import Request

    from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable
    from litellm.proxy.proxy_server import hash_token, user_api_key_cache

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm, "max_internal_user_budget", 10)
    setattr(litellm, "internal_user_budget_duration", "5m")
    await litellm.proxy.proxy_server.prisma_client.connect()
    user = f"ishaan {uuid.uuid4().hex}"
    _team_id = "litellm-test-client-id-new"
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=_team_id,
        blocked=False,
        last_refreshed_at=time.time(),
        metadata={"guardrails": {"modify_guardrails": False}},
    )
    # user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)
    user_api_key_cache.set_cache(key="team_id:{}".format(_team_id), value=team_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    if new_member_method == "user_id":
        data = {
            "team_id": _team_id,
            "member": [{"role": "user", "user_id": user}],
        }
    elif new_member_method == "user_email":
        data = {
            "team_id": _team_id,
            "member": [{"role": "user", "user_email": user}],
        }
    team_member_add_request = TeamMemberAddRequest(**data)

    with patch(
        "litellm.proxy.proxy_server.prisma_client.db.litellm_usertable",
        new_callable=AsyncMock,
    ) as mock_litellm_usertable, patch(
        "litellm.proxy.auth.auth_checks._get_team_object_from_user_api_key_cache",
        new=AsyncMock(return_value=team_obj),
    ) as mock_team_obj:

        mock_client = AsyncMock(
            return_value=LiteLLM_UserTable(
                user_id="1234", max_budget=100, user_email="1234"
            )
        )
        mock_litellm_usertable.upsert = mock_client
        mock_litellm_usertable.find_many = AsyncMock(return_value=None)
        team_mock_client = AsyncMock()
        original_val = getattr(
            litellm.proxy.proxy_server.prisma_client.db, "litellm_teamtable"
        )
        litellm.proxy.proxy_server.prisma_client.db.litellm_teamtable = team_mock_client

        team_mock_client.update = AsyncMock(
            return_value=LiteLLM_TeamTableCachedObj(team_id="1234")
        )

        print(f"team_member_add_request={team_member_add_request}")
        await team_member_add(
            data=team_member_add_request,
            user_api_key_dict=UserAPIKeyAuth(user_role="proxy_admin"),
        )

        mock_client.assert_called()

        print(f"mock_client.call_args: {mock_client.call_args}")
        print("mock_client.call_args.kwargs: {}".format(mock_client.call_args.kwargs))

        assert (
            mock_client.call_args.kwargs["data"]["create"]["max_budget"]
            == litellm.max_internal_user_budget
        )
        assert (
            mock_client.call_args.kwargs["data"]["create"]["budget_duration"]
            == litellm.internal_user_budget_duration
        )

        litellm.proxy.proxy_server.prisma_client.db.litellm_teamtable = original_val


@pytest.mark.parametrize("team_member_role", ["admin", "user"])
@pytest.mark.parametrize("team_route", ["/team/member_add", "/team/member_delete"])
@pytest.mark.asyncio
async def test_create_team_member_add_team_admin_user_api_key_auth(
    prisma_client, team_member_role, team_route
):
    import time

    from fastapi import Request

    from litellm.proxy._types import LiteLLM_TeamTableCachedObj, Member
    from litellm.proxy.proxy_server import (
        ProxyException,
        hash_token,
        user_api_key_auth,
        user_api_key_cache,
    )

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm, "max_internal_user_budget", 10)
    setattr(litellm, "internal_user_budget_duration", "5m")
    await litellm.proxy.proxy_server.prisma_client.connect()
    user = f"ishaan {uuid.uuid4().hex}"
    _team_id = "litellm-test-client-id-new"
    user_key = "sk-12345678"

    valid_token = UserAPIKeyAuth(
        team_id=_team_id,
        token=hash_token(user_key),
        team_member=Member(role=team_member_role, user_id=user),
        last_refreshed_at=time.time(),
    )
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)

    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=_team_id,
        blocked=False,
        last_refreshed_at=time.time(),
        metadata={"guardrails": {"modify_guardrails": False}},
    )

    user_api_key_cache.set_cache(key="team_id:{}".format(_team_id), value=team_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)

    ## TEST IF TEAM ADMIN ALLOWED TO CALL /MEMBER_ADD ENDPOINT
    import json

    from starlette.datastructures import URL

    request = Request(scope={"type": "http"})
    request._url = URL(url=team_route)

    body = {}
    json_bytes = json.dumps(body).encode("utf-8")

    request._body = json_bytes

    ## ALLOWED BY USER_API_KEY_AUTH
    await user_api_key_auth(request=request, api_key="Bearer " + user_key)


@pytest.mark.parametrize("new_member_method", ["user_id", "user_email"])
@pytest.mark.parametrize("user_role", ["admin", "user"])
@pytest.mark.asyncio
async def test_create_team_member_add_team_admin(
    prisma_client, new_member_method, user_role
):
    """
    Relevant issue - https://github.com/BerriAI/litellm/issues/5300

    Allow team admins to:
        - Add and remove team members
        - raise error if team member not an existing 'internal_user'
    """
    import time

    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_TeamTableCachedObj,
        LiteLLM_UserTable,
        Member,
    )
    from litellm.proxy.proxy_server import (
        HTTPException,
        ProxyException,
        hash_token,
        user_api_key_auth,
        user_api_key_cache,
    )

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm, "max_internal_user_budget", 10)
    setattr(litellm, "internal_user_budget_duration", "5m")
    await litellm.proxy.proxy_server.prisma_client.connect()
    user = f"ishaan {uuid.uuid4().hex}"
    _team_id = "litellm-test-client-id-new"
    user_key = "sk-12345678"
    team_admin = f"krrish {uuid.uuid4().hex}"

    valid_token = UserAPIKeyAuth(
        team_id=_team_id,
        user_id=team_admin,
        token=hash_token(user_key),
        last_refreshed_at=time.time(),
    )
    user_api_key_cache.set_cache(key=hash_token(user_key), value=valid_token)

    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=_team_id,
        blocked=False,
        last_refreshed_at=time.time(),
        members_with_roles=[Member(role=user_role, user_id=team_admin)],
        metadata={"guardrails": {"modify_guardrails": False}},
    )

    user_api_key_cache.set_cache(key="team_id:{}".format(_team_id), value=team_obj)

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    if new_member_method == "user_id":
        data = {
            "team_id": _team_id,
            "member": [{"role": "user", "user_id": user}],
        }
    elif new_member_method == "user_email":
        data = {
            "team_id": _team_id,
            "member": [{"role": "user", "user_email": user}],
        }
    team_member_add_request = TeamMemberAddRequest(**data)

    with patch(
        "litellm.proxy.proxy_server.prisma_client.db.litellm_usertable",
        new_callable=AsyncMock,
    ) as mock_litellm_usertable, patch(
        "litellm.proxy.auth.auth_checks._get_team_object_from_user_api_key_cache",
        new=AsyncMock(return_value=team_obj),
    ) as mock_team_obj:
        mock_client = AsyncMock(
            return_value=LiteLLM_UserTable(
                user_id="1234", max_budget=100, user_email="1234"
            )
        )
        mock_litellm_usertable.upsert = mock_client
        mock_litellm_usertable.find_many = AsyncMock(return_value=None)

        team_mock_client = AsyncMock()
        original_val = getattr(
            litellm.proxy.proxy_server.prisma_client.db, "litellm_teamtable"
        )
        litellm.proxy.proxy_server.prisma_client.db.litellm_teamtable = team_mock_client

        team_mock_client.update = AsyncMock(
            return_value=LiteLLM_TeamTableCachedObj(team_id="1234")
        )

        try:
            await team_member_add(
                data=team_member_add_request,
                user_api_key_dict=valid_token,
            )
        except HTTPException as e:
            if user_role == "user":
                assert e.status_code == 403
                return
            else:
                raise e

        mock_client.assert_called()

        print(f"mock_client.call_args: {mock_client.call_args}")
        print("mock_client.call_args.kwargs: {}".format(mock_client.call_args.kwargs))

        assert (
            mock_client.call_args.kwargs["data"]["create"]["max_budget"]
            == litellm.max_internal_user_budget
        )
        assert (
            mock_client.call_args.kwargs["data"]["create"]["budget_duration"]
            == litellm.internal_user_budget_duration
        )

        litellm.proxy.proxy_server.prisma_client.db.litellm_teamtable = original_val


@pytest.mark.asyncio
async def test_user_info_team_list(prisma_client):
    """Assert user_info for admin calls team_list function"""
    from litellm.proxy._types import LiteLLM_UserTable

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.list_team",
        new_callable=AsyncMock,
    ) as mock_client:

        prisma_client.get_data = AsyncMock(
            return_value=LiteLLM_UserTable(
                user_role="proxy_admin",
                user_id="default_user_id",
                max_budget=None,
                user_email="",
            )
        )

        try:
            await user_info(
                request=MagicMock(),
                user_id=None,
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="sk-1234", user_id="default_user_id"
                ),
            )
        except Exception:
            pass

        mock_client.assert_called()


@pytest.mark.skip(reason="Local test")
@pytest.mark.asyncio
async def test_add_callback_via_key(prisma_client):
    """
    Test if callback specified in key, is used.
    """
    global headers
    import json

    from fastapi import HTTPException, Request, Response
    from starlette.datastructures import URL

    from litellm.proxy.proxy_server import chat_completion

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    litellm.set_verbose = True

    try:
        # Your test data
        test_data = {
            "model": "azure/gpt-4.1-mini",
            "messages": [
                {"role": "user", "content": "write 1 sentence poem"},
            ],
            "max_tokens": 10,
            "mock_response": "Hello world",
            "api_key": "my-fake-key",
        }

        request = Request(scope={"type": "http", "method": "POST", "headers": {}})
        request._url = URL(url="/chat/completions")

        json_bytes = json.dumps(test_data).encode("utf-8")

        request._body = json_bytes

        with patch.object(
            litellm.litellm_core_utils.litellm_logging,
            "LangFuseLogger",
            new=MagicMock(),
        ) as mock_client:
            resp = await chat_completion(
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(
                    metadata={
                        "logging": [
                            {
                                "callback_name": "langfuse",  # 'otel', 'langfuse', 'lunary'
                                "callback_type": "success",  # set, if required by integration - future improvement, have logging tools work for success + failure by default
                                "callback_vars": {
                                    "langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY",
                                    "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY",
                                    "langfuse_host": "https://us.cloud.langfuse.com",
                                },
                            }
                        ]
                    }
                ),
            )
            print(resp)
            mock_client.assert_called()
            mock_client.return_value.log_event.assert_called()
            args, kwargs = mock_client.return_value.log_event.call_args
            kwargs = kwargs["kwargs"]
            assert "user_api_key_metadata" in kwargs["litellm_params"]["metadata"]
            assert (
                "logging"
                in kwargs["litellm_params"]["metadata"]["user_api_key_metadata"]
            )
            checked_keys = False
            for item in kwargs["litellm_params"]["metadata"]["user_api_key_metadata"][
                "logging"
            ]:
                for k, v in item["callback_vars"].items():
                    print("k={}, v={}".format(k, v))
                    if "key" in k:
                        assert "os.environ" in v
                        checked_keys = True

            assert checked_keys
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_type, expected_success_callbacks, expected_failure_callbacks",
    [
        ("success", ["langfuse"], []),
        ("failure", [], ["langfuse"]),
        ("success_and_failure", ["langfuse"], ["langfuse"]),
    ],
)
async def test_add_callback_via_key_litellm_pre_call_utils(
    prisma_client, callback_type, expected_success_callbacks, expected_failure_callbacks
):
    import json

    from fastapi import HTTPException, Request, Response
    from starlette.datastructures import URL

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    proxy_config = getattr(litellm.proxy.proxy_server, "proxy_config")

    request = Request(scope={"type": "http", "method": "POST", "headers": {}})
    request._url = URL(url="/chat/completions")

    test_data = {
        "model": "azure/gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": "write 1 sentence poem"},
        ],
        "max_tokens": 10,
        "mock_response": "Hello world",
        "api_key": "my-fake-key",
    }

    json_bytes = json.dumps(test_data).encode("utf-8")

    request._body = json_bytes

    data = {
        "data": {
            "model": "azure/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "write 1 sentence poem"}],
            "max_tokens": 10,
            "mock_response": "Hello world",
            "api_key": "my-fake-key",
        },
        "request": request,
        "user_api_key_dict": UserAPIKeyAuth(
            token=None,
            key_name=None,
            key_alias=None,
            spend=0.0,
            max_budget=None,
            expires=None,
            models=[],
            aliases={},
            config={},
            user_id=None,
            team_id=None,
            max_parallel_requests=None,
            metadata={
                "logging": [
                    {
                        "callback_name": "langfuse",
                        "callback_type": callback_type,
                        "callback_vars": {
                            "langfuse_public_key": "my-mock-public-key",
                            "langfuse_secret_key": "my-mock-secret-key",
                            "langfuse_host": "https://us.cloud.langfuse.com",
                        },
                    }
                ]
            },
            tpm_limit=None,
            rpm_limit=None,
            budget_duration=None,
            budget_reset_at=None,
            allowed_cache_controls=[],
            permissions={},
            model_spend={},
            model_max_budget={},
            soft_budget_cooldown=False,
            litellm_budget_table=None,
            org_id=None,
            team_spend=None,
            team_alias=None,
            team_tpm_limit=None,
            team_rpm_limit=None,
            team_max_budget=None,
            team_models=[],
            team_blocked=False,
            soft_budget=None,
            team_model_aliases=None,
            team_member_spend=None,
            team_metadata=None,
            end_user_id=None,
            end_user_tpm_limit=None,
            end_user_rpm_limit=None,
            end_user_max_budget=None,
            last_refreshed_at=None,
            api_key=None,
            user_role=None,
            allowed_model_region=None,
            parent_otel_span=None,
        ),
        "proxy_config": proxy_config,
        "general_settings": {},
        "version": "0.0.0",
    }

    new_data = await add_litellm_data_to_request(**data)
    print("NEW DATA: {}".format(new_data))

    assert "langfuse_public_key" in new_data
    assert new_data["langfuse_public_key"] == "my-mock-public-key"
    assert "langfuse_secret_key" in new_data
    assert new_data["langfuse_secret_key"] == "my-mock-secret-key"

    if expected_success_callbacks:
        assert "success_callback" in new_data
        assert new_data["success_callback"] == expected_success_callbacks

    if expected_failure_callbacks:
        assert "failure_callback" in new_data
        assert new_data["failure_callback"] == expected_failure_callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disable_fallbacks_set",
    [
        True,
        False,
    ],
)
async def test_disable_fallbacks_by_key(disable_fallbacks_set):
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    key_metadata = {"disable_fallbacks": disable_fallbacks_set}
    existing_data = {
        "model": "azure/gpt-4.1-mini",
        "messages": [{"role": "user", "content": "write 1 sentence poem"}],
    }
    data = LiteLLMProxyRequestSetup.add_key_level_controls(
        key_metadata=key_metadata,
        data=existing_data,
        _metadata_variable_name="metadata",
    )

    assert data["disable_fallbacks"] == disable_fallbacks_set


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_type, expected_success_callbacks, expected_failure_callbacks",
    [
        ("success", ["gcs_bucket"], []),
        ("failure", [], ["gcs_bucket"]),
        ("success_and_failure", ["gcs_bucket"], ["gcs_bucket"]),
    ],
)
async def test_add_callback_via_key_litellm_pre_call_utils_gcs_bucket(
    prisma_client, callback_type, expected_success_callbacks, expected_failure_callbacks
):
    import json

    from fastapi import HTTPException, Request, Response
    from starlette.datastructures import URL

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    proxy_config = getattr(litellm.proxy.proxy_server, "proxy_config")

    request = Request(scope={"type": "http", "method": "POST", "headers": {}})
    request._url = URL(url="/chat/completions")

    test_data = {
        "model": "azure/gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": "write 1 sentence poem"},
        ],
        "max_tokens": 10,
        "mock_response": "Hello world",
        "api_key": "my-fake-key",
    }

    json_bytes = json.dumps(test_data).encode("utf-8")

    request._body = json_bytes

    data = {
        "data": {
            "model": "azure/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "write 1 sentence poem"}],
            "max_tokens": 10,
            "mock_response": "Hello world",
            "api_key": "my-fake-key",
        },
        "request": request,
        "user_api_key_dict": UserAPIKeyAuth(
            token=None,
            key_name=None,
            key_alias=None,
            spend=0.0,
            max_budget=None,
            expires=None,
            models=[],
            aliases={},
            config={},
            user_id=None,
            team_id=None,
            max_parallel_requests=None,
            metadata={
                "logging": [
                    {
                        "callback_name": "gcs_bucket",
                        "callback_type": callback_type,
                        "callback_vars": {
                            "gcs_bucket_name": "key-logging-project1",
                            "gcs_path_service_account": "pathrise-convert-1606954137718-a956eef1a2a8.json",
                        },
                    }
                ]
            },
            tpm_limit=None,
            rpm_limit=None,
            budget_duration=None,
            budget_reset_at=None,
            allowed_cache_controls=[],
            permissions={},
            model_spend={},
            model_max_budget={},
            soft_budget_cooldown=False,
            litellm_budget_table=None,
            org_id=None,
            team_spend=None,
            team_alias=None,
            team_tpm_limit=None,
            team_rpm_limit=None,
            team_max_budget=None,
            team_models=[],
            team_blocked=False,
            soft_budget=None,
            team_model_aliases=None,
            team_member_spend=None,
            team_metadata=None,
            end_user_id=None,
            end_user_tpm_limit=None,
            end_user_rpm_limit=None,
            end_user_max_budget=None,
            last_refreshed_at=None,
            api_key=None,
            user_role=None,
            allowed_model_region=None,
            parent_otel_span=None,
        ),
        "proxy_config": proxy_config,
        "general_settings": {},
        "version": "0.0.0",
    }

    new_data = await add_litellm_data_to_request(**data)
    print("NEW DATA: {}".format(new_data))

    assert "gcs_bucket_name" in new_data
    assert new_data["gcs_bucket_name"] == "key-logging-project1"
    assert "gcs_path_service_account" in new_data
    assert (
        new_data["gcs_path_service_account"]
        == "pathrise-convert-1606954137718-a956eef1a2a8.json"
    )

    if expected_success_callbacks:
        assert "success_callback" in new_data
        assert new_data["success_callback"] == expected_success_callbacks

    if expected_failure_callbacks:
        assert "failure_callback" in new_data
        assert new_data["failure_callback"] == expected_failure_callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_type, expected_success_callbacks, expected_failure_callbacks",
    [
        ("success", ["langsmith"], []),
        ("failure", [], ["langsmith"]),
        ("success_and_failure", ["langsmith"], ["langsmith"]),
    ],
)
async def test_add_callback_via_key_litellm_pre_call_utils_langsmith(
    prisma_client, callback_type, expected_success_callbacks, expected_failure_callbacks
):
    import json

    from fastapi import HTTPException, Request, Response
    from starlette.datastructures import URL

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    proxy_config = getattr(litellm.proxy.proxy_server, "proxy_config")

    request = Request(scope={"type": "http", "method": "POST", "headers": {}})
    request._url = URL(url="/chat/completions")

    test_data = {
        "model": "azure/gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": "write 1 sentence poem"},
        ],
        "max_tokens": 10,
        "mock_response": "Hello world",
        "api_key": "my-fake-key",
    }

    json_bytes = json.dumps(test_data).encode("utf-8")

    request._body = json_bytes

    data = {
        "data": {
            "model": "azure/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "write 1 sentence poem"}],
            "max_tokens": 10,
            "mock_response": "Hello world",
            "api_key": "my-fake-key",
        },
        "request": request,
        "user_api_key_dict": UserAPIKeyAuth(
            token=None,
            key_name=None,
            key_alias=None,
            spend=0.0,
            max_budget=None,
            expires=None,
            models=[],
            aliases={},
            config={},
            user_id=None,
            team_id=None,
            max_parallel_requests=None,
            metadata={
                "logging": [
                    {
                        "callback_name": "langsmith",
                        "callback_type": callback_type,
                        "callback_vars": {
                            "langsmith_api_key": "ls-1234",
                            "langsmith_project": "pr-brief-resemblance-72",
                            "langsmith_base_url": "https://api.smith.langchain.com",
                        },
                    }
                ]
            },
            tpm_limit=None,
            rpm_limit=None,
            budget_duration=None,
            budget_reset_at=None,
            allowed_cache_controls=[],
            permissions={},
            model_spend={},
            model_max_budget={},
            soft_budget_cooldown=False,
            litellm_budget_table=None,
            org_id=None,
            team_spend=None,
            team_alias=None,
            team_tpm_limit=None,
            team_rpm_limit=None,
            team_max_budget=None,
            team_models=[],
            team_blocked=False,
            soft_budget=None,
            team_model_aliases=None,
            team_member_spend=None,
            team_metadata=None,
            end_user_id=None,
            end_user_tpm_limit=None,
            end_user_rpm_limit=None,
            end_user_max_budget=None,
            last_refreshed_at=None,
            api_key=None,
            user_role=None,
            allowed_model_region=None,
            parent_otel_span=None,
        ),
        "proxy_config": proxy_config,
        "general_settings": {},
        "version": "0.0.0",
    }

    new_data = await add_litellm_data_to_request(**data)
    print("NEW DATA: {}".format(new_data))

    assert "langsmith_api_key" in new_data
    assert new_data["langsmith_api_key"] == "ls-1234"
    assert "langsmith_project" in new_data
    assert new_data["langsmith_project"] == "pr-brief-resemblance-72"
    assert "langsmith_base_url" in new_data
    assert new_data["langsmith_base_url"] == "https://api.smith.langchain.com"

    if expected_success_callbacks:
        assert "success_callback" in new_data
        assert new_data["success_callback"] == expected_success_callbacks

    if expected_failure_callbacks:
        assert "failure_callback" in new_data
        assert new_data["failure_callback"] == expected_failure_callbacks


@pytest.mark.asyncio
async def test_gemini_pass_through_endpoint():
    from starlette.datastructures import URL

    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        Request,
        Response,
        gemini_proxy_route,
    )

    body = b"""
        {
            "contents": [{
                "parts":[{
                "text": "The quick brown fox jumps over the lazy dog."
                }]
                }]
        }
        """

    # Construct the scope dictionary
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gemini/v1beta/models/gemini-2.5-flash:countTokens",
        "query_string": b"key=sk-1234",
        "headers": [
            (b"content-type", b"application/json"),
        ],
    }

    # Create a new Request object
    async def async_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        scope=scope,
        receive=async_receive,
    )

    resp = await gemini_proxy_route(
        endpoint="v1beta/models/gemini-2.5-flash:countTokens?key=sk-1234",
        request=request,
        fastapi_response=Response(),
    )

    print(resp.body)


@pytest.mark.parametrize("hidden", [True, False])
@pytest.mark.asyncio
async def test_proxy_model_group_alias_checks(prisma_client, hidden):
    """
    Check if model group alias is returned on

    `/v1/models`
    `/v1/model/info`
    `/v1/model_group/info`
    """
    import json

    from fastapi import HTTPException, Request, Response
    from starlette.datastructures import URL

    from litellm.proxy.proxy_server import model_group_info, model_info_v1, model_list

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    proxy_config = getattr(litellm.proxy.proxy_server, "proxy_config")

    _model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo"},
        }
    ]
    model_alias = "gpt-4"
    router = litellm.Router(
        model_list=_model_list,
        model_group_alias={model_alias: {"model": "gpt-3.5-turbo", "hidden": hidden}},
    )
    setattr(litellm.proxy.proxy_server, "llm_router", router)
    setattr(litellm.proxy.proxy_server, "llm_model_list", _model_list)

    request = Request(scope={"type": "http", "method": "POST", "headers": {}})
    request._url = URL(url="/v1/models")

    resp = await model_list(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
    )

    if hidden:
        assert len(resp["data"]) == 1
    else:
        assert len(resp["data"]) == 2
    print(resp)

    resp = await model_info_v1(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
    )
    models = resp["data"]
    is_model_alias_in_list = False
    for item in models:
        if model_alias == item["model_name"]:
            is_model_alias_in_list = True

    if hidden:
        assert is_model_alias_in_list is False
    else:
        assert is_model_alias_in_list

    resp = await model_group_info(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
    )
    print(f"resp: {resp}")
    models = resp["data"]
    is_model_alias_in_list = False
    print(f"model_alias: {model_alias}, models: {models}")
    for item in models:
        if model_alias == item.model_group:
            is_model_alias_in_list = True

    if hidden:
        assert is_model_alias_in_list is False
    else:
        assert is_model_alias_in_list, f"models: {models}"


@pytest.mark.asyncio
async def test_proxy_model_group_info_rerank(prisma_client):
    """
    Check if rerank model is returned on the following endpoints

    `/v1/models`
    `/v1/model/info`
    `/v1/model_group/info`
    """
    import json

    from fastapi import HTTPException, Request, Response
    from starlette.datastructures import URL

    from litellm.proxy.proxy_server import model_group_info, model_info_v1, model_list

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    proxy_config = getattr(litellm.proxy.proxy_server, "proxy_config")

    _model_list = [
        {
            "model_name": "rerank-english-v3.0",
            "litellm_params": {"model": "cohere/rerank-english-v3.0"},
            "model_info": {
                "mode": "rerank",
            },
        }
    ]
    router = litellm.Router(model_list=_model_list)
    setattr(litellm.proxy.proxy_server, "llm_router", router)
    setattr(litellm.proxy.proxy_server, "llm_model_list", _model_list)

    request = Request(scope={"type": "http", "method": "POST", "headers": {}})
    request._url = URL(url="/v1/models")

    resp = await model_list(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
    )

    assert len(resp["data"]) == 1
    print(resp)

    resp = await model_info_v1(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
    )
    models = resp["data"]
    assert models[0]["model_info"]["mode"] == "rerank"
    resp = await model_group_info(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
    )

    print(resp)
    models = resp["data"]
    assert models[0].mode == "rerank"


# @pytest.mark.asyncio
# async def test_proxy_team_member_add(prisma_client):
#     """
#     Add 10 people to a team. Confirm all 10 are added.
#     """
#     from litellm.proxy.management_endpoints.team_endpoints import (
#         team_member_add,
#         new_team,
#     )
#     from litellm.proxy._types import TeamMemberAddRequest, Member, NewTeamRequest

#     setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     try:

#         async def test():
#             await litellm.proxy.proxy_server.prisma_client.connect()
#             from litellm.proxy.proxy_server import user_api_key_cache

#             user_api_key_dict = UserAPIKeyAuth(
#                 user_role=LitellmUserRoles.PROXY_ADMIN,
#                 api_key="sk-1234",
#                 user_id="1234",
#             )

#             new_team()
#             for _ in range(10):
#                 request = TeamMemberAddRequest(
#                     team_id="1234",
#                     member=Member(
#                         user_id="1234",
#                         user_role=LitellmUserRoles.INTERNAL_USER,
#                     ),
#                 )
#                 key = await team_member_add(
#                     request, user_api_key_dict=user_api_key_dict
#                 )

#             print(key)
#             user_id = key.user_id

#             # check /user/info to verify user_role was set correctly
#             new_user_info = await user_info(
#                 user_id=user_id, user_api_key_dict=user_api_key_dict
#             )
#             new_user_info = new_user_info.user_info
#             print("new_user_info=", new_user_info)
#             assert new_user_info["user_role"] == LitellmUserRoles.INTERNAL_USER
#             assert new_user_info["user_id"] == user_id

#             generated_key = key.key
#             bearer_token = "Bearer " + generated_key

#             assert generated_key not in user_api_key_cache.in_memory_cache.cache_dict

#             value_from_prisma = await prisma_client.get_data(
#                 token=generated_key,
#             )
#             print("token from prisma", value_from_prisma)

#             request = Request(
#                 {
#                     "type": "http",
#                     "route": api_route,
#                     "path": api_route.path,
#                     "headers": [("Authorization", bearer_token)],
#                 }
#             )

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#         asyncio.run(test())
#     except Exception as e:
#         pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.asyncio
async def test_proxy_server_prisma_setup():
    from litellm.proxy.proxy_server import ProxyStartupEvent, proxy_state
    from litellm.proxy.utils import ProxyLogging
    from litellm.caching import DualCache

    user_api_key_cache = DualCache()

    with patch.object(
        litellm.proxy.proxy_server, "PrismaClient", new=MagicMock()
    ) as mock_prisma_client:
        mock_client = mock_prisma_client.return_value  # This is the mocked instance
        mock_client.connect = AsyncMock()  # Mock the connect method
        mock_client.check_view_exists = AsyncMock()  # Mock the check_view_exists method
        mock_client.health_check = AsyncMock()  # Mock the health_check method
        mock_client._set_spend_logs_row_count_in_proxy_state = (
            AsyncMock()
        )  # Mock the _set_spend_logs_row_count_in_proxy_state method

        await ProxyStartupEvent._setup_prisma_client(
            database_url=os.getenv("DATABASE_URL"),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
            user_api_key_cache=user_api_key_cache,
        )

        # Verify our mocked methods were called
        mock_client.connect.assert_called_once()
        mock_client.check_view_exists.assert_called_once()

        # Note: This is REALLY IMPORTANT to check that the health check is called
        # This is how we ensure the DB is ready before proceeding
        mock_client.health_check.assert_called_once()

        # check that the spend logs row count is set in proxy state
        mock_client._set_spend_logs_row_count_in_proxy_state.assert_called_once()
        assert proxy_state.get_proxy_state_variable("spend_logs_row_count") is not None


@pytest.mark.asyncio
async def test_proxy_server_prisma_setup_invalid_db():
    """
    PROD TEST: Test that proxy server startup fails when it's unable to connect to the database

    Think 2-3 times before editing / deleting this test, it's important for PROD
    """
    from litellm.proxy.proxy_server import ProxyStartupEvent
    from litellm.proxy.utils import ProxyLogging
    from litellm.caching import DualCache

    user_api_key_cache = DualCache()
    invalid_db_url = "postgresql://invalid:invalid@localhost:5432/nonexistent"

    _old_db_url = os.getenv("DATABASE_URL")
    os.environ["DATABASE_URL"] = invalid_db_url

    with pytest.raises(Exception) as exc_info:
        await ProxyStartupEvent._setup_prisma_client(
            database_url=invalid_db_url,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
            user_api_key_cache=user_api_key_cache,
        )
        print("GOT EXCEPTION=", exc_info)

        assert "httpx.ConnectError" in str(exc_info.value)

    # # Verify the error message indicates a database connection issue
    # assert any(x in str(exc_info.value).lower() for x in ["database", "connection", "authentication"])

    if _old_db_url:
        os.environ["DATABASE_URL"] = _old_db_url


@pytest.mark.asyncio
async def test_get_ui_settings_spend_logs_threshold():
    """
    Test that get_ui_settings correctly sets DISABLE_EXPENSIVE_DB_QUERIES based on spend_logs_row_count threshold
    """
    from litellm.proxy.management_endpoints.ui_sso import get_ui_settings
    from litellm.proxy.proxy_server import proxy_state
    from fastapi import Request
    from litellm.constants import MAX_SPENDLOG_ROWS_TO_QUERY

    # Create a mock request
    mock_request = Request(
        scope={
            "type": "http",
            "headers": [],
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/sso/get/ui_settings",
            "query_string": b"",
        }
    )

    # Test case 1: When spend_logs_row_count > MAX_SPENDLOG_ROWS_TO_QUERY
    proxy_state.set_proxy_state_variable(
        "spend_logs_row_count", MAX_SPENDLOG_ROWS_TO_QUERY + 1
    )
    response = await get_ui_settings(mock_request)
    print("response from get_ui_settings", json.dumps(response, indent=4))
    assert response["DISABLE_EXPENSIVE_DB_QUERIES"] is True
    assert response["NUM_SPEND_LOGS_ROWS"] == MAX_SPENDLOG_ROWS_TO_QUERY + 1

    # Test case 2: When spend_logs_row_count < MAX_SPENDLOG_ROWS_TO_QUERY
    proxy_state.set_proxy_state_variable(
        "spend_logs_row_count", MAX_SPENDLOG_ROWS_TO_QUERY - 1
    )
    response = await get_ui_settings(mock_request)
    print("response from get_ui_settings", json.dumps(response, indent=4))
    assert response["DISABLE_EXPENSIVE_DB_QUERIES"] is False
    assert response["NUM_SPEND_LOGS_ROWS"] == MAX_SPENDLOG_ROWS_TO_QUERY - 1

    # Test case 3: Edge case - exactly MAX_SPENDLOG_ROWS_TO_QUERY
    proxy_state.set_proxy_state_variable(
        "spend_logs_row_count", MAX_SPENDLOG_ROWS_TO_QUERY
    )
    response = await get_ui_settings(mock_request)
    print("response from get_ui_settings", json.dumps(response, indent=4))
    assert response["DISABLE_EXPENSIVE_DB_QUERIES"] is False
    assert response["NUM_SPEND_LOGS_ROWS"] == MAX_SPENDLOG_ROWS_TO_QUERY

    # Clean up
    proxy_state.set_proxy_state_variable("spend_logs_row_count", 0)


@pytest.mark.asyncio
async def test_run_background_health_check_reflects_llm_model_list(monkeypatch):
    """
    Test that _run_background_health_check reflects changes to llm_model_list in each health check iteration.
    """
    import litellm.proxy.proxy_server as proxy_server
    import copy

    test_model_list_1 = [{"model_name": "model-a"}]
    test_model_list_2 = [{"model_name": "model-b"}]
    called_model_lists = []

    async def fake_perform_health_check(model_list, details):
        called_model_lists.append(copy.deepcopy(model_list))
        return (["healthy"], ["unhealthy"])

    monkeypatch.setattr(proxy_server, "health_check_interval", 1)
    monkeypatch.setattr(proxy_server, "health_check_details", None)
    monkeypatch.setattr(
        proxy_server, "llm_model_list", copy.deepcopy(test_model_list_1)
    )
    monkeypatch.setattr(proxy_server, "perform_health_check", fake_perform_health_check)
    monkeypatch.setattr(proxy_server, "health_check_results", {})

    async def fake_sleep(interval):
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    try:
        await proxy_server._run_background_health_check()
    except asyncio.CancelledError:
        pass

    monkeypatch.setattr(
        proxy_server, "llm_model_list", copy.deepcopy(test_model_list_2)
    )

    try:
        await proxy_server._run_background_health_check()
    except asyncio.CancelledError:
        pass

    assert len(called_model_lists) >= 2
    assert called_model_lists[0] == test_model_list_1
    assert called_model_lists[1] == test_model_list_2


@pytest.mark.asyncio
async def test_background_health_check_skip_disabled_models(monkeypatch):
    """Ensure models with disable_background_health_check are skipped."""
    import litellm.proxy.proxy_server as proxy_server
    import copy

    test_model_list = [
        {"model_name": "model-a"},
        {"model_name": "model-b", "model_info": {"disable_background_health_check": True}},
    ]
    called_model_lists = []

    async def fake_perform_health_check(model_list, details):
        called_model_lists.append(copy.deepcopy(model_list))
        return (["healthy"], [])

    monkeypatch.setattr(proxy_server, "health_check_interval", 1)
    monkeypatch.setattr(proxy_server, "health_check_details", None)
    monkeypatch.setattr(proxy_server, "llm_model_list", copy.deepcopy(test_model_list))
    monkeypatch.setattr(proxy_server, "perform_health_check", fake_perform_health_check)
    monkeypatch.setattr(proxy_server, "health_check_results", {})

    async def fake_sleep(interval):
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    try:
        await proxy_server._run_background_health_check()
    except asyncio.CancelledError:
        pass

    assert called_model_lists == [[{"model_name": "model-a"}]]


def test_get_timeout_from_request():
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    headers = {
        "x-litellm-timeout": "90",
    }
    timeout = LiteLLMProxyRequestSetup._get_timeout_from_request(headers)
    assert timeout == 90

    headers = {
        "x-litellm-timeout": "90.5",
    }
    timeout = LiteLLMProxyRequestSetup._get_timeout_from_request(headers)
    assert timeout == 90.5


@pytest.mark.parametrize(
    "ui_exists, ui_has_content",
    [
        (True, True),   # UI path exists and has content
        (True, False),  # UI path exists but is empty
        (False, False), # UI path doesn't exist
    ],
)
def test_non_root_ui_path_logic(monkeypatch, tmp_path, ui_exists, ui_has_content):
    """
    Test the non-root Docker UI path detection logic.
    
    Tests that when LITELLM_NON_ROOT is set to "true":
    - If UI path exists and has content, it should be used
    - If UI path doesn't exist or is empty, proper error logging occurs
    """
    import tempfile
    import shutil
    from unittest.mock import MagicMock
    
    # Create a temporary directory to act as /tmp/litellm_ui
    test_ui_path = tmp_path / "litellm_ui"
    
    if ui_exists:
        test_ui_path.mkdir(parents=True, exist_ok=True)
        if ui_has_content:
            # Create some dummy files to simulate built UI
            (test_ui_path / "index.html").write_text("<html></html>")
            (test_ui_path / "app.js").write_text("console.log('test');")
    
    # Mock the environment variable and os.path operations
    monkeypatch.setenv("LITELLM_NON_ROOT", "true")
    
    # Create a mock logger to capture log messages
    mock_logger = MagicMock()
    
    # We need to reimport or reload the relevant code section
    # Since this is module-level code, we'll test the logic directly
    ui_path = None
    non_root_ui_path = str(test_ui_path)
    
    # Simulate the logic from proxy_server.py lines 909-920
    if os.getenv("LITELLM_NON_ROOT", "").lower() == "true":
        if os.path.exists(non_root_ui_path) and os.listdir(non_root_ui_path):
            mock_logger.info(f"Using pre-built UI for non-root Docker: {non_root_ui_path}")
            mock_logger.info(f"UI files found: {len(os.listdir(non_root_ui_path))} items")
            ui_path = non_root_ui_path
        else:
            mock_logger.error(f"UI not found at {non_root_ui_path}. UI will not be available.")
            mock_logger.error(f"Path exists: {os.path.exists(non_root_ui_path)}, Has content: {os.path.exists(non_root_ui_path) and bool(os.listdir(non_root_ui_path))}")
    
    # Verify behavior based on test parameters
    if ui_exists and ui_has_content:
        # UI should be found and used
        assert ui_path == non_root_ui_path
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call(f"Using pre-built UI for non-root Docker: {non_root_ui_path}")
        # Verify the second info call mentions the number of items
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("UI files found:" in call and "items" in call for call in info_calls)
        assert mock_logger.error.call_count == 0
    else:
        # UI should not be found, error should be logged
        assert ui_path is None
        assert mock_logger.error.call_count == 2
        mock_logger.error.assert_any_call(f"UI not found at {non_root_ui_path}. UI will not be available.")
        # Verify the second error call has path existence info
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("Path exists:" in call for call in error_calls)
        assert mock_logger.info.call_count == 0


@pytest.mark.asyncio
async def test_update_config_success_callback_normalization():
    """
    Ensure success_callback values are normalized to lowercase when updating config.
    This prevents delete_callback (which searches lowercase) from failing on mixed case inputs like 'SQS'.
    """
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import ConfigYAML

    # Ensure feature is enabled and prisma_client is set
    setattr(proxy_server, "store_model_in_db", True)
    setattr(proxy_server, "proxy_logging_obj", MagicMock())

    class MockPrisma:
        def __init__(self):
            self.db = MagicMock()
            self.db.litellm_config = MagicMock()
            self.db.litellm_config.upsert = AsyncMock()

        # proxy_server.update_config expects this to be sync returning a dict
        def jsonify_object(self, obj):
            return obj

    setattr(proxy_server, "prisma_client", MockPrisma())

    class MockProxyConfig:
        def __init__(self):
            self.saved_config = None

        async def get_config(self):
            # Existing config has one lowercase callback already
            return {"litellm_settings": {"success_callback": ["langfuse"]}}

        async def save_config(self, new_config: dict):
            self.saved_config = new_config

        async def add_deployment(self, prisma_client=None, proxy_logging_obj=None):
            return None

    mock_proxy_config = MockProxyConfig()
    setattr(proxy_server, "proxy_config", mock_proxy_config)

    # Update config with mixed-case callbacks - expect normalization to lowercase
    config_update = ConfigYAML(litellm_settings={"success_callback": ["SQS", "sQs"]})
    await proxy_server.update_config(config_update)

    saved = mock_proxy_config.saved_config
    assert saved is not None, "save_config was not called"
    callbacks = saved["litellm_settings"]["success_callback"]

    # Deduped and normalized
    assert "sqs" in callbacks
    assert "SQS" not in callbacks
    assert "sQs" not in callbacks
    # Existing callback should still be present
    assert "langfuse" in callbacks
