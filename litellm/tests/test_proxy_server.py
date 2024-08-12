import os
import sys
import traceback
from unittest import mock

from dotenv import load_dotenv

import litellm.proxy
import litellm.proxy.proxy_server

load_dotenv()
import io
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
        )
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


from litellm.tests.test_custom_callback_input import CompletionCustomHandler


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
            "model": "azure/chatgpt-v-2",
            "messages": [
                {"role": "user", "content": "write 1 sentence poem"},
            ],
            "max_tokens": 10,
        }

        print("testing proxy server with Azure Request /chat/completions")
        response = client_no_auth.post("/v1/chat/completions", json=test_data)

        mock_acompletion.assert_called_once_with(
            model="azure/chatgpt-v-2",
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
            "model": "azure/chatgpt-v-2",
            "messages": [
                {"role": "user", "content": "write 1 sentence poem"},
            ],
            "max_tokens": 10,
        }

        url = "/openai/deployments/azure/chatgpt-v-2/chat/completions"
        print(f"testing proxy server with Azure Request {url}")
        response = client_no_auth.post(url, json=test_data)

        mock_acompletion.assert_called_once_with(
            model="azure/chatgpt-v-2",
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
            "model": "azure/azure-embedding-model",
            "input": ["good morning from litellm"],
        }

        response = client_no_auth.post("/v1/embeddings", json=test_data)

        mock_aembedding.assert_called_once_with(
            model="azure/azure-embedding-model",
            input=["good morning from litellm"],
            specific_deployment=True,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10  # this usually has len==1536 so
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
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
        )
        assert response.status_code == 200
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
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.auth.auth_checks import _cache_team_object

    proxy_logging_obj: ProxyLogging = getattr(
        litellm.proxy.proxy_server, "proxy_logging_obj"
    )

    proxy_logging_obj.internal_usage_cache.redis_cache = RedisCache()

    with patch.object(
        proxy_logging_obj.internal_usage_cache.redis_cache,
        "async_set_cache",
        new=AsyncMock(),
    ) as mock_client:
        await _cache_team_object(
            team_id="1234",
            team_table=LiteLLM_TeamTableCachedObj(),
            user_api_key_cache=DualCache(),
            proxy_logging_obj=proxy_logging_obj,
        )

        mock_client.assert_called()


@pytest.mark.asyncio
async def test_get_team_redis(client_no_auth):
    """
    Tests if get_team_object gets value from redis cache, if set
    """
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy.auth.auth_checks import _cache_team_object, get_team_object

    proxy_logging_obj: ProxyLogging = getattr(
        litellm.proxy.proxy_server, "proxy_logging_obj"
    )

    proxy_logging_obj.internal_usage_cache.redis_cache = RedisCache()

    with patch.object(
        proxy_logging_obj.internal_usage_cache.redis_cache,
        "async_get_cache",
        new=AsyncMock(),
    ) as mock_client:
        try:
            await get_team_object(
                team_id="1234",
                user_api_key_cache=DualCache(),
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
                prisma_client=AsyncMock(),
            )
        except Exception as e:
            pass

        mock_client.assert_called_once()


import random
import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from litellm.proxy._types import (
    LitellmUserRoles,
    NewUserRequest,
    TeamMemberAddRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.proxy.management_endpoints.team_endpoints import team_member_add
from litellm.tests.test_key_generate_prisma import prisma_client


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

    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
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
    ) as mock_litellm_usertable:
        mock_client = AsyncMock()
        mock_litellm_usertable.upsert = mock_client
        mock_litellm_usertable.find_many = AsyncMock(return_value=None)

        await team_member_add(
            data=team_member_add_request,
            user_api_key_dict=UserAPIKeyAuth(),
            http_request=Request(
                scope={"type": "http", "path": "/user/new"},
            ),
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
