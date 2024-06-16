import sys, os
import traceback
from unittest import mock
from dotenv import load_dotenv

load_dotenv()
import os, io

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the desired logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import (
    app,
    save_worker_config,
    initialize,
)  # Replace with the actual module where your FastAPI router is defined

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
    import time
    from litellm._logging import verbose_logger, verbose_proxy_logger
    import logging

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
