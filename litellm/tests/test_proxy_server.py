import sys, os
import traceback
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
    router,
    save_worker_config,
    initialize,
)  # Replace with the actual module where your FastAPI router is defined

# Your bearer token
token = "sk-1234"

headers = {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def client_no_auth():
    # Assuming litellm.proxy.proxy_server is an object
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
    # initialize can get run in parallel, it sets specific variables for the fast api app, sinc eit gets run in parallel different tests use the wrong variables
    asyncio.run(initialize(config=config_fp, debug=True))
    app = FastAPI()
    app.include_router(router)  # Include your router in the test app

    return TestClient(app)


def test_chat_completion(client_no_auth):
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
        print(f"response - {response.text}")
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


# Run the test


def test_chat_completion_azure(client_no_auth):
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

        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
        assert len(result["choices"][0]["message"]["content"]) > 0
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


# Run the test
# test_chat_completion_azure()


### EMBEDDING
def test_embedding(client_no_auth):
    global headers
    from litellm.proxy.proxy_server import user_custom_auth

    try:
        test_data = {
            "model": "azure/azure-embedding-model",
            "input": ["good morning from litellm"],
        }

        response = client_no_auth.post("/v1/embeddings", json=test_data)

        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10  # this usually has len==1536 so
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


def test_bedrock_embedding(client_no_auth):
    global headers
    from litellm.proxy.proxy_server import user_custom_auth

    try:
        test_data = {
            "model": "amazon-embeddings",
            "input": ["good morning from litellm"],
        }

        response = client_no_auth.post("/v1/embeddings", json=test_data)

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


def test_img_gen(client_no_auth):
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

        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["url"]))
        assert len(result["data"][0]["url"]) > 10
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


#### ADDITIONAL
# @pytest.mark.skip(reason="hitting yaml load issues on circle-ci")
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


def test_chat_completion_optional_params(client_no_auth):
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
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)


# Run the test
# test_chat_completion_optional_params()

# Test Reading config.yaml file
from litellm.proxy.proxy_server import ProxyConfig


def test_load_router_config():
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
        pytest.fail("Proxy: Got exception reading config", e)


# test_load_router_config()
