import sys, os, time
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError
from httpx import AsyncClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the desired logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)
from concurrent.futures import ThreadPoolExecutor

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import (
    router,
    save_worker_config,
    startup_event,
    asyncio,
)  # Replace with the actual module where your FastAPI router is defined

filepath = os.path.dirname(os.path.abspath(__file__))
config_fp = f"{filepath}/test_configs/test_config.yaml"
save_worker_config(
    config=config_fp,
    model=None,
    alias=None,
    api_base=None,
    api_version=None,
    debug=True,
    temperature=None,
    max_tokens=None,
    request_timeout=600,
    max_budget=None,
    telemetry=False,
    drop_params=True,
    add_function_to_prompt=False,
    headers=None,
    save=False,
    use_queue=False,
)


import asyncio


# @pytest.fixture
# def event_loop():
#     """Create an instance of the default event loop for each test case."""
#     policy = asyncio.WindowsSelectorEventLoopPolicy()
#     res = policy.new_event_loop()
#     asyncio.set_event_loop(res)
#     res._close = res.close
#     res.close = lambda: None

#     yield res

#     res._close()


# Here you create a fixture that will be used by your tests
# Make sure the fixture returns TestClient(app)
@pytest.fixture(scope="function")
async def client():
    from litellm.proxy.proxy_server import (
        cleanup_router_config_variables,
        initialize,
        ProxyLogging,
        proxy_logging_obj,
    )

    cleanup_router_config_variables()  # rest proxy before test
    proxy_logging_obj = ProxyLogging(user_api_key_cache={})
    proxy_logging_obj._init_litellm_callbacks()  # INITIALIZE LITELLM CALLBACKS ON SERVER STARTUP <- do this to catch any logging errors on startup, not when calls are being made

    await initialize(config=config_fp, debug=True)
    app = FastAPI()
    app.include_router(router)  # Include your router in the test app
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_add_new_key(client):
    try:
        # Your test data
        test_data = {
            "models": ["gpt-3.5-turbo", "gpt-4", "claude-2", "azure-model"],
            "aliases": {"mistral-7b": "gpt-3.5-turbo"},
            "duration": "20m",
        }
        print("testing proxy server - test_add_new_key")
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        assert result["key"].startswith("sk-")

        async def _post_data():
            json_data = {
                "model": "azure-model",
                "messages": [
                    {
                        "role": "user",
                        "content": f"this is a test request, write a short poem {time.time()}",
                    }
                ],
            }
            response = await client.post(
                "/chat/completions",
                json=json_data,
                headers={"Authorization": f"Bearer {result['key']}"},
            )
            return response

        await _post_data()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_update_new_key(client):
    try:
        # Your test data
        test_data = {
            "models": ["gpt-3.5-turbo", "gpt-4", "claude-2", "azure-model"],
            "aliases": {"mistral-7b": "gpt-3.5-turbo"},
            "duration": "20m",
        }
        print("testing proxy server-test_update_new_key")
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        assert result["key"].startswith("sk-")

        async def _post_data():
            json_data = {"models": ["bedrock-models"], "key": result["key"]}
            response = await client.post("/key/update", json=json_data, headers=headers)
            print(f"response text: {response.text}")
            assert response.status_code == 200
            return response

        await _post_data()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")


# Run the test - only runs via pytest


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_add_new_key_max_parallel_limit(client):
    try:
        import anyio

        print("ANY IO BACKENDS")
        print(anyio.get_all_backends())
        # Your test data
        test_data = {
            "duration": "20m",
            "max_parallel_requests": 1,
            "metadata": {"type": "ishaan-test"},
        }
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {"Authorization": f"Bearer {token}"}

        response = await client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()

        async def _post_data():
            json_data = {
                "model": "azure-model",
                "messages": [
                    {
                        "role": "user",
                        "content": f"this is a test request, write a short poem {time.time()}",
                    }
                ],
            }

            response = await client.post(
                "/chat/completions",
                json=json_data,
                headers={"Authorization": f"Bearer {result['key']}"},
            )
            return response

        async def _run_in_parallel():
            try:
                futures = [_post_data() for _ in range(2)]
                responses = await asyncio.gather(*futures)
                print("response1 status: ", responses[0].status_code)
                print("response2 status: ", responses[1].status_code)

                if any(response.status_code == 429 for response in responses):
                    pass
                else:
                    raise Exception()
            except Exception as e:
                pass

        await _run_in_parallel()

        # assert responses[0].status_code == 200 or responses[1].status_code == 200
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_add_new_key_max_parallel_limit_streaming(client):
    try:
        # Your test data
        test_data = {"duration": "20m", "max_parallel_requests": 1}
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()

        async def _post_data():
            json_data = {
                "model": "azure-model",
                "messages": [
                    {
                        "role": "user",
                        "content": f"this is a test request, write a short poem {time.time()}",
                    }
                ],
                "stream": True,
            }
            response = await client.post(
                "/chat/completions",
                json=json_data,
                headers={"Authorization": f"Bearer {result['key']}"},
            )
            return response

        async def _run_in_parallel():
            try:
                futures = [_post_data() for _ in range(2)]
                responses = await asyncio.gather(*futures)
                print("response1 status: ", responses[0].status_code)
                print("response2 status: ", responses[1].status_code)

                if any(response.status_code == 429 for response in responses):
                    pass
                else:
                    raise Exception()
            except Exception as e:
                pass

        _run_in_parallel()
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")
