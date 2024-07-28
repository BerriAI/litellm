import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds-the parent directory to the system path

import asyncio

import httpx

from litellm.proxy.proxy_server import app, initialize_pass_through_endpoints


# Mock the async_client used in the pass_through_request function
async def mock_request(*args, **kwargs):
    return httpx.Response(200, json={"message": "Mocked response"})


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_pass_through_endpoint(client, monkeypatch):
    # Mock the httpx.AsyncClient.request method
    monkeypatch.setattr("httpx.AsyncClient.request", mock_request)

    # Define a pass-through endpoint
    pass_through_endpoints = [
        {
            "path": "/test-endpoint",
            "target": "https://api.example.com/v1/chat/completions",
            "headers": {"Authorization": "Bearer test-token"},
        }
    ]

    # Initialize the pass-through endpoint
    await initialize_pass_through_endpoints(pass_through_endpoints)

    # Make a request to the pass-through endpoint
    response = client.post("/test-endpoint", json={"prompt": "Hello, world!"})

    # Assert the response
    assert response.status_code == 200
    assert response.json() == {"message": "Mocked response"}


@pytest.mark.asyncio
async def test_pass_through_endpoint_rerank(client):
    _cohere_api_key = os.environ.get("COHERE_API_KEY")

    # Define a pass-through endpoint
    pass_through_endpoints = [
        {
            "path": "/v1/rerank",
            "target": "https://api.cohere.com/v1/rerank",
            "headers": {"Authorization": f"bearer {_cohere_api_key}"},
        }
    ]

    # Initialize the pass-through endpoint
    await initialize_pass_through_endpoints(pass_through_endpoints)

    _json_data = {
        "model": "rerank-english-v3.0",
        "query": "What is the capital of the United States?",
        "top_n": 3,
        "documents": [
            "Carson City is the capital city of the American state of Nevada."
        ],
    }

    # Make a request to the pass-through endpoint
    response = client.post("/v1/rerank", json=_json_data)

    print("JSON response: ", _json_data)

    # Assert the response
    assert response.status_code == 200


@pytest.mark.parametrize(
    "auth, rpm_limit, expected_error_code",
    [(True, 0, 429), (True, 1, 200), (False, 0, 401)],
)
@pytest.mark.asyncio
async def test_pass_through_endpoint_rpm_limit(auth, expected_error_code, rpm_limit):
    client = TestClient(app)
    import litellm
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import ProxyLogging, hash_token, user_api_key_cache

    mock_api_key = "sk-my-test-key"
    cache_value = UserAPIKeyAuth(token=hash_token(mock_api_key), rpm_limit=rpm_limit)

    _cohere_api_key = os.environ.get("COHERE_API_KEY")

    user_api_key_cache.set_cache(key=hash_token(mock_api_key), value=cache_value)

    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
    proxy_logging_obj._init_litellm_callbacks()

    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "prisma_client", "FAKE-VAR")
    setattr(litellm.proxy.proxy_server, "proxy_logging_obj", proxy_logging_obj)

    # Define a pass-through endpoint
    pass_through_endpoints = [
        {
            "path": "/v1/rerank",
            "target": "https://api.cohere.com/v1/rerank",
            "auth": auth,
            "headers": {"Authorization": f"bearer {_cohere_api_key}"},
        }
    ]

    # Initialize the pass-through endpoint
    await initialize_pass_through_endpoints(pass_through_endpoints)

    _json_data = {
        "model": "rerank-english-v3.0",
        "query": "What is the capital of the United States?",
        "top_n": 3,
        "documents": [
            "Carson City is the capital city of the American state of Nevada."
        ],
    }

    # Make a request to the pass-through endpoint
    response = client.post(
        "/v1/rerank",
        json=_json_data,
        headers={"Authorization": "Bearer {}".format(mock_api_key)},
    )

    print("JSON response: ", _json_data)

    # Assert the response
    assert response.status_code == expected_error_code


@pytest.mark.asyncio
async def test_pass_through_endpoint_anthropic(client):
    import litellm
    from litellm import Router
    from litellm.adapters.anthropic_adapter import anthropic_adapter

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "mock_response": "Hey, how's it going?",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", router)

    # Define a pass-through endpoint
    pass_through_endpoints = [
        {
            "path": "/v1/test-messages",
            "target": anthropic_adapter,
            "headers": {"litellm_user_api_key": "my-test-header"},
        }
    ]

    # Initialize the pass-through endpoint
    await initialize_pass_through_endpoints(pass_through_endpoints)

    _json_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Who are you?"}],
    }

    # Make a request to the pass-through endpoint
    response = client.post(
        "/v1/test-messages", json=_json_data, headers={"my-test-header": "my-test-key"}
    )

    print("JSON response: ", _json_data)

    # Assert the response
    assert response.status_code == 200
