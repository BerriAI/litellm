import os
import sys
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds-the parent directory to the system path

import asyncio
from unittest.mock import Mock

import httpx

from litellm.proxy.proxy_server import app, initialize_pass_through_endpoints


# Mock the async_client used in the pass_through_request function
async def mock_request(*args, **kwargs):
    mock_response = httpx.Response(200, json={"message": "Mocked response"})
    mock_response.request = Mock(spec=httpx.Request)
    return mock_response


def remove_rerank_route():
    for route in app.routes:
        if route.path == "/v1/rerank" and "POST" in route.methods:
            app.routes.remove(route)
            print("Rerank route removed successfully")
    print("ALL Routes on app=", app.routes)


@pytest.fixture
def client():
    remove_rerank_route()  # remove the native rerank route on the litellm proxy - since we're testing the pass through endpoints
    return TestClient(app)


@pytest.mark.asyncio
async def test_pass_through_endpoint(client, monkeypatch):
    # Mock the httpx.AsyncClient.request method
    monkeypatch.setattr("httpx.AsyncClient.request", mock_request)
    import litellm

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
    general_settings: Optional[dict] = (
        getattr(litellm.proxy.proxy_server, "general_settings", {}) or {}
    )
    general_settings.update({"pass_through_endpoints": pass_through_endpoints})
    setattr(litellm.proxy.proxy_server, "general_settings", general_settings)

    # Make a request to the pass-through endpoint
    response = client.post("/test-endpoint", json={"prompt": "Hello, world!"})

    # Assert the response
    assert response.status_code == 200
    assert response.json() == {"message": "Mocked response"}


@pytest.mark.asyncio
async def test_pass_through_endpoint_rerank(client):
    _cohere_api_key = os.environ.get("COHERE_API_KEY")
    import litellm

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
    general_settings: Optional[dict] = (
        getattr(litellm.proxy.proxy_server, "general_settings", {}) or {}
    )
    general_settings.update({"pass_through_endpoints": pass_through_endpoints})
    setattr(litellm.proxy.proxy_server, "general_settings", general_settings)

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
    [(True, 0, 429), (True, 1, 200), (False, 0, 200)],
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
    general_settings: Optional[dict] = (
        getattr(litellm.proxy.proxy_server, "general_settings", {}) or {}
    )
    general_settings.update({"pass_through_endpoints": pass_through_endpoints})
    setattr(litellm.proxy.proxy_server, "general_settings", general_settings)

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


@pytest.mark.parametrize(
    "auth, rpm_limit, expected_error_code",
    [(True, 0, 429), (True, 1, 207), (False, 0, 207)],
)
@pytest.mark.asyncio
async def test_aaapass_through_endpoint_pass_through_keys_langfuse(
    auth, expected_error_code, rpm_limit
):

    client = TestClient(app)
    import litellm
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import ProxyLogging, hash_token, user_api_key_cache

    # Store original values
    original_user_api_key_cache = getattr(
        litellm.proxy.proxy_server, "user_api_key_cache", None
    )
    original_master_key = getattr(litellm.proxy.proxy_server, "master_key", None)
    original_prisma_client = getattr(litellm.proxy.proxy_server, "prisma_client", None)
    original_proxy_logging_obj = getattr(
        litellm.proxy.proxy_server, "proxy_logging_obj", None
    )

    try:

        mock_api_key = "sk-my-test-key"
        cache_value = UserAPIKeyAuth(
            token=hash_token(mock_api_key), rpm_limit=rpm_limit
        )

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
                "path": "/api/public/ingestion",
                "target": "https://cloud.langfuse.com/api/public/ingestion",
                "auth": auth,
                "custom_auth_parser": "langfuse",
                "headers": {
                    "LANGFUSE_PUBLIC_KEY": "os.environ/LANGFUSE_PUBLIC_KEY",
                    "LANGFUSE_SECRET_KEY": "os.environ/LANGFUSE_SECRET_KEY",
                },
            }
        ]

        # Initialize the pass-through endpoint
        await initialize_pass_through_endpoints(pass_through_endpoints)
        general_settings: Optional[dict] = (
            getattr(litellm.proxy.proxy_server, "general_settings", {}) or {}
        )
        old_general_settings = general_settings
        general_settings.update({"pass_through_endpoints": pass_through_endpoints})
        setattr(litellm.proxy.proxy_server, "general_settings", general_settings)

        _json_data = {
            "batch": [
                {
                    "id": "80e2141f-0ca6-47b7-9c06-dde5e97de690",
                    "type": "trace-create",
                    "body": {
                        "id": "0687af7b-4a75-4de8-a4f6-cba1cdc00865",
                        "timestamp": "2024-08-14T02:38:56.092950Z",
                        "name": "test-trace-litellm-proxy-passthrough",
                    },
                    "timestamp": "2024-08-14T02:38:56.093352Z",
                }
            ],
            "metadata": {
                "batch_size": 1,
                "sdk_integration": "default",
                "sdk_name": "python",
                "sdk_version": "2.27.0",
                "public_key": "anything",
            },
        }

        # Make a request to the pass-through endpoint
        response = client.post(
            "/api/public/ingestion",
            json=_json_data,
            headers={"Authorization": "Basic c2stbXktdGVzdC1rZXk6YW55dGhpbmc="},
        )

        print("JSON response: ", _json_data)

        print("RESPONSE RECEIVED - {}".format(response.text))

        # Assert the response
        assert response.status_code == expected_error_code

        setattr(litellm.proxy.proxy_server, "general_settings", old_general_settings)
    finally:
        # Reset to original values
        setattr(
            litellm.proxy.proxy_server,
            "user_api_key_cache",
            original_user_api_key_cache,
        )
        setattr(litellm.proxy.proxy_server, "master_key", original_master_key)
        setattr(litellm.proxy.proxy_server, "prisma_client", original_prisma_client)
        setattr(
            litellm.proxy.proxy_server, "proxy_logging_obj", original_proxy_logging_obj
        )


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
    general_settings: Optional[dict] = (
        getattr(litellm.proxy.proxy_server, "general_settings", {}) or {}
    )
    general_settings.update({"pass_through_endpoints": pass_through_endpoints})
    setattr(litellm.proxy.proxy_server, "general_settings", general_settings)

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
