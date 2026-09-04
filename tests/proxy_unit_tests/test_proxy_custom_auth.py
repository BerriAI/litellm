import os
import traceback

from dotenv import load_dotenv

load_dotenv()
import io

# this file is to test litellm/proxy

import asyncio

import pytest
from fastapi import FastAPI

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.proxy.proxy_server import (  # Replace with the actual module where your FastAPI router is defined
    ProxyConfig,
    initialize,
    router,
    save_worker_config,
)


# Here you create a fixture that will be used by your tests
# Make sure the fixture returns TestClient(app)
@pytest.fixture(scope="function")
def client():
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_config_custom_auth.yaml"
    # initialize can get run in parallel, it sets specific variables for the fast api app, sinc eit gets run in parallel different tests use the wrong variables
    app = FastAPI()
    asyncio.run(initialize(config=config_fp))

    app.include_router(router)  # Include your router in the test app
    return TestClient(app)


def test_custom_auth(client):
    # Your test data
    test_data = {
        "model": "openai-model",
        "messages": [
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 10,
    }
    # Your bearer token
    token = os.getenv("PROXY_MASTER_KEY")
    print(f"token: {token}")
    headers = {"Authorization": f"Bearer {token}"}
    with pytest.raises(Exception, match="Authentication Error, Failed custom auth") as exc_info:
        client.post("/chat/completions", json=test_data, headers=headers)
    assert exc_info.value.code == "401"


def test_custom_auth_bearer(client):
    # Your test data
    test_data = {
        "model": "openai-model",
        "messages": [
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 10,
    }
    # Your bearer token
    token = os.getenv("PROXY_MASTER_KEY")

    headers = {"Authorization": f"WITHOUT BEAR Er  {token}"}
    with pytest.raises(Exception, match="CustomAuth - Malformed API Key passed in") as exc_info:
        client.post("/chat/completions", json=test_data, headers=headers)
    assert exc_info.value.code == "401"
    assert (
        exc_info.value.message
        == "Authentication Error, CustomAuth - Malformed API Key passed in. Ensure Key has `Bearer` prefix"
    )
