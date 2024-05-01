import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, asyncio
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import (
    router,
    save_worker_config,
    initialize,
    ProxyConfig,
)  # Replace with the actual module where your FastAPI router is defined


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
    try:
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
        response = client.post("/chat/completions", json=test_data, headers=headers)
        pytest.fail("LiteLLM Proxy test failed. This request should have been rejected")
    except Exception as e:
        print(vars(e))
        print("got an exception")
        assert e.code == 401
        assert e.message == "Authentication Error, Failed custom auth"
        pass


def test_custom_auth_bearer(client):
    try:
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
        response = client.post("/chat/completions", json=test_data, headers=headers)
        pytest.fail("LiteLLM Proxy test failed. This request should have been rejected")
    except Exception as e:
        print(vars(e))
        print("got an exception")
        assert e.code == 401
        assert (
            e.message
            == "Authentication Error, CustomAuth - Malformed API Key passed in. Ensure Key has `Bearer` prefix"
        )
        pass
