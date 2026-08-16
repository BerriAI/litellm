#### What this tests ####
#    This tests using caching w/ litellm which requires SSL=True
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
    config_fp = f"{filepath}/test_configs/test_cloudflare_azure_with_cache_config.yaml"
    # initialize can get run in parallel, it sets specific variables for the fast api app, sinc eit gets run in parallel different tests use the wrong variables
    asyncio.run(initialize(config=config_fp, debug=True))
    app = FastAPI()
    app.include_router(router)  # Include your router in the test app

    return TestClient(app)


def generate_random_word(length=4):
    import string, random

    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


@pytest.mark.skip(reason="AWS Suspended Account")
def test_chat_completion(client_no_auth):
    global headers
    try:
        user_message = f"Write a poem about {generate_random_word()}"
        messages = [{"content": user_message, "role": "user"}]
        # Your test data
        test_data = {
            "model": "azure-cloudflare",
            "messages": messages,
            "max_tokens": 10,
        }

        print("testing proxy server with chat completions")
        response = client_no_auth.post("/v1/chat/completions", json=test_data)
        print(f"response - {response.text}")
        assert response.status_code == 200

        response = response.json()
        print(response)

        content = response["choices"][0]["message"]["content"]
        response1_id = response["id"]

        print("\n content", content)

        assert len(content) > 1

        print("\nmaking 2nd request to proxy. Testing caching + non streaming")
        response = client_no_auth.post("/v1/chat/completions", json=test_data)
        print(f"response - {response.text}")
        assert response.status_code == 200

        response = response.json()
        print(response)
        response2_id = response["id"]
        assert response1_id == response2_id
        litellm.disable_cache()

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")
