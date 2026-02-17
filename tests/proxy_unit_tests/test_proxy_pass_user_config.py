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
import os
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


@pytest.mark.skipif(
    os.environ.get("AZURE_API_KEY") is None or os.environ.get("OPENAI_API_KEY") is None,
    reason="AZURE_API_KEY or OPENAI_API_KEY not set - skipping integration test"
)
def test_chat_completion(client_no_auth):
    global headers

    from litellm.types.router import RouterConfig, ModelConfig
    from litellm.types.completion import CompletionRequest

    user_config = RouterConfig(
        model_list=[
            ModelConfig(
                model_name="user-azure-instance",
                litellm_params=CompletionRequest(
                    model="azure/gpt-4.1-mini",
                    api_key=os.getenv("AZURE_API_KEY"),
                    api_version=os.getenv("AZURE_API_VERSION"),
                    api_base=os.getenv("AZURE_API_BASE"),
                    timeout=10,
                ),
                tpm=240000,
                rpm=1800,
            ),
            ModelConfig(
                model_name="user-openai-instance",
                litellm_params=CompletionRequest(
                    model="gpt-3.5-turbo",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=10,
                ),
                tpm=240000,
                rpm=1800,
            ),
        ],
        num_retries=2,
        allowed_fails=3,
        fallbacks=[{"user-azure-instance": ["user-openai-instance"]}],
    ).dict()

    try:
        # Your test data
        test_data = {
            "model": "user-azure-instance",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
            "user_config": user_config,
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
