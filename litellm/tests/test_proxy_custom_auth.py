import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path 
import pytest
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import router, save_worker_config, startup_event  # Replace with the actual module where your FastAPI router is defined
filepath = os.path.dirname(os.path.abspath(__file__))
config_fp = f"{filepath}/test_configs/test_config_custom_auth.yaml"
save_worker_config(config=config_fp, model=None, alias=None, api_base=None, api_version=None, debug=False, temperature=None, max_tokens=None, request_timeout=600, max_budget=None, telemetry=False, drop_params=True, add_function_to_prompt=False, headers=None, save=False, use_queue=False)
app = FastAPI()
app.include_router(router)  # Include your router in the test app
@app.on_event("startup")
async def wrapper_startup_event():
    await startup_event()

# Here you create a fixture that will be used by your tests
# Make sure the fixture returns TestClient(app)
@pytest.fixture(autouse=True)
def client():
    with TestClient(app) as client:
        yield client

def test_custom_auth(client):
    try:
         # Your test data
        test_data = {
            "model": "openai-model",
            "messages": [
                {
                    "role": "user",
                    "content": "hi"
                },
            ],
            "max_tokens": 10,
        }
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = client.post("/chat/completions", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 401
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)