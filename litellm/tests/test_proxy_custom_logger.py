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
import importlib, inspect

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import router, save_worker_config, initialize  # Replace with the actual module where your FastAPI router is defined
filepath = os.path.dirname(os.path.abspath(__file__))
config_fp = f"{filepath}/test_configs/test_custom_logger.yaml"
python_file_path = f"{filepath}/test_configs/custom_callbacks.py"
save_worker_config(config=config_fp, model=None, alias=None, api_base=None, api_version=None, debug=False, temperature=None, max_tokens=None, request_timeout=600, max_budget=None, telemetry=False, drop_params=True, add_function_to_prompt=False, headers=None, save=False, use_queue=False)
app = FastAPI()
app.include_router(router)  # Include your router in the test app
@app.on_event("startup")
async def wrapper_startup_event():
    initialize(config=config_fp, model=None, alias=None, api_base=None, api_version=None, debug=True, temperature=None, max_tokens=None, request_timeout=600, max_budget=None, telemetry=False, drop_params=True, add_function_to_prompt=False, headers=None, save=False, use_queue=False)

# Here you create a fixture that will be used by your tests
# Make sure the fixture returns TestClient(app)
@pytest.fixture(autouse=True)
def client():
    with TestClient(app) as client:
        yield client

# Your bearer token
token = os.getenv("PROXY_MASTER_KEY")

headers = {
    "Authorization": f"Bearer {token}"
}


def test_chat_completion(client):
    try:
         # Your test data
        print("initialized proxy")
        # import the initialized custom logger
        print(litellm.callbacks)

        assert len(litellm.callbacks) == 1 # assert litellm is initialized with 1 callback
        my_custom_logger = litellm.callbacks[0]
        assert my_custom_logger.async_success == False

        test_data = {
            "model": "Azure OpenAI GPT-4 Canada",
            "messages": [
                {
                    "role": "user",
                    "content": "hi"
                },
            ],
            "max_tokens": 10,
        }


        response = client.post("/chat/completions", json=test_data, headers=headers)
        print("made request", response.status_code, response.text)
        assert my_custom_logger.async_success == True                   # checks if the status of async_success is True, only the async_log_success_event can set this to true
        assert my_custom_logger.async_completion_kwargs["model"] == "chatgpt-v-2" # checks if kwargs passed to async_log_success_event are correct
        
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)



def test_embedding(client):
    try:
         # Your test data
        print("initialized proxy")
        # import the initialized custom logger
        print(litellm.callbacks)

        assert len(litellm.callbacks) == 1 # assert litellm is initialized with 1 callback
        my_custom_logger = litellm.callbacks[0]
        assert my_custom_logger.async_success_embedding == False

        test_data = {
            "model": "azure-embedding-model",
            "input": ["hello"]
        }
        response = client.post("/embeddings", json=test_data, headers=headers)
        print("made request", response.status_code, response.text)
        assert my_custom_logger.async_success_embedding == True                             # checks if the status of async_success is True, only the async_log_success_event can set this to true
        assert my_custom_logger.async_embedding_kwargs["model"] == "azure-embedding-model"  # checks if kwargs passed to async_log_success_event are correct
        
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)