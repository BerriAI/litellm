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



def test_chat_completion(client):
    try:
         # Your test data
        print("initialized proxy")
        # import the initialized custom logger
        my_custom_logger = importlib.util.spec_from_file_location("my_custom_logger", python_file_path)
        print("my_custom_logger", my_custom_logger)

        blue_color_code = "\033[94m"
        reset_color_code = "\033[0m"
        print(f"{blue_color_code}Initialized LiteLLM custom logger")
        try:
            print(f"Logger Initialized with following methods:")
            methods = [method for method in dir(my_custom_logger) if inspect.ismethod(getattr(my_custom_logger, method))]
            
            # Pretty print the methods
            for method in methods:
                print(f" - {method}")
            print(f"{reset_color_code}")
        except:
            pass

        for attribute in dir(my_custom_logger):
            print(f"{attribute}: {getattr(my_custom_logger, attribute)}")
        test_data = {
            "model": "litellm-test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "hi"
                },
            ],
            "max_tokens": 10,
        }


        response = client.post("/chat/completions", json=test_data)
        print("made request", response.status_code, response.text)
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)