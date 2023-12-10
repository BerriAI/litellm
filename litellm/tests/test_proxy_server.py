import sys, os
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
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the desired logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import router, save_worker_config, initialize  # Replace with the actual module where your FastAPI router is defined
filepath = os.path.dirname(os.path.abspath(__file__))
config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
save_worker_config(config=config_fp, model=None, alias=None, api_base=None, api_version=None, debug=False, temperature=None, max_tokens=None, request_timeout=600, max_budget=None, telemetry=False, drop_params=True, add_function_to_prompt=False, headers=None, save=False, use_queue=False)
app = FastAPI()
app.include_router(router)  # Include your router in the test app
@app.on_event("startup")
async def wrapper_startup_event():
    initialize(config=config_fp)

# Your bearer token
token = os.getenv("PROXY_MASTER_KEY")

headers = {
    "Authorization": f"Bearer {token}"
}
    
# Here you create a fixture that will be used by your tests
# Make sure the fixture returns TestClient(app)
@pytest.fixture(autouse=True)
def client():
    with TestClient(app) as client:
        yield client

def test_chat_completion(client):
    global headers
    try:
        # Your test data
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "hi"
                },
            ],
            "max_tokens": 10,
        }
        
        print("testing proxy server")
        response = client.post("/v1/chat/completions", json=test_data, headers=headers)
        print(f"response - {response.text}")
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")

# Run the test

def test_chat_completion_azure(client):
    global headers
    try:
        # Your test data
        test_data = {
            "model": "azure/chatgpt-v-2",
            "messages": [
                {
                    "role": "user",
                    "content": "write 1 sentence poem"
                },
            ],
            "max_tokens": 10,
        }
        
        print("testing proxy server with Azure Request")
        response = client.post("/v1/chat/completions", json=test_data, headers=headers)

        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
        assert len(result["choices"][0]["message"]["content"]) > 0 
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")

# Run the test
# test_chat_completion_azure()


def test_embedding(client):
    global headers
    try:
        test_data = {
            "model": "azure/azure-embedding-model",
            "input": ["good morning from litellm"],
        }
        print("testing proxy server with OpenAI embedding")
        response = client.post("/v1/embeddings", json=test_data, headers=headers)

        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10 # this usually has len==1536 so
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")

# Run the test
# test_embedding()

# @pytest.mark.skip(reason="hitting yaml load issues on circle-ci")
def test_add_new_model(client): 
    global headers
    try: 
        test_data = {
            "model_name": "test_openai_models",
            "litellm_params": {
                "model": "gpt-3.5-turbo", 
            },
            "model_info": {
                "description": "this is a test openai model"
            }
        }
        client.post("/model/new", json=test_data, headers=headers)
        response = client.get("/model/info", headers=headers)
        assert response.status_code == 200
        result = response.json() 
        print(f"response: {result}")
        model_info = None
        for m in result["data"]:
            if m["model_name"] == "test_openai_models":
                model_info = m["model_info"]
        assert model_info["description"] == "this is a test openai model"
    except Exception as e: 
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")

# test_add_new_model()

from litellm.integrations.custom_logger import CustomLogger
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")

    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")
        assert kwargs["user"] == "proxy-user"
        assert kwargs["model"] == "gpt-3.5-turbo"
        assert kwargs["max_tokens"] == 10

customHandler = MyCustomHandler()


def test_chat_completion_optional_params(client):
    # [PROXY: PROD TEST] - DO NOT DELETE
    # This tests if all the /chat/completion params are passed to litellm

    try:
        # Your test data
        litellm.set_verbose=True
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "hi"
                },
            ],
            "max_tokens": 10,
            "user": "proxy-user"
        }
        
        litellm.callbacks = [customHandler]
        print("testing proxy server: optional params")
        response = client.post("/v1/chat/completions", json=test_data, headers=headers)
        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)

# Run the test
# test_chat_completion_optional_params()

# Test Reading config.yaml file 
from litellm.proxy.proxy_server import load_router_config

def test_load_router_config():
    try:
        print("testing reading config")
        # this is a basic config.yaml with only a model
        filepath = os.path.dirname(os.path.abspath(__file__))
        result = load_router_config(router=None, config_file_path=f"{filepath}/example_config_yaml/simple_config.yaml")
        print(result)
        assert len(result[1]) == 1

        # this is a load balancing config yaml
        result = load_router_config(router=None, config_file_path=f"{filepath}/example_config_yaml/azure_config.yaml")
        print(result)
        assert len(result[1]) == 2

        # config with general settings - custom callbacks
        result = load_router_config(router=None, config_file_path=f"{filepath}/example_config_yaml/azure_config.yaml")
        print(result)
        assert len(result[1]) == 2

    except Exception as e:
        pytest.fail("Proxy: Got exception reading config", e)
# test_load_router_config()
