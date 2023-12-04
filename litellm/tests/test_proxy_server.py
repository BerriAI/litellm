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
from litellm.proxy.proxy_server import router  # Replace with the actual module where your FastAPI router is defined
app = FastAPI()
app.include_router(router)  # Include your router in the test app
client = TestClient(app)
def test_chat_completion():
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
        response = client.post("/v1/chat/completions", json=test_data)

        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)

# Run the test
# test_chat_completion()


def test_chat_completion_azure():
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
        response = client.post("/v1/chat/completions", json=test_data)

        assert response.status_code == 200
        result = response.json()
        print(f"Received response: {result}")
        assert len(result["choices"][0]["message"]["content"]) > 0 
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)

# Run the test
# test_chat_completion_azure()


def test_embedding():
    try:
        test_data = {
            "model": "azure/azure-embedding-model",
            "input": ["good morning from litellm"],
        }
        print("testing proxy server with OpenAI embedding")
        response = client.post("/v1/embeddings", json=test_data)

        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10 # this usually has len==1536 so
    except Exception as e:
        pytest.fail("LiteLLM Proxy test failed. Exception", e)

# Run the test
# test_embedding()


def test_add_new_model(): 
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
        client.post("/model/new", json=test_data)
        response = client.get("/model/info")
        assert response.status_code == 200
        result = response.json() 
        print(f"response: {result}")
        model_info = None
        for m in result["data"]:
            if m["id"]["model_name"] == "test_openai_models":
                model_info = m["id"]["model_info"]
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


def test_chat_completion_optional_params():
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
        response = client.post("/v1/chat/completions", json=test_data)
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
        result = load_router_config(router=None, config_file_path="../proxy/example_config_yaml/simple_config.yaml")
        print(result)
        assert len(result[1]) == 1

        # this is a load balancing config yaml
        result = load_router_config(router=None, config_file_path="../proxy/example_config_yaml/azure_config.yaml")
        print(result)
        assert len(result[1]) == 2


    except Exception as e:
        pytest.fail("Proxy: Got exception reading config", e)
test_load_router_config()
