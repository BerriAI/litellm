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
