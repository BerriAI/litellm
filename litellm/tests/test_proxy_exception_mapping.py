# test that the proxy actually does exception mapping to the OpenAI format

import sys, os
from dotenv import load_dotenv

load_dotenv()
import os, io, asyncio
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path 
import pytest
import litellm, openai
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import router, save_worker_config, initialize  # Replace with the actual module where your FastAPI router is defined

@pytest.fixture
def client():
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_bad_config.yaml"
    initialize(config=config_fp)
    app = FastAPI()
    app.include_router(router)  # Include your router in the test app
    return TestClient(app)

# raise openai.AuthenticationError
def test_chat_completion_exception(client):
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

        response = client.post("/chat/completions", json=test_data)

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(response=response)
        assert isinstance(openai_exception, openai.AuthenticationError)

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")
def test_chat_completion_exception_azure(client):
    try:
        # Your test data
        test_data = {
            "model": "azure-gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "hi"
                },
            ],
            "max_tokens": 10,
        }

        response = client.post("/chat/completions", json=test_data)

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(response=response)
        assert isinstance(openai_exception, openai.AuthenticationError)

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")