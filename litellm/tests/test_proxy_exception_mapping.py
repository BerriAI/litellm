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

def test_chat_completion_exception(client):
    try:
        base_url = client.base_url
        print("Base url of client= ", base_url)

        openai_client = openai.OpenAI(
            api_key="anything",
            base_url="http://0.0.0.0:8000",
        )

        response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages = [
            {
                "role": "user",
                "content": "this is a test request, write a short poem"
            },
        ])
    except openai.AuthenticationError:
        print("Got openai Auth Exception. Good job. The proxy mapped to OpenAI exceptions")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")