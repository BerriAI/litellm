import sys, os, time
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
from concurrent.futures import ThreadPoolExecutor
# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import router, save_worker_config, startup_event  # Replace with the actual module where your FastAPI router is defined
filepath = os.path.dirname(os.path.abspath(__file__))
config_fp = f"{filepath}/test_configs/test_config.yaml"
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
    from litellm.proxy.proxy_server import cleanup_router_config_variables
    cleanup_router_config_variables()
    with TestClient(app) as client:
        yield client

def test_add_new_key(client):
    try:
        # Your test data
        test_data = {
            "models": ["gpt-3.5-turbo", "gpt-4", "claude-2", "azure-model"], 
            "aliases": {"mistral-7b": "gpt-3.5-turbo"}, 
            "duration": "20m"
        }
        print("testing proxy server")
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        assert result["key"].startswith("sk-")
        def _post_data():
            json_data = {'model': 'azure-model', "messages": [{"role": "user", "content": f"this is a test request, write a short poem {time.time()}"}]}
            response = client.post("/chat/completions", json=json_data, headers={"Authorization": f"Bearer {result['key']}"})
            return response
        _post_data()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")


def test_update_new_key(client):
    try:
        # Your test data
        test_data = {
            "models": ["gpt-3.5-turbo", "gpt-4", "claude-2", "azure-model"], 
            "aliases": {"mistral-7b": "gpt-3.5-turbo"}, 
            "duration": "20m"
        }
        print("testing proxy server")
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        assert result["key"].startswith("sk-")
        def _post_data():
            json_data = {'models': ['bedrock-models'], "key": result["key"]}
            response = client.post("/key/update", json=json_data, headers=headers)
            print(f"response text: {response.text}")
            assert response.status_code == 200
            return response
        _post_data()
        print(f"Received response: {result}")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")

# # Run the test - only runs via pytest


def test_add_new_key_max_parallel_limit(client):
    try:
        # Your test data
        test_data = {"duration": "20m", "max_parallel_requests": 1}
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        def _post_data():
            json_data = {'model': 'azure-model', "messages": [{"role": "user", "content": f"this is a test request, write a short poem {time.time()}"}]}
            response = client.post("/chat/completions", json=json_data, headers={"Authorization": f"Bearer {result['key']}"})
            return response
        def _run_in_parallel():
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(_post_data)
                future2 = executor.submit(_post_data)

                # Obtain the results from the futures
                response1 = future1.result()
                response2 = future2.result()
                if response1.status_code == 429 or response2.status_code == 429:
                    pass
                else: 
                    raise Exception()
        _run_in_parallel()
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")

def test_add_new_key_max_parallel_limit_streaming(client):
    try:
        # Your test data
        test_data = {"duration": "20m", "max_parallel_requests": 1}
        # Your bearer token
        token = os.getenv('PROXY_MASTER_KEY')

        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = client.post("/key/generate", json=test_data, headers=headers)
        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        def _post_data():
            json_data = {'model': 'azure-model', "messages": [{"role": "user", "content": f"this is a test request, write a short poem {time.time()}"}], "stream": True}
            response = client.post("/chat/completions", json=json_data, headers={"Authorization": f"Bearer {result['key']}"})
            return response
        def _run_in_parallel():
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(_post_data)
                future2 = executor.submit(_post_data)

                # Obtain the results from the futures
                response1 = future1.result()
                response2 = future2.result()
                if response1.status_code == 429 or response2.status_code == 429:
                    pass
                else: 
                    raise Exception()
        _run_in_parallel()
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")