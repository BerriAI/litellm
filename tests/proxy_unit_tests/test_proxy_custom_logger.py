import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io, asyncio

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, time
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError
import importlib, inspect

# test /chat/completion request to the proxy
from fastapi.testclient import TestClient
from fastapi import FastAPI
from litellm.proxy.proxy_server import (
    router,
    save_worker_config,
    initialize,
)  # Replace with the actual module where your FastAPI router is defined

filepath = os.path.dirname(os.path.abspath(__file__))
python_file_path = f"{filepath}/test_configs/custom_callbacks.py"


@pytest.fixture
def client():
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_custom_logger.yaml"
    app = FastAPI()
    asyncio.run(initialize(config=config_fp))
    app.include_router(router)  # Include your router in the test app
    return TestClient(app)


# Your bearer token
token = os.getenv("PROXY_MASTER_KEY")

headers = {"Authorization": f"Bearer {token}"}


print("Testing proxy custom logger")


def test_embedding(client):
    try:
        litellm.set_verbose = False
        from litellm.proxy.types_utils.utils import get_instance_fn

        my_custom_logger = get_instance_fn(
            value="custom_callbacks.my_custom_logger", config_file_path=python_file_path
        )
        print("id of initialized custom logger", id(my_custom_logger))
        litellm.callbacks = [my_custom_logger]
        # Your test data
        print("initialized proxy")
        # import the initialized custom logger
        print(litellm.callbacks)

        # assert len(litellm.callbacks) == 1 # assert litellm is initialized with 1 callback
        print("my_custom_logger", my_custom_logger)
        assert my_custom_logger.async_success_embedding is False

        test_data = {"model": "azure-embedding-model", "input": ["hello"]}
        response = client.post("/embeddings", json=test_data, headers=headers)
        print("made request", response.status_code, response.text)
        print(
            "vars my custom logger /embeddings",
            vars(my_custom_logger),
            "id",
            id(my_custom_logger),
        )
        assert (
            my_custom_logger.async_success_embedding is True
        )  # checks if the status of async_success is True, only the async_log_success_event can set this to true
        assert (
            my_custom_logger.async_embedding_kwargs["model"] == "text-embedding-ada-002"
        )  # checks if kwargs passed to async_log_success_event are correct
        kwargs = my_custom_logger.async_embedding_kwargs
        litellm_params = kwargs.get("litellm_params")
        metadata = litellm_params.get("metadata", None)
        print("\n\n Metadata in custom logger kwargs", litellm_params.get("metadata"))
        assert metadata is not None
        assert "user_api_key" in metadata
        assert "headers" in metadata
        proxy_server_request = litellm_params.get("proxy_server_request")
        model_info = litellm_params.get("model_info")
        assert proxy_server_request == {
            "url": "http://testserver/embeddings",
            "method": "POST",
            "headers": {
                "host": "testserver",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, zstd",
                "connection": "keep-alive",
                "user-agent": "testclient",
                "content-length": "51",
                "content-type": "application/json",
            },
            "body": {"model": "azure-embedding-model", "input": ["hello"]},
        }
        assert model_info == {
            "input_cost_per_token": 0.002,
            "mode": "embedding",
            "id": "hello",
            "db_model": False,
        }
        result = response.json()
        print(f"Received response: {result}")
        print("Passed Embedding custom logger on proxy!")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


def test_chat_completion(client):
    try:
        # Your test data
        litellm.set_verbose = False
        from litellm.proxy.types_utils.utils import get_instance_fn

        my_custom_logger = get_instance_fn(
            value="custom_callbacks.my_custom_logger", config_file_path=python_file_path
        )

        print("id of initialized custom logger", id(my_custom_logger))

        litellm.callbacks = [my_custom_logger]
        # import the initialized custom logger
        print(litellm.callbacks)

        # assert len(litellm.callbacks) == 1 # assert litellm is initialized with 1 callback

        print("LiteLLM Callbacks", litellm.callbacks)
        print("my_custom_logger", my_custom_logger)
        assert my_custom_logger.async_success == False

        test_data = {
            "model": "Azure OpenAI GPT-4 Canada",
            "messages": [
                {"role": "user", "content": "write a litellm poem"},
            ],
            "max_tokens": 10,
        }

        response = client.post("/chat/completions", json=test_data, headers=headers)
        print("made request", response.status_code, response.text)
        print("LiteLLM Callbacks", litellm.callbacks)
        time.sleep(1)  # sleep while waiting for callback to run

        print(
            "my_custom_logger in /chat/completions",
            my_custom_logger,
            "id",
            id(my_custom_logger),
        )
        print("vars my custom logger, ", vars(my_custom_logger))
        assert (
            my_custom_logger.async_success == True
        )  # checks if the status of async_success is True, only the async_log_success_event can set this to true
        assert (
            my_custom_logger.async_completion_kwargs["model"] == "gpt-4.1-nano"
        )  # checks if kwargs passed to async_log_success_event are correct
        print(
            "\n\n Custom Logger Async Completion args",
            my_custom_logger.async_completion_kwargs,
        )
        litellm_params = my_custom_logger.async_completion_kwargs.get("litellm_params")
        metadata = litellm_params.get("metadata", None)
        print("\n\n Metadata in custom logger kwargs", litellm_params.get("metadata"))
        assert metadata is not None
        assert "user_api_key" in metadata
        assert "user_api_key_metadata" in metadata
        assert "headers" in metadata
        config_model_info = litellm_params.get("model_info")
        proxy_server_request_object = litellm_params.get("proxy_server_request")

        assert config_model_info == {
            "id": "gm",
            "input_cost_per_token": 0.0002,
            "mode": "chat",
            "db_model": False,
        }

        assert "authorization" not in proxy_server_request_object["headers"]
        assert proxy_server_request_object == {
            "url": "http://testserver/chat/completions",
            "method": "POST",
            "headers": {
                "host": "testserver",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, zstd",
                "connection": "keep-alive",
                "user-agent": "testclient",
                "content-length": "115",
                "content-type": "application/json",
            },
            "body": {
                "model": "Azure OpenAI GPT-4 Canada",
                "messages": [{"role": "user", "content": "write a litellm poem"}],
                "max_tokens": 10,
            },
        }
        result = response.json()
        print(f"Received response: {result}")
        print("\nPassed /chat/completions with Custom Logger!")
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


def test_chat_completion_stream(client):
    try:
        # Your test data
        litellm.set_verbose = False
        from litellm.proxy.types_utils.utils import get_instance_fn

        my_custom_logger = get_instance_fn(
            value="custom_callbacks.my_custom_logger", config_file_path=python_file_path
        )

        print("id of initialized custom logger", id(my_custom_logger))

        litellm.callbacks = [my_custom_logger]
        import json

        print("initialized proxy")
        # import the initialized custom logger
        print(litellm.callbacks)

        print("LiteLLM Callbacks", litellm.callbacks)
        print("my_custom_logger", my_custom_logger)

        assert (
            my_custom_logger.streaming_response_obj == None
        )  # no streaming response obj is set pre call

        test_data = {
            "model": "Azure OpenAI GPT-4 Canada",
            "messages": [
                {"role": "user", "content": "write 1 line poem about LiteLLM"},
            ],
            "max_tokens": 40,
            "stream": True,  # streaming  call
        }

        response = client.post("/chat/completions", json=test_data, headers=headers)
        print("made request", response.status_code, response.text)
        complete_response = ""
        for line in response.iter_lines():
            if line:
                # Process the streaming data line here
                print("\n\n Line", line)
                print(line)
                line = str(line)

                json_data = line.replace("data: ", "")

                if "[DONE]" in json_data:
                    break

                # Parse the JSON string
                data = json.loads(json_data)

                print("\n\n decode_data", data)

                # Access the content of choices[0]['message']['content']
                content = data["choices"][0]["delta"].get("content", None) or ""

                # Process the content as needed
                print("Content:", content)

                complete_response += content

        print("\n\nHERE is the complete streaming response string", complete_response)
        print("\n\nHERE IS the streaming Response from callback\n\n")
        print(my_custom_logger.streaming_response_obj)
        import time

        time.sleep(0.5)

        streamed_response = my_custom_logger.streaming_response_obj
        assert (
            complete_response == streamed_response["choices"][0]["message"]["content"]
        )

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")
