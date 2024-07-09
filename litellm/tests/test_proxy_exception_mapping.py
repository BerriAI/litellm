# test that the proxy actually does exception mapping to the OpenAI format

import json
import os
import sys
from unittest import mock

from dotenv import load_dotenv

load_dotenv()
import asyncio
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import openai
import pytest
from fastapi import Response
from fastapi.testclient import TestClient

import litellm
from litellm.proxy.proxy_server import (  # Replace with the actual module where your FastAPI router is defined
    initialize,
    router,
    save_worker_config,
)

invalid_authentication_error_response = Response(
    status_code=401,
    content=json.dumps({"error": "Invalid Authentication"}),
)
context_length_exceeded_error_response_dict = {
    "error": {
        "message": "AzureException - Error code: 400 - {'error': {'message': \"This model's maximum context length is 4096 tokens. However, your messages resulted in 10007 tokens. Please reduce the length of the messages.\", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}",
        "type": None,
        "param": None,
        "code": 400,
    },
}
context_length_exceeded_error_response = Response(
    status_code=400,
    content=json.dumps(context_length_exceeded_error_response_dict),
)


@pytest.fixture
def client():
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_bad_config.yaml"
    asyncio.run(initialize(config=config_fp))
    from litellm.proxy.proxy_server import app

    return TestClient(app)


# raise openai.AuthenticationError
def test_chat_completion_exception(client):
    try:
        # Your test data
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        response = client.post("/chat/completions", json=test_data)

        json_response = response.json()
        print("keys in json response", json_response.keys())
        assert json_response.keys() == {"error"}
        print("ERROR=", json_response["error"])
        assert isinstance(json_response["error"]["message"], str)
        assert (
            json_response["error"]["message"]
            == "litellm.AuthenticationError: AuthenticationError: OpenAIException - Incorrect API key provided: bad-key. You can find your API key at https://platform.openai.com/account/api-keys."
        )

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(
            response=response
        )
        assert isinstance(openai_exception, openai.AuthenticationError)

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


# raise openai.AuthenticationError
@mock.patch(
    "litellm.proxy.proxy_server.llm_router.acompletion",
    return_value=invalid_authentication_error_response,
)
def test_chat_completion_exception_azure(mock_acompletion, client):
    try:
        # Your test data
        test_data = {
            "model": "azure-gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        response = client.post("/chat/completions", json=test_data)

        mock_acompletion.assert_called_once_with(
            **test_data,
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
        )

        json_response = response.json()
        print("keys in json response", json_response.keys())
        assert json_response.keys() == {"error"}

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(
            response=response
        )
        print(openai_exception)
        assert isinstance(openai_exception, openai.AuthenticationError)

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


# raise openai.AuthenticationError
@mock.patch(
    "litellm.proxy.proxy_server.llm_router.aembedding",
    return_value=invalid_authentication_error_response,
)
def test_embedding_auth_exception_azure(mock_aembedding, client):
    try:
        # Your test data
        test_data = {"model": "azure-embedding", "input": ["hi"]}

        response = client.post("/embeddings", json=test_data)
        mock_aembedding.assert_called_once_with(
            **test_data,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
        )
        print("Response from proxy=", response)

        json_response = response.json()
        print("keys in json response", json_response.keys())
        assert json_response.keys() == {"error"}

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(
            response=response
        )
        print("Exception raised=", openai_exception)
        assert isinstance(openai_exception, openai.AuthenticationError)

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


# raise openai.BadRequestError
# chat/completions openai
def test_exception_openai_bad_model(client):
    try:
        # Your test data
        test_data = {
            "model": "azure/GPT-12",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        response = client.post("/chat/completions", json=test_data)

        json_response = response.json()
        print("keys in json response", json_response.keys())
        assert json_response.keys() == {"error"}

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(
            response=response
        )
        print("Type of exception=", type(openai_exception))
        assert isinstance(openai_exception, openai.BadRequestError)

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


# chat/completions any model
def test_chat_completion_exception_any_model(client):
    try:
        # Your test data
        test_data = {
            "model": "Lite-GPT-12",
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 10,
        }

        response = client.post("/chat/completions", json=test_data)

        json_response = response.json()
        assert json_response.keys() == {"error"}

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(
            response=response
        )
        assert isinstance(openai_exception, openai.BadRequestError)
        _error_message = openai_exception.message
        assert "chat_completion: Invalid model name passed in model=Lite-GPT-12" in str(
            _error_message
        )

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


# embeddings any model
def test_embedding_exception_any_model(client):
    try:
        # Your test data
        test_data = {"model": "Lite-GPT-12", "input": ["hi"]}

        response = client.post("/embeddings", json=test_data)
        print("Response from proxy=", response)
        print(response.json())

        json_response = response.json()
        print("keys in json response", json_response.keys())
        assert json_response.keys() == {"error"}

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(
            response=response
        )
        print("Exception raised=", openai_exception)
        assert isinstance(openai_exception, openai.BadRequestError)
        _error_message = openai_exception.message
        assert "embeddings: Invalid model name passed in model=Lite-GPT-12" in str(
            _error_message
        )

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")


# raise openai.BadRequestError
@mock.patch(
    "litellm.proxy.proxy_server.llm_router.acompletion",
    return_value=context_length_exceeded_error_response,
)
def test_chat_completion_exception_azure_context_window(mock_acompletion, client):
    try:
        # Your test data
        test_data = {
            "model": "working-azure-gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "hi" * 10000},
            ],
            "max_tokens": 10,
        }
        response = None

        response = client.post("/chat/completions", json=test_data)
        print("got response from server", response)

        mock_acompletion.assert_called_once_with(
            **test_data,
            litellm_call_id=mock.ANY,
            litellm_logging_obj=mock.ANY,
            request_timeout=mock.ANY,
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
        )

        json_response = response.json()

        print("keys in json response", json_response.keys())

        assert json_response.keys() == {"error"}

        assert json_response == context_length_exceeded_error_response_dict

        # make an openai client to call _make_status_error_from_response
        openai_client = openai.OpenAI(api_key="anything")
        openai_exception = openai_client._make_status_error_from_response(
            response=response
        )
        print("exception from proxy", openai_exception)
        assert isinstance(openai_exception, openai.BadRequestError)
        print("passed exception is of type BadRequestError")

    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception {str(e)}")
