import os
import sys

import pytest
import requests

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import responses

from litellm.proxy.client.chat import ChatClient
from litellm.proxy.client.exceptions import UnauthorizedError


@pytest.fixture
def base_url():
    return "http://localhost:8000"


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def client(base_url, api_key):
    return ChatClient(base_url=base_url, api_key=api_key)


@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Name 3 countries"},
    ]


def test_client_initialization(base_url, api_key):
    """Test that the ChatClient is properly initialized"""
    client = ChatClient(base_url=base_url, api_key=api_key)

    assert client._base_url == base_url
    assert client._api_key == api_key


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    base_url = "http://localhost:8000/////"
    client = ChatClient(base_url=base_url)

    assert client._base_url == "http://localhost:8000"


def test_client_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = ChatClient(base_url=base_url)

    assert client._api_key is None


def test_completions_request_creation(client, base_url, api_key, sample_messages):
    """Test that completions creates a request with correct URL, headers, and body"""
    request = client.completions(
        model="gpt-4",
        messages=sample_messages,
        temperature=0.7,
        max_tokens=100,
        return_request=True,
    )

    # Check request method and URL
    assert request.method == "POST"
    assert request.url == f"{base_url}/chat/completions"

    # Check headers
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"

    # Check request body
    assert request.json == {
        "model": "gpt-4",
        "messages": sample_messages,
        "temperature": 0.7,
        "max_tokens": 100,
    }


def test_completions_minimal_request(client, sample_messages):
    """Test that completions works with only required parameters"""
    request = client.completions(
        model="gpt-4", messages=sample_messages, return_request=True
    )

    # Check request body has only required fields
    assert request.json == {"model": "gpt-4", "messages": sample_messages}


def test_completions_all_parameters(client, sample_messages):
    """Test that completions accepts all optional parameters"""
    request = client.completions(
        model="gpt-4",
        messages=sample_messages,
        temperature=0.7,
        top_p=0.9,
        n=2,
        max_tokens=100,
        presence_penalty=0.5,
        frequency_penalty=-0.5,
        user="test-user",
        return_request=True,
    )

    # Check all parameters are included in request body
    assert request.json == {
        "model": "gpt-4",
        "messages": sample_messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 2,
        "max_tokens": 100,
        "presence_penalty": 0.5,
        "frequency_penalty": -0.5,
        "user": "test-user",
    }


@responses.activate
def test_completions_mock_response(client, sample_messages):
    """Test completions with a mocked successful response"""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    # Mock the POST request
    responses.add(
        responses.POST,
        f"{client._base_url}/chat/completions",
        json=mock_response,
        status=200,
    )

    response = client.completions(model="gpt-4", messages=sample_messages)

    assert response == mock_response
    assert (
        response["choices"][0]["message"]["content"]
        == "Hello! How can I help you today?"
    )


@responses.activate
def test_completions_unauthorized_error(client, sample_messages):
    """Test that completions raises UnauthorizedError for 401 responses"""
    # Mock a 401 response
    responses.add(
        responses.POST,
        f"{client._base_url}/chat/completions",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.completions(model="gpt-4", messages=sample_messages)


@responses.activate
def test_completions_other_errors(client, sample_messages):
    """Test that completions raises HTTPError for other error responses"""
    # Mock a 500 response
    responses.add(
        responses.POST,
        f"{client._base_url}/chat/completions",
        status=500,
        json={"error": "Internal Server Error"},
    )

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.completions(model="gpt-4", messages=sample_messages)
    assert exc_info.value.response.status_code == 500
