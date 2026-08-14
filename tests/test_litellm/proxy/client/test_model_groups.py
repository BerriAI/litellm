import os
import sys

import pytest
import requests

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import responses

from litellm.proxy.client import Client, ModelGroupsManagementClient
from litellm.proxy.client.exceptions import UnauthorizedError


@pytest.fixture
def base_url():
    return "http://localhost:8000"


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def client(base_url, api_key):
    return ModelGroupsManagementClient(base_url=base_url, api_key=api_key)


def test_info_request_creation(client, base_url, api_key):
    """Test that info creates a request with correct URL and headers when return_request=True"""
    request = client.info(return_request=True)

    # Check request method
    assert request.method == "GET"

    # Check URL construction
    expected_url = f"{base_url}/model_group/info"
    assert request.url == expected_url

    # Check authorization header
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == f"Bearer {api_key}"


def test_info_request_no_auth(base_url):
    """Test that info creates a request without auth header when no api_key is provided"""
    client = ModelGroupsManagementClient(base_url=base_url)  # No API key
    request = client.info(return_request=True)

    # Check URL is still correct
    assert request.url == f"{base_url}/model_group/info"

    # Check that there's no authorization header
    assert "Authorization" not in request.headers


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("http://localhost:8000", "http://localhost:8000/model_group/info"),
        (
            "http://localhost:8000/",
            "http://localhost:8000/model_group/info",
        ),  # With trailing slash
        ("https://api.example.com", "https://api.example.com/model_group/info"),
        ("http://127.0.0.1:3000", "http://127.0.0.1:3000/model_group/info"),
    ],
)
def test_info_url_variants(base_url, expected):
    """Test that info handles different base URL formats correctly"""
    client = ModelGroupsManagementClient(base_url=base_url)
    request = client.info(return_request=True)
    assert request.url == expected


@responses.activate
def test_info_with_mock_response(client):
    """Test the full info execution with a mocked response"""
    mock_data = {
        "data": [
            {
                "model_group_name": "gpt4-group",
                "models": ["gpt-4", "gpt-4-32k"],
                "litellm_params": {"timeout": 30, "max_retries": 3},
            },
            {
                "model_group_name": "azure-group",
                "models": ["azure-gpt-4", "azure-gpt-35"],
                "litellm_params": {
                    "api_base": "https://azure-endpoint.com",
                    "api_version": "2023-05-15",
                },
            },
        ]
    }
    responses.add(
        responses.GET,
        f"{client._base_url}/model_group/info",
        json=mock_data,
        status=200,
    )

    response = client.info()
    assert response == mock_data["data"]
    assert len(response) == 2
    assert response[0]["model_group_name"] == "gpt4-group"
    assert response[1]["model_group_name"] == "azure-group"


@responses.activate
def test_info_unauthorized_error(client):
    """Test that info raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.GET,
        f"{client._base_url}/model_group/info",
        status=401,
        json={"error": "Invalid API key"},
    )

    with pytest.raises(UnauthorizedError) as exc_info:
        client.info()
    assert exc_info.value.orig_exception.response.status_code == 401


@responses.activate
def test_info_other_errors(client):
    """Test that info raises normal HTTPError for non-401 errors"""
    responses.add(
        responses.GET,
        f"{client._base_url}/model_group/info",
        status=500,
        json={"error": "Internal Server Error"},
    )

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.info()
    assert exc_info.value.response.status_code == 500


@pytest.mark.parametrize(
    "api_key",
    [
        "",  # Empty string
        None,  # None value
    ],
)
def test_info_invalid_api_keys(base_url, api_key):
    """Test that the client handles invalid API keys appropriately"""
    client = ModelGroupsManagementClient(base_url=base_url, api_key=api_key)
    request = client.info(return_request=True)
    assert "Authorization" not in request.headers


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    client = ModelGroupsManagementClient(base_url="http://localhost:8000/////")
    assert client._base_url == "http://localhost:8000"


def test_client_initialization(base_url, api_key):
    """Test that the Client properly initializes the model_groups client"""
    client = Client(base_url=base_url, api_key=api_key)

    # Check that model_groups client is properly initialized
    assert isinstance(client.model_groups, ModelGroupsManagementClient)
    assert client.model_groups._base_url == base_url
    assert client.model_groups._api_key == api_key


def test_client_initialization_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = Client(base_url=base_url)

    assert client._api_key is None
    assert client.model_groups._api_key is None
