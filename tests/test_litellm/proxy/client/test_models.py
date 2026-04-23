import os
import sys

import pytest
import requests

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path



import responses

from litellm.proxy.client import Client, ModelsManagementClient
from litellm.proxy.client.exceptions import NotFoundError, UnauthorizedError


@pytest.fixture
def base_url():
    return "http://localhost:8000"


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def client(base_url, api_key):
    return ModelsManagementClient(base_url=base_url, api_key=api_key)


def test_list_request_creation(client, base_url, api_key):
    """Test that list creates a request with correct URL and headers when return_request=True"""
    request = client.list(return_request=True)

    # Check request method
    assert request.method == "GET"

    # Check URL construction
    expected_url = f"{base_url}/models"
    assert request.url == expected_url

    # Check authorization header
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == f"Bearer {api_key}"


def test_list_request_no_auth(base_url):
    """Test that list creates a request without auth header when no api_key is provided"""
    client = ModelsManagementClient(base_url=base_url)  # No API key
    request = client.list(return_request=True)

    # Check URL is still correct
    assert request.url == f"{base_url}/models"

    # Check that there's no authorization header
    assert "Authorization" not in request.headers


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("http://localhost:8000", "http://localhost:8000/models"),
        (
            "http://localhost:8000/",
            "http://localhost:8000/models",
        ),  # With trailing slash
        ("https://api.example.com", "https://api.example.com/models"),
        ("http://127.0.0.1:3000", "http://127.0.0.1:3000/models"),
    ],
)
def test_list_url_variants(base_url, expected):
    """Test that list handles different base URL formats correctly"""
    client = ModelsManagementClient(base_url=base_url)
    request = client.list(return_request=True)
    assert request.url == expected


@responses.activate
def test_list_with_mock_response(client):
    """Test the full list execution with a mocked response"""
    mock_data = {
        "data": [
            {"id": "gpt-4", "type": "model"},
            {"id": "gpt-3.5-turbo", "type": "model"},
        ]
    }
    responses.add(
        responses.GET,
        "http://localhost:8000/models",
        json=mock_data,
        status=200,
    )

    response = client.list()
    assert response == mock_data["data"]
    assert len(response) == 2
    assert response[0]["id"] == "gpt-4"


@responses.activate
def test_list_unauthorized_error(client):
    """Test that list raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.GET,
        "http://localhost:8000/models",
        status=401,
        json={"error": "Invalid API key"},
    )

    with pytest.raises(UnauthorizedError) as exc_info:
        client.list()
    assert exc_info.value.orig_exception.response.status_code == 401


@responses.activate
def test_list_other_errors(client):
    """Test that list raises normal HTTPError for non-401 errors"""
    responses.add(
        responses.GET,
        "http://localhost:8000/models",
        status=500,
        json={"error": "Internal Server Error"},
    )

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.list()
    assert exc_info.value.response.status_code == 500


@pytest.mark.parametrize(
    "api_key",
    [
        "",  # Empty string
        None,  # None value
    ],
)
def test_list_invalid_api_keys(base_url, api_key):
    """Test that the client handles invalid API keys appropriately"""
    client = ModelsManagementClient(base_url=base_url, api_key=api_key)
    request = client.list(return_request=True)
    assert "Authorization" not in request.headers


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    client = ModelsManagementClient(base_url="http://localhost:8000/////")
    assert client._base_url == "http://localhost:8000"


def test_client_initialization(base_url, api_key):
    """Test that the Client is properly initialized with all resource clients"""
    client = Client(base_url=base_url, api_key=api_key)

    # Check base properties
    assert client._base_url == base_url
    assert client._api_key == api_key

    # Check resource clients
    assert isinstance(client.models, ModelsManagementClient)
    assert client.models._base_url == base_url
    assert client.models._api_key == api_key


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    base_url = "http://localhost:8000/////"
    client = Client(base_url=base_url)

    assert client._base_url == "http://localhost:8000"
    assert client.models._base_url == "http://localhost:8000"


def test_client_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = Client(base_url=base_url)

    assert client._api_key is None
    assert client.models._api_key is None


def test_new_request_creation(client, base_url, api_key):
    """Test that new creates a request with correct URL, headers, and body when return_request=True"""
    model_name = "gpt-4"
    model_params = {"model": "openai/gpt-4", "api_base": "https://api.openai.com/v1"}
    model_info = {"description": "GPT-4 model", "metadata": {"version": "1.0"}}

    request = client.new(
        model_name=model_name,
        model_params=model_params,
        model_info=model_info,
        return_request=True,
    )

    # Check request method and URL
    assert request.method == "POST"
    assert request.url == f"{base_url}/model/new"

    # Check headers
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == f"Bearer {api_key}"

    # Check request body
    assert request.json == {
        "model_name": model_name,
        "litellm_params": model_params,
        "model_info": model_info,
    }


def test_new_without_model_info(client):
    """Test that new works correctly without optional model_info"""
    model_name = "gpt-4"
    model_params = {"model": "openai/gpt-4", "api_base": "https://api.openai.com/v1"}

    request = client.new(
        model_name=model_name, model_params=model_params, return_request=True
    )

    # Check request body doesn't include model_info
    assert request.json == {"model_name": model_name, "litellm_params": model_params}


@responses.activate
def test_new_mock_response(client):
    """Test new with a mocked successful response"""
    model_name = "gpt-4"
    model_params = {"model": "openai/gpt-4"}
    mock_response = {"model_id": "123", "status": "success"}

    # Mock the POST request
    responses.add(
        responses.POST,
        f"{client._base_url}/model/new",
        json=mock_response,
        status=200,
    )

    response = client.new(model_name=model_name, model_params=model_params)

    assert response == mock_response


@responses.activate
def test_new_unauthorized_error(client):
    """Test that new raises UnauthorizedError for 401 responses"""
    model_name = "gpt-4"
    model_params = {"model": "openai/gpt-4"}

    # Mock a 401 response
    responses.add(
        responses.POST,
        f"{client._base_url}/model/new",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.new(model_name=model_name, model_params=model_params)


def test_delete_request_creation(client, base_url, api_key):
    """Test that delete creates a request with correct URL, headers, and body when return_request=True"""
    model_id = "model-123"

    request = client.delete(model_id=model_id, return_request=True)

    # Check request method and URL
    assert request.method == "POST"
    assert request.url == f"{base_url}/model/delete"

    # Check headers
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == f"Bearer {api_key}"

    # Check request body
    assert request.json == {"id": model_id}


@responses.activate
def test_delete_mock_response(client):
    """Test delete with a mocked successful response"""
    model_id = "model-123"
    mock_response = {"message": "Model: model-123 deleted successfully"}

    # Mock the POST request
    responses.add(
        responses.POST,
        f"{client._base_url}/model/delete",
        json=mock_response,
        status=200,
    )

    response = client.delete(model_id=model_id)
    assert response == mock_response


@responses.activate
def test_delete_unauthorized_error(client):
    """Test that delete raises UnauthorizedError for 401 responses"""
    model_id = "model-123"

    # Mock a 401 response
    responses.add(
        responses.POST,
        f"{client._base_url}/model/delete",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.delete(model_id=model_id)


@responses.activate
def test_delete_404_error(client):
    """Test that delete raises NotFoundError for 404 responses"""
    model_id = "model-123"

    # Mock a 404 response
    responses.add(
        responses.POST,
        f"{client._base_url}/model/delete",
        status=404,
        json={"error": "Model not found"},
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.delete(model_id=model_id)
    assert exc_info.value.orig_exception.response.status_code == 404


@responses.activate
def test_delete_not_found_in_text(client):
    """Test that delete raises NotFoundError when response contains 'not found'"""
    model_id = "model-123"

    # Mock a response with "not found" in text but different status code
    responses.add(
        responses.POST,
        f"{client._base_url}/model/delete",
        status=400,  # Different status code
        json={"error": "The specified model was not found in the system"},
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.delete(model_id=model_id)
    assert "not found" in exc_info.value.orig_exception.response.text.lower()


@responses.activate
def test_delete_other_errors(client):
    """Test that delete raises normal HTTPError for other error responses"""
    model_id = "model-123"

    # Mock a 500 response
    responses.add(
        responses.POST,
        f"{client._base_url}/model/delete",
        status=500,
        json={"error": "Internal Server Error"},
    )

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.delete(model_id=model_id)
    assert exc_info.value.response.status_code == 500


def test_info_request_creation(client, base_url, api_key):
    """Test that info creates a correct request"""
    request = client.info(return_request=True)

    # Check request method and URL
    assert request.method == "GET"
    assert request.url == f"{base_url}/v1/model/info"

    # Check headers
    assert request.headers["Authorization"] == f"Bearer {api_key}"


@responses.activate
def test_info_success(client):
    """Test info with a successful response"""
    mock_response = {
        "data": [
            {
                "model_name": "gpt-4",
                "model_info": {"id": "model-123", "description": "GPT-4 model"},
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_base": "https://api.openai.com/v1",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "model_info": {"id": "model-456", "description": "GPT-3.5 Turbo model"},
                "litellm_params": {"model": "openai/gpt-3.5-turbo"},
            },
        ]
    }

    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        json=mock_response,
        status=200,
    )

    response = client.info()
    assert response == mock_response["data"]
    assert len(response) == 2
    assert response[0]["model_name"] == "gpt-4"
    assert response[1]["model_name"] == "gpt-3.5-turbo"


@responses.activate
def test_info_unauthorized(client):
    """Test that info raises UnauthorizedError for unauthorized requests"""
    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError) as exc_info:
        client.info()
    assert exc_info.value.orig_exception.response.status_code == 401


@responses.activate
def test_info_server_error(client):
    """Test that info raises HTTPError for server errors"""
    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        status=500,
        json={"error": "Internal Server Error"},
    )

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.info()
    assert exc_info.value.response.status_code == 500


def test_get_by_id_request_creation(client, base_url, api_key):
    """Test that get creates a correct request when using model_id"""
    model_id = "model-123"
    request = client.get(model_id=model_id, return_request=True)

    # Check request method and URL
    assert request.method == "GET"
    assert request.url == f"{base_url}/v1/model/info"

    # Check headers
    assert request.headers["Authorization"] == f"Bearer {api_key}"


def test_get_by_name_request_creation(client, base_url):
    """Test that get creates a correct request when using model_name"""
    model_name = "gpt-4"
    request = client.get(model_name=model_name, return_request=True)

    # Check it's a GET request to /v1/model/info
    assert request.method == "GET"
    assert request.url == f"{base_url}/v1/model/info"


def test_get_invalid_params():
    """Test that get raises ValueError for invalid parameter combinations"""
    client = ModelsManagementClient(base_url="http://localhost:8000")

    # Test with no parameters
    with pytest.raises(ValueError) as exc_info:
        client.get()
    assert "Exactly one of model_id or model_name must be provided" in str(
        exc_info.value
    )

    # Test with both parameters
    with pytest.raises(ValueError) as exc_info:
        client.get(model_id="123", model_name="gpt-4")
    assert "Exactly one of model_id or model_name must be provided" in str(
        exc_info.value
    )


@responses.activate
def test_get_success_by_id(client):
    """Test get successfully finding a model by ID"""
    model_id = "model-123"
    mock_models = {
        "data": [
            {"model_name": "gpt-3.5-turbo", "model_info": {"id": "other-model"}},
            {
                "model_name": "gpt-4",
                "model_info": {"id": model_id},
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_base": "https://api.openai.com/v1",
                },
            },
        ]
    }

    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        json=mock_models,
        status=200,
    )

    response = client.get(model_id=model_id)
    assert response["model_info"]["id"] == model_id
    assert response["model_name"] == "gpt-4"


@responses.activate
def test_get_success_by_name(client):
    """Test get successfully finding a model by name"""
    model_name = "gpt-4"
    mock_models = {
        "data": [
            {
                "model_name": model_name,
                "model_info": {"id": "model-123"},
                "litellm_params": {"model": "openai/gpt-4"},
            }
        ]
    }

    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        json=mock_models,
        status=200,
    )

    response = client.get(model_name=model_name)
    assert response["model_name"] == model_name


@responses.activate
def test_get_not_found(client):
    """Test that get raises NotFoundError when model is not found"""
    model_name = "nonexistent-model"

    # Mock successful response but with no matching model
    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        json={
            "data": [
                {"model_name": "gpt-3.5-turbo", "model_info": {"id": "other-model"}}
            ]
        },
        status=200,
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.get(model_name=model_name)
    assert "not found" in str(exc_info.value).lower()
    assert "model_name=" + model_name in str(exc_info.value)


@responses.activate
def test_get_unauthorized(client):
    """Test that get raises UnauthorizedError for unauthorized requests"""
    model_id = "model-123"

    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError) as exc_info:
        client.get(model_id=model_id)
    assert exc_info.value.orig_exception.response.status_code == 401


@responses.activate
def test_get_server_error(client):
    """Test that get raises HTTPError for server errors"""
    model_id = "model-123"

    responses.add(
        responses.GET,
        f"{client._base_url}/v1/model/info",
        status=500,
        json={"error": "Internal Server Error"},
    )

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.get(model_id=model_id)
    assert exc_info.value.response.status_code == 500


def test_update_request_creation(client, base_url, api_key):
    """Test that update creates a request with correct URL, headers, and body when return_request=True"""
    model_id = "model-123"
    model_params = {"model": "openai/gpt-4", "api_base": "https://api.openai.com/v1"}
    model_info = {"description": "Updated GPT-4 model", "metadata": {"version": "2.0"}}

    request = client.update(
        model_id=model_id,
        model_params=model_params,
        model_info=model_info,
        return_request=True,
    )

    # Check request method and URL
    assert request.method == "POST"
    assert request.url == f"{base_url}/model/update"

    # Check headers
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == f"Bearer {api_key}"

    # Check request body
    assert request.json == {
        "id": model_id,
        "litellm_params": model_params,
        "model_info": model_info,
    }


def test_update_without_model_info(client):
    """Test that update works correctly without optional model_info"""
    model_id = "model-123"
    model_params = {"model": "openai/gpt-4", "api_base": "https://api.openai.com/v1"}

    request = client.update(
        model_id=model_id, model_params=model_params, return_request=True
    )

    # Check request body doesn't include model_info
    assert request.json == {"id": model_id, "litellm_params": model_params}


@responses.activate
def test_update_mock_response(client):
    """Test update with a mocked successful response"""
    model_id = "model-123"
    model_params = {"model": "openai/gpt-4"}
    mock_response = {
        "id": model_id,
        "status": "success",
        "message": "Model updated successfully",
    }

    # Mock the POST request
    responses.add(
        responses.POST,
        f"{client._base_url}/model/update",
        json=mock_response,
        status=200,
    )

    response = client.update(model_id=model_id, model_params=model_params)

    assert response == mock_response


@responses.activate
def test_update_unauthorized_error(client):
    """Test that update raises UnauthorizedError for 401 responses"""
    model_id = "model-123"
    model_params = {"model": "openai/gpt-4"}

    # Mock a 401 response
    responses.add(
        responses.POST,
        f"{client._base_url}/model/update",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.update(model_id=model_id, model_params=model_params)


@responses.activate
def test_update_404_error(client):
    """Test that update raises NotFoundError for 404 responses"""
    model_id = "model-123"
    model_params = {"model": "openai/gpt-4"}

    # Mock a 404 response
    responses.add(
        responses.POST,
        f"{client._base_url}/model/update",
        status=404,
        json={"error": "Model not found"},
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.update(model_id=model_id, model_params=model_params)
    assert exc_info.value.orig_exception.response.status_code == 404


@responses.activate
def test_update_not_found_in_text(client):
    """Test that update raises NotFoundError when response contains 'not found'"""
    model_id = "model-123"
    model_params = {"model": "openai/gpt-4"}

    # Mock a response with "not found" in text but different status code
    responses.add(
        responses.POST,
        f"{client._base_url}/model/update",
        status=400,  # Different status code
        json={"error": "The specified model was not found in the system"},
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.update(model_id=model_id, model_params=model_params)
    assert "not found" in exc_info.value.orig_exception.response.text.lower()


@responses.activate
def test_update_other_errors(client):
    """Test that update raises normal HTTPError for other error responses"""
    model_id = "model-123"
    model_params = {"model": "openai/gpt-4"}

    # Mock a 500 response
    responses.add(
        responses.POST,
        f"{client._base_url}/model/update",
        status=500,
        json={"error": "Internal Server Error"},
    )

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.update(model_id=model_id, model_params=model_params)
    assert exc_info.value.response.status_code == 500
