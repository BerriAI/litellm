import pytest
import requests
from litellm.proxy.client import Client, ModelsManagementClient
from litellm.proxy.client.exceptions import UnauthorizedError

@pytest.fixture
def base_url():
    return "http://localhost:8000"

@pytest.fixture
def api_key():
    return "test-api-key"

@pytest.fixture
def client(base_url, api_key):
    return ModelsManagementClient(base_url=base_url, api_key=api_key)

def test_list_models_request_creation(client, base_url, api_key):
    """Test that list_models creates a request with correct URL and headers when return_request=True"""
    request = client.list_models(return_request=True)
    
    # Check request method
    assert request.method == 'GET'
    
    # Check URL construction
    expected_url = f"{base_url}/models"
    assert request.url == expected_url
    
    # Check authorization header
    assert 'Authorization' in request.headers
    assert request.headers['Authorization'] == f"Bearer {api_key}"

def test_list_models_request_no_auth(base_url):
    """Test that list_models creates a request without auth header when no api_key is provided"""
    client = ModelsManagementClient(base_url=base_url)  # No API key
    request = client.list_models(return_request=True)
    
    # Check URL is still correct
    assert request.url == f"{base_url}/models"
    
    # Check that there's no authorization header
    assert 'Authorization' not in request.headers

@pytest.mark.parametrize("base_url,expected", [
    ("http://localhost:8000", "http://localhost:8000/models"),
    ("http://localhost:8000/", "http://localhost:8000/models"),  # With trailing slash
    ("https://api.example.com", "https://api.example.com/models"),
    ("http://127.0.0.1:3000", "http://127.0.0.1:3000/models"),
])
def test_list_models_url_variants(base_url, expected):
    """Test that list_models handles different base URL formats correctly"""
    client = ModelsManagementClient(base_url=base_url)
    request = client.list_models(return_request=True)
    assert request.url == expected

def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    client = ModelsManagementClient(base_url="http://localhost:8000/////")
    assert client.base_url == "http://localhost:8000"
    
def test_list_models_with_mock_response(client, requests_mock):
    """Test the full list_models execution with a mocked response"""
    mock_data = {
        "data": [
            {"id": "gpt-4", "type": "model"},
            {"id": "gpt-3.5-turbo", "type": "model"}
        ]
    }
    requests_mock.get("http://localhost:8000/models", json=mock_data)
    
    response = client.list_models()
    assert response == mock_data["data"]
    assert len(response) == 2
    assert response[0]["id"] == "gpt-4"

def test_list_models_unauthorized_error(client, requests_mock):
    """Test that list_models raises UnauthorizedError for 401 responses"""
    requests_mock.get(
        "http://localhost:8000/models",
        status_code=401,
        json={"error": "Invalid API key"}
    )
    
    with pytest.raises(UnauthorizedError) as exc_info:
        client.list_models()
    assert exc_info.value.orig_exception.response.status_code == 401

def test_list_models_other_errors(client, requests_mock):
    """Test that list_models raises normal HTTPError for non-401 errors"""
    requests_mock.get(
        "http://localhost:8000/models",
        status_code=500,
        json={"error": "Internal Server Error"}
    )
    
    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.list_models()
    assert exc_info.value.response.status_code == 500

@pytest.mark.parametrize("api_key", [
    "",  # Empty string
    None,  # None value
])
def test_list_models_invalid_api_keys(base_url, api_key):
    """Test that the client handles invalid API keys appropriately"""
    client = ModelsManagementClient(base_url=base_url, api_key=api_key)
    request = client.list_models(return_request=True)
    assert 'Authorization' not in request.headers 

def test_client_initialization(base_url, api_key):
    """Test that the Client is properly initialized with all resource clients"""
    client = Client(base_url=base_url, api_key=api_key)
    
    # Check base properties
    assert client.base_url == base_url
    assert client.api_key == api_key
    
    # Check resource clients
    assert isinstance(client.models, ModelsManagementClient)
    assert client.models.base_url == base_url
    assert client.models.api_key == api_key

def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    base_url = "http://localhost:8000/////"
    client = Client(base_url=base_url)
    
    assert client.base_url == "http://localhost:8000"
    assert client.models.base_url == "http://localhost:8000"

def test_client_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = Client(base_url=base_url)
    
    assert client.api_key is None
    assert client.models.api_key is None

def test_add_model_request_creation(client, base_url, api_key):
    """Test that add_model creates a request with correct URL, headers, and body when return_request=True"""
    model_name = "gpt-4"
    model_params = {
        "model": "openai/gpt-4",
        "api_base": "https://api.openai.com/v1"
    }
    model_info = {
        "description": "GPT-4 model",
        "metadata": {"version": "1.0"}
    }
    
    request = client.add_model(
        model_name=model_name,
        model_params=model_params,
        model_info=model_info,
        return_request=True
    )
    
    # Check request method and URL
    assert request.method == 'POST'
    assert request.url == f"{base_url}/model/new"
    
    # Check headers
    assert 'Authorization' in request.headers
    assert request.headers['Authorization'] == f"Bearer {api_key}"
    
    # Check request body
    assert request.json == {
        "model_name": model_name,
        "litellm_params": model_params,
        "model_info": model_info
    }

def test_add_model_without_model_info(client):
    """Test that add_model works correctly without optional model_info"""
    model_name = "gpt-4"
    model_params = {
        "model": "openai/gpt-4",
        "api_base": "https://api.openai.com/v1"
    }
    
    request = client.add_model(
        model_name=model_name,
        model_params=model_params,
        return_request=True
    )
    
    # Check request body doesn't include model_info
    assert request.json == {
        "model_name": model_name,
        "litellm_params": model_params
    }

def test_add_model_mock_response(client, requests_mock):
    """Test add_model with a mocked successful response"""
    model_name = "gpt-4"
    model_params = {"model": "openai/gpt-4"}
    mock_response = {
        "model_id": "123",
        "status": "success"
    }
    
    # Mock the POST request
    requests_mock.post(
        f"{client.base_url}/model/new",
        json=mock_response
    )
    
    response = client.add_model(
        model_name=model_name,
        model_params=model_params
    )
    
    assert response == mock_response

def test_add_model_unauthorized_error(client, requests_mock):
    """Test that add_model raises UnauthorizedError for 401 responses"""
    model_name = "gpt-4"
    model_params = {"model": "openai/gpt-4"}
    
    # Mock a 401 response
    requests_mock.post(
        f"{client.base_url}/model/new",
        status_code=401,
        json={"error": "Unauthorized"}
    )
    
    with pytest.raises(UnauthorizedError):
        client.add_model(
            model_name=model_name,
            model_params=model_params
        )

def test_delete_model_request_creation(client, base_url, api_key):
    """Test that delete_model creates a request with correct URL, headers, and body when return_request=True"""
    model_id = "model-123"
    
    request = client.delete_model(
        model_id=model_id,
        return_request=True
    )
    
    # Check request method and URL
    assert request.method == 'POST'
    assert request.url == f"{base_url}/model/delete"
    
    # Check headers
    assert 'Authorization' in request.headers
    assert request.headers['Authorization'] == f"Bearer {api_key}"
    
    # Check request body
    assert request.json == {"id": model_id}

def test_delete_model_mock_response(client, requests_mock):
    """Test delete_model with a mocked successful response"""
    model_id = "model-123"
    mock_response = {
        "message": "Model: model-123 deleted successfully"
    }
    
    # Mock the POST request
    requests_mock.post(
        f"{client.base_url}/model/delete",
        json=mock_response
    )
    
    response = client.delete_model(model_id=model_id)
    assert response == mock_response

def test_delete_model_unauthorized_error(client, requests_mock):
    """Test that delete_model raises UnauthorizedError for 401 responses"""
    model_id = "model-123"
    
    # Mock a 401 response
    requests_mock.post(
        f"{client.base_url}/model/delete",
        status_code=401,
        json={"error": "Unauthorized"}
    )
    
    with pytest.raises(UnauthorizedError):
        client.delete_model(model_id=model_id) 