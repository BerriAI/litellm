import os
import sys
import traceback

import pytest
import requests

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import responses

from litellm.proxy.client.exceptions import NotFoundError, UnauthorizedError
from litellm.proxy.client.keys import KeysManagementClient


@pytest.fixture
def base_url():
    return "http://localhost:8000"


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def client(base_url, api_key):
    return KeysManagementClient(base_url=base_url, api_key=api_key)


def test_client_initialization(base_url, api_key):
    """Test that the KeysManagementClient is properly initialized"""
    client = KeysManagementClient(base_url=base_url, api_key=api_key)

    assert client._base_url == base_url
    assert client._api_key == api_key


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    base_url = "http://localhost:8000/////"
    client = KeysManagementClient(base_url=base_url)

    assert client._base_url == "http://localhost:8000"


def test_client_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = KeysManagementClient(base_url=base_url)

    assert client._api_key is None


def test_list_request_minimal(client, base_url, api_key):
    """Test list request with minimal parameters"""
    request = client.list(return_request=True)

    assert request.method == "GET"
    assert request.url == f"{base_url}/key/list"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"
    assert not request.params


def test_list_request_pagination(client):
    """Test list request with pagination parameters"""
    request = client.list(page=2, size=10, return_request=True)

    assert request.params == {"page": 2, "size": 10}


def test_list_request_filters(client):
    """Test list request with filtering parameters"""
    request = client.list(
        user_id="user123",
        team_id="team456",
        organization_id="org789",
        key_hash="hash123",
        key_alias="alias123",
        return_request=True,
    )

    assert request.params == {
        "user_id": "user123",
        "team_id": "team456",
        "organization_id": "org789",
        "key_hash": "hash123",
        "key_alias": "alias123",
    }


def test_list_request_flags(client):
    """Test list request with boolean flag parameters"""
    request = client.list(
        return_full_object=True, include_team_keys=False, return_request=True
    )

    assert request.params == {
        "return_full_object": "true",
        "include_team_keys": "false",
    }


def test_list_request_all_parameters(client):
    """Test list request with all parameters"""
    request = client.list(
        page=2,
        size=10,
        user_id="user123",
        team_id="team456",
        organization_id="org789",
        key_hash="hash123",
        key_alias="alias123",
        return_full_object=True,
        include_team_keys=False,
        return_request=True,
    )

    assert request.params == {
        "page": 2,
        "size": 10,
        "user_id": "user123",
        "team_id": "team456",
        "organization_id": "org789",
        "key_hash": "hash123",
        "key_alias": "alias123",
        "return_full_object": "true",
        "include_team_keys": "false",
    }


@responses.activate
def test_list_mock_response_pagination(client):
    """Test list with a mocked paginated response"""
    mock_response = {
        "data": {
            "keys": [
                {
                    "key": "key1",
                    "expires": "2024-12-31T23:59:59Z",
                    "models": ["gpt-4"],
                    "aliases": {"gpt4": "gpt-4"},
                    "spend": 100.0,
                },
                {
                    "key": "key2",
                    "expires": None,
                    "models": ["gpt-3.5-turbo"],
                    "aliases": {},
                    "spend": None,
                },
            ],
            "total": 5,
            "page": 1,
            "size": 2,
        }
    }

    responses.add(
        responses.GET,
        f"{client._base_url}/key/list?page=1&size=2",
        json=mock_response,
        status=200,
    )

    response = client.list(page=1, size=2)
    assert response == mock_response


@responses.activate
def test_list_mock_response_filtered(client):
    """Test list with a mocked filtered response"""
    mock_response = {
        "keys": [
            {
                "key": "key1",
                "user_id": "user123",
                "team_id": "team456",
                "expires": "2024-12-31T23:59:59Z",
                "models": ["gpt-4"],
                "aliases": {"gpt4": "gpt-4"},
                "spend": 100.0,
            }
        ]
    }

    responses.add(
        responses.GET,
        f"{client._base_url}/key/list?user_id=user123&team_id=team456",
        json=mock_response,
        status=200,
    )

    response = client.list(user_id="user123", team_id="team456")
    assert response == mock_response


@responses.activate
def test_list_unauthorized_error(client):
    """Test that list raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.GET,
        f"{client._base_url}/key/list",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.list()


def test_generate_request_minimal(client, base_url, api_key):
    """Test generate with minimal parameters"""
    request = client.generate(return_request=True)

    assert request.method == "POST"
    assert request.url == f"{base_url}/key/generate"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"


def test_generate_request_full(client):
    """Test generate with all parameters"""
    request = client.generate(
        models=["gpt-4", "gpt-3.5-turbo"],
        aliases={"gpt4": "gpt-4", "turbo": "gpt-3.5-turbo"},
        spend=100.0,
        duration="24h",
        key_alias="test-key-alias",
        team_id="team123",
        user_id="user456",
        budget_id="budget789",
        config={"max_parallel_requests": 5},
        return_request=True,
    )

    assert request.json == {
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "aliases": {"gpt4": "gpt-4", "turbo": "gpt-3.5-turbo"},
        "spend": 100.0,
        "duration": "24h",
        "key_alias": "test-key-alias",
        "team_id": "team123",
        "user_id": "user456",
        "budget_id": "budget789",
        "config": {"max_parallel_requests": 5},
    }


@responses.activate
def test_generate_mock_response(client):
    """Test generate with a mocked successful response"""
    mock_response = {
        "key": "new-test-key",
        "expires": "2024-12-31T23:59:59Z",
        "models": ["gpt-4"],
        "aliases": {"gpt4": "gpt-4"},
        "spend": 100.0,
        "key_alias": "test-key-alias",
        "team_id": "team123",
        "user_id": "user456",
        "budget_id": "budget789",
        "config": {"max_parallel_requests": 5},
    }

    responses.add(
        responses.POST,
        f"{client._base_url}/key/generate",
        json=mock_response,
        status=200,
    )

    response = client.generate(
        key_alias="test-key-alias",
        team_id="team123",
        user_id="user456",
        budget_id="budget789",
        config={"max_parallel_requests": 5},
    )
    assert response == mock_response


@responses.activate
def test_generate_unauthorized_error(client):
    """Test that generate raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.POST,
        f"{client._base_url}/key/generate",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.generate()


def test_delete_request_minimal(client, base_url, api_key):
    """Test delete request with minimal parameters"""
    request = client.delete(return_request=True)

    assert request.method == "POST"
    assert request.url == f"{base_url}/key/delete"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"
    assert request.json == {"keys": None, "key_aliases": None}


def test_delete_request_with_keys(client):
    """Test delete request with keys list"""
    keys_to_delete = ["key1", "key2", "key3"]
    request = client.delete(keys=keys_to_delete, return_request=True)

    assert request.json == {"keys": keys_to_delete, "key_aliases": None}


def test_delete_request_with_aliases(client):
    """Test delete request with key aliases list"""
    aliases_to_delete = ["alias1", "alias2"]
    request = client.delete(key_aliases=aliases_to_delete, return_request=True)

    assert request.json == {"keys": None, "key_aliases": aliases_to_delete}


def test_delete_request_with_keys_and_aliases(client):
    """Test delete request with both keys and aliases"""
    keys_to_delete = ["key1", "key2"]
    aliases_to_delete = ["alias1", "alias2"]
    request = client.delete(
        keys=keys_to_delete, key_aliases=aliases_to_delete, return_request=True
    )

    assert request.json == {"keys": keys_to_delete, "key_aliases": aliases_to_delete}


@responses.activate
def test_delete_mock_response(client):
    """Test delete with a mocked successful response"""
    mock_response = {
        "status": "success",
        "deleted_keys": ["key1", "key2"],
        "deleted_aliases": ["alias1"],
    }
    responses.add(
        responses.POST,
        f"{client._base_url}/key/delete",
        json=mock_response,
        status=200,
    )

    response = client.delete(keys=["key1", "key2"], key_aliases=["alias1"])
    assert response == mock_response


@responses.activate
def test_delete_unauthorized_error(client):
    """Test that delete raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.POST,
        f"{client._base_url}/key/delete",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.delete(keys=["key-to-delete"])


def test_info_request_minimal(client, base_url, api_key):
    """Test info request with minimal parameters"""
    request = client.info(key="test-key", return_request=True)
    assert request.method == "GET"
    assert request.url == f"{base_url}/key/info?key=test-key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"


@responses.activate
def test_info_mock_response(client):
    """Test info with a mocked successful response"""
    mock_response = {
        "key": "test-key",
        "user_id": "user123",
        "team_id": "team456",
        "models": ["gpt-4"],
        "spend": 100.0,
    }
    responses.add(
        responses.GET,
        f"{client._base_url}/key/info?key=test-key",
        json=mock_response,
        status=200,
    )
    response = client.info(key="test-key")
    assert response == mock_response


@responses.activate
def test_info_unauthorized_error(client):
    """Test that info raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.GET,
        f"{client._base_url}/key/info?key=test-key",
        status=401,
        json={"error": "Unauthorized"},
    )
    with pytest.raises(UnauthorizedError):
        client.info(key="test-key")


@responses.activate
def test_info_server_error(client):
    """Test that info raises HTTPError for server errors"""
    responses.add(
        responses.GET,
        f"{client._base_url}/key/info?key=test-key",
        status=500,
        json={"error": "Internal Server Error"},
    )
    with pytest.raises(requests.exceptions.HTTPError):
        client.info(key="test-key")


LEAKY_KEY = "sk-1234567890abcdefghijklmnop"


def _render_full_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


@responses.activate
def test_info_not_found_redacts_key_everywhere(client):
    """A 404 must not echo the raw key embedded in the request URL.

    Covers str(exc) and the rendered traceback, since the chain through
    __cause__ / __context__ is what logging.exception() and the default
    excepthook print.
    """
    responses.add(
        responses.GET,
        f"{client._base_url}/key/info?key={LEAKY_KEY}",
        status=404,
        json={"error": {"message": "Key not found", "code": "404"}},
    )
    with pytest.raises(requests.exceptions.HTTPError) as excinfo:
        client.info(key=LEAKY_KEY)

    exc = excinfo.value
    assert LEAKY_KEY not in str(exc)
    assert "REDACTED" in str(exc)
    assert LEAKY_KEY not in _render_full_traceback(exc)
    assert exc.__cause__ is None and exc.__suppress_context__
    assert exc.response is not None
    assert exc.response.status_code == 404
    assert exc.request is not None
    # Known residual: the live request URL still carries the key, since the
    # response is preserved so callers keep status_code / text. str(exc) and the
    # traceback are scrubbed; the URL-borne key is the root issue tracked in
    # LIT-4013 (move the lookup key out of the query string server-side).
    assert LEAKY_KEY in exc.response.request.url


@responses.activate
def test_info_unauthorized_redacts_key_everywhere(client):
    """A 401 surfaced as UnauthorizedError must not echo the raw key in the
    message, the retained original, or the rendered traceback chain."""
    responses.add(
        responses.GET,
        f"{client._base_url}/key/info?key={LEAKY_KEY}",
        status=401,
        json={"error": "Unauthorized"},
    )
    with pytest.raises(UnauthorizedError) as excinfo:
        client.info(key=LEAKY_KEY)

    exc = excinfo.value
    assert LEAKY_KEY not in str(exc)
    assert "REDACTED" in str(exc)
    assert LEAKY_KEY not in str(exc.orig_exception)
    assert LEAKY_KEY not in _render_full_traceback(exc)
    assert exc.__cause__ is None and exc.__suppress_context__
    assert isinstance(exc.orig_exception, requests.exceptions.HTTPError)
    assert exc.orig_exception.response is not None
    assert exc.orig_exception.response.status_code == 401


def _http_error_with_key(prefix: str, status: int) -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.exceptions.HTTPError(
        f"{prefix} for url: http://x/key/info?key={LEAKY_KEY}", response=resp
    )


def test_unauthorized_error_redacts_wrapped_key():
    """UnauthorizedError scrubs the key in str(exc) and in the retained
    orig_exception, while preserving the response for structured access."""
    wrapped = UnauthorizedError(
        _http_error_with_key("401 Client Error: Unauthorized", 401)
    )
    assert LEAKY_KEY not in str(wrapped)
    assert "REDACTED" in str(wrapped)
    assert LEAKY_KEY not in str(wrapped.orig_exception)
    assert wrapped.orig_exception.response.status_code == 401


def test_not_found_error_redacts_wrapped_key():
    """NotFoundError scrubs the key in str(exc) and in the retained
    orig_exception, while preserving the response for structured access."""
    wrapped = NotFoundError(_http_error_with_key("404 Client Error: Not Found", 404))
    assert LEAKY_KEY not in str(wrapped)
    assert "REDACTED" in str(wrapped)
    assert LEAKY_KEY not in str(wrapped.orig_exception)
    assert wrapped.orig_exception.response.status_code == 404
