"""Tests for the HTTP client."""

import json
import os
import sys

import pytest
import requests

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import responses

from litellm.proxy.client.http_client import HTTPClient


@pytest.fixture
def client():
    """Create a test HTTP client."""
    return HTTPClient(
        base_url="http://localhost:4000",
        api_key="test-key",
    )


@responses.activate
def test_request_get(client):
    """Test making a GET request."""
    # Mock response
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        json={"models": []},
        status=200,
    )

    # Make request
    response = client.request("GET", "/models")

    # Check response
    assert response == {"models": []}

    # Check request
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "http://localhost:4000/models"
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-key"


@responses.activate
def test_request_post_with_json(client):
    """Test making a POST request with JSON data."""
    # Mock response
    responses.add(
        responses.POST,
        "http://localhost:4000/models",
        json={"id": "model-123"},
        status=200,
    )

    # Test data
    json_data = {"model": "gpt-4", "params": {"temperature": 0.7}}

    # Make request
    response = client.request(
        "POST",
        "/models",
        json=json_data,
    )

    # Check response
    assert response == {"id": "model-123"}

    # Check request
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "http://localhost:4000/models"
    assert json.loads(responses.calls[0].request.body) == json_data


@responses.activate
def test_request_with_custom_headers(client):
    """Test making a request with custom headers."""
    # Mock response
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        json={"models": []},
        status=200,
    )

    # Make request with custom headers
    custom_headers = {
        "X-Custom-Header": "test-value",
        "Accept": "application/json",
    }
    response = client.request(
        "GET",
        "/models",
        headers=custom_headers,
    )

    # Check request headers
    assert len(responses.calls) == 1
    request_headers = responses.calls[0].request.headers
    assert request_headers["X-Custom-Header"] == "test-value"
    assert request_headers["Accept"] == "application/json"
    assert request_headers["Authorization"] == "Bearer test-key"


@responses.activate
def test_request_http_error(client):
    """Test handling of HTTP errors."""
    # Mock error response
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        json={"error": "Not authorized"},
        status=401,
    )

    # Check that request raises exception
    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.request("GET", "/models")

    assert exc_info.value.response.status_code == 401


@responses.activate
def test_request_invalid_json(client):
    """Test handling of invalid JSON responses."""
    # Mock invalid JSON response
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        body="not json",
        status=200,
    )

    # Check that request raises exception
    with pytest.raises(requests.exceptions.JSONDecodeError) as exc_info:
        client.request("GET", "/models")


def test_base_url_trailing_slash():
    """Test that trailing slashes in base_url are handled correctly."""
    client = HTTPClient(
        base_url="http://localhost:4000/",
        api_key="test-key",
    )
    assert client._base_url == "http://localhost:4000"


def test_uri_leading_slash():
    """Test that URIs with and without leading slashes work."""
    client = HTTPClient(base_url="http://localhost:4000")

    with responses.RequestsMock() as rsps:
        # Mock endpoint
        rsps.add(
            responses.GET,
            "http://localhost:4000/models",
            json={"models": []},
        )

        # Both of these should work and hit the same endpoint
        client.request("GET", "/models")
        client.request("GET", "models")

        # Check that both requests went to the same URL
        assert len(rsps.calls) == 2
        assert rsps.calls[0].request.url == "http://localhost:4000/models"
        assert rsps.calls[1].request.url == "http://localhost:4000/models"
