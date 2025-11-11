"""
Tests for public model hub endpoints.

This test suite verifies that:
1. /public/model_hub is accessible without authentication
2. /public/model_hub/info is accessible without authentication
3. Headers X-Forwarded-* are properly used for URL construction in Kubernetes environments
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.public_endpoints.public_endpoints import router
from litellm.proxy.utils import get_custom_url


def test_public_model_hub_no_auth_required():
    """
    Test that /public/model_hub can be accessed without authentication.
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    # Mock litellm.public_model_groups and llm_router
    # Note: litellm and llm_router are imported inside the function, so we patch them at the source modules
    mock_router_obj = MagicMock()
    with patch("litellm.public_model_groups", ["gpt-4", "claude-3"]), patch(
        "litellm.proxy.proxy_server._get_model_group_info"
    ) as mock_get_info, patch(
        "litellm.proxy.proxy_server.llm_router", mock_router_obj, create=True
    ):
        # Set up mocks
        mock_get_info.return_value = [
            {
                "model_group": "gpt-4",
                "providers": ["openai"],
                "max_input_tokens": 8192,
                "max_output_tokens": 4096,
            }
        ]

        # Access without authentication header
        response = client.get("/public/model_hub")

        # Should return 200, not 401 or 403
        # Note: 400 is returned if llm_router is None, 200 if mocked correctly
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)


def test_public_model_hub_info_no_auth_required():
    """
    Test that /public/model_hub/info can be accessed without authentication.
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    # Access without authentication header
    response = client.get("/public/model_hub/info")

    # Should return 200, not 401 or 403
    assert response.status_code == 200
    data = response.json()
    assert "docs_title" in data
    assert "litellm_version" in data
    assert "useful_links" in data


def test_get_custom_url_with_x_forwarded_headers():
    """
    Test that get_custom_url uses X-Forwarded-* headers when available.
    """
    # Create a mock Request with X-Forwarded-* headers
    mock_request = MagicMock(spec=Request)
    # FastAPI Request.headers is a Headers object that supports .get() method
    mock_request.headers.get = lambda key, default=None: {
        "X-Forwarded-Host": "example.com",
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Port": "443",
    }.get(key, default)
    mock_request.base_url = "http://10.244.0.1:4000"

    # Test with X-Forwarded-* headers (should use them instead of base_url)
    url = get_custom_url(
        request_base_url=str(mock_request.base_url), request=mock_request
    )

    # Should use the forwarded host, not the pod IP
    assert "example.com" in url
    assert "https" in url
    assert "10.244.0.1" not in url


def test_get_custom_url_with_x_forwarded_headers_custom_port():
    """
    Test that get_custom_url handles custom ports in X-Forwarded-Port.
    """
    # Create a mock Request with X-Forwarded-* headers and custom port
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get = lambda key, default=None: {
        "X-Forwarded-Host": "example.com",
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Port": "8443",
    }.get(key, default)
    mock_request.base_url = "http://10.244.0.1:4000"

    url = get_custom_url(
        request_base_url=str(mock_request.base_url), request=mock_request
    )

    # Should include the custom port
    assert "example.com" in url
    assert "https" in url
    assert ":8443" in url


def test_get_custom_url_fallback_to_base_url():
    """
    Test that get_custom_url falls back to request.base_url when X-Forwarded-* headers are not present.
    """
    # Create a mock Request without X-Forwarded-* headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get = lambda key, default=None: default
    mock_request.base_url = "http://localhost:4000"

    url = get_custom_url(
        request_base_url=str(mock_request.base_url), request=mock_request
    )

    # Should use the base_url
    assert "localhost" in url or "4000" in url


def test_get_custom_url_prioritizes_proxy_base_url():
    """
    Test that PROXY_BASE_URL takes priority over X-Forwarded-* headers.
    """
    # Create a mock Request with X-Forwarded-* headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get = lambda key, default=None: {
        "X-Forwarded-Host": "example.com",
        "X-Forwarded-Proto": "https",
    }.get(key, default)
    mock_request.base_url = "http://10.244.0.1:4000"

    # Set PROXY_BASE_URL environment variable
    with patch.dict(os.environ, {"PROXY_BASE_URL": "https://configured-host.com"}):
        url = get_custom_url(
            request_base_url=str(mock_request.base_url), request=mock_request
        )

        # Should use PROXY_BASE_URL, not X-Forwarded-* headers
        assert "configured-host.com" in url
        assert "example.com" not in url


def test_get_custom_url_with_route():
    """
    Test that get_custom_url correctly appends routes.
    """
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get = lambda key, default=None: {
        "X-Forwarded-Host": "example.com",
        "X-Forwarded-Proto": "https",
    }.get(key, default)
    mock_request.base_url = "http://10.244.0.1:4000"

    url = get_custom_url(
        request_base_url=str(mock_request.base_url),
        route="sso/callback",
        request=mock_request,
    )

    # Should include the route
    assert "sso/callback" in url
    assert "example.com" in url


def test_get_custom_url_without_request_object():
    """
    Test that get_custom_url works without Request object (backward compatibility).
    """
    url = get_custom_url(request_base_url="http://localhost:4000")

    # Should work without Request object
    assert url is not None
    assert isinstance(url, str)

