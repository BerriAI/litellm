"""
Unit tests for the /sso/logout endpoint and token cookie handling.
Tests the logout functionality without requiring database connection.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
from fastapi.responses import RedirectResponse, JSONResponse


@pytest.mark.asyncio
async def test_sso_logout_clears_cookie_and_redirects_to_ui():
    """
    Test that sso_logout clears the token cookie and redirects to /ui/
    when PROXY_LOGOUT_URL is not set.
    """
    from litellm.proxy.proxy_server import sso_logout

    mock_request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/sso/logout",
            "query_string": b"",
            "headers": {},
        }
    )

    with patch.dict("os.environ", {}, clear=False):
        if "PROXY_LOGOUT_URL" in __import__("os").environ:
            del __import__("os").environ["PROXY_LOGOUT_URL"]
        response = await sso_logout(mock_request)

    assert isinstance(response, RedirectResponse)
    assert response.status_code == 303
    assert "/ui/" in response.headers["location"]

    set_cookie = response.headers.get("set-cookie", "")
    assert "token=" in set_cookie
    assert "path=/" in set_cookie.lower()


@pytest.mark.asyncio
async def test_sso_logout_redirects_to_proxy_logout_url():
    """
    Test that sso_logout redirects to PROXY_LOGOUT_URL when configured.
    """
    from litellm.proxy.proxy_server import sso_logout

    mock_request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/sso/logout",
            "query_string": b"",
            "headers": {},
        }
    )

    with patch.dict(
        "os.environ", {"PROXY_LOGOUT_URL": "https://example.com/logout"}, clear=False
    ):
        response = await sso_logout(mock_request)

    assert isinstance(response, RedirectResponse)
    assert response.status_code == 303
    assert response.headers["location"] == "https://example.com/logout"


@pytest.mark.asyncio
async def test_sso_logout_sets_secure_flag_for_https():
    """
    Test that sso_logout sets the secure flag when the request is HTTPS.
    """
    from litellm.proxy.proxy_server import sso_logout

    mock_request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "server": ("testserver", 443),
            "path": "/sso/logout",
            "query_string": b"",
            "headers": {},
        }
    )

    with patch.dict("os.environ", {}, clear=False):
        if "PROXY_LOGOUT_URL" in __import__("os").environ:
            del __import__("os").environ["PROXY_LOGOUT_URL"]
        response = await sso_logout(mock_request)

    set_cookie = response.headers.get("set-cookie", "")
    assert "secure" in set_cookie.lower()
