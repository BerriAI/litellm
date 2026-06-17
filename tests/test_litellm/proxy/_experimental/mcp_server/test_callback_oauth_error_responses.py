"""Regression tests for LIT-2750.

The MCP OAuth ``/callback`` endpoint must handle IdP error responses
(e.g. ``?error=access_denied``) gracefully instead of returning a 422
because ``code`` and ``state`` were declared as required FastAPI query
params. Per RFC 6749 §4.1.2.1 the IdP redirects to the configured
redirect URI with ``error`` / ``error_description`` / ``error_uri``
query params and no ``code`` when the user denies access.

These tests cover both the propagate-to-client path (when state decodes
to a trusted ``redirect_uri``) and the in-page fallback (when state is
missing, undecryptable, or carries an untrusted redirect_uri). They also
pin the success path (``code`` + ``state``) against accidental
regressions.
"""

import pytest


@pytest.fixture(autouse=True)
def _mock_mcp_client_ip():
    """Bypass IP-based access control for the in-process TestClient.

    Mirrors the autouse fixture in ``test_discoverable_endpoints.py`` so
    these tests don't require a real client IP context.
    """
    from unittest.mock import patch

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
        return_value=None,
    ):
        yield


@pytest.fixture
def callback_test_client(monkeypatch):
    """FastAPI TestClient mounted with the MCP discoverable router.

    Sets a deterministic ``LITELLM_SALT_KEY`` so encoded states minted
    in-test can be decrypted by the handler.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-for-LIT-2750")

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        router,
    )

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestCallbackOAuthErrorResponses:
    """LIT-2750: IdP error responses to ``/callback`` must not 422."""

    def test_idp_error_with_no_state_returns_400_html(self, callback_test_client):
        """Pre-fix: 422 Pydantic. Post-fix: 400 HTML with the IdP's error."""
        resp = callback_test_client.get(
            "/callback",
            params={
                "error": "access_denied",
                "error_description": "User declined access",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 400
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "access_denied" in body
        assert "User declined access" in body
        # Sanity: must not leak the Pydantic validation error.
        assert "Field required" not in body

    def test_idp_error_html_escapes_user_controlled_fields(
        self, callback_test_client
    ):
        """A malicious IdP must not be able to inject HTML/JS via error params."""
        resp = callback_test_client.get(
            "/callback",
            params={
                "error": "<script>alert(1)</script>",
                "error_description": "<img src=x onerror=alert(2)>",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 400
        body = resp.text
        # Raw tags must be escaped, not present verbatim.
        assert "<script>alert(1)</script>" not in body
        assert "<img src=x onerror=alert(2)>" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body

    def test_idp_error_with_trusted_state_propagates_to_client_redirect_uri(
        self, callback_test_client
    ):
        """When state decodes to a trusted (loopback) redirect_uri, propagate
        the error back so the MCP client's OAuth library can surface it
        instead of timing out waiting on the loopback."""
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            encode_state_with_base_url,
        )

        state = encode_state_with_base_url(
            base_url="http://localhost:3000/",
            original_state="client-original-state-xyz",
            client_redirect_uri="http://127.0.0.1:60108/callback",
        )

        resp = callback_test_client.get(
            "/callback",
            params={
                "error": "access_denied",
                "error_description": "User declined access",
                "state": state,
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302
        location = resp.headers["location"]
        assert location.startswith("http://127.0.0.1:60108/callback?")
        assert "error=access_denied" in location
        # Original client state must be round-tripped, not our wrapped state.
        assert "state=client-original-state-xyz" in location
        # error_description percent-encoded but present.
        assert "error_description=User" in location
        # Wrapped/encrypted state must NOT leak to the client.
        assert state not in location

    def test_idp_error_with_untrusted_redirect_uri_does_not_open_redirect(
        self, callback_test_client
    ):
        """If the state minted earlier carries a redirect_uri that the proxy
        no longer trusts, we must surface the error inline rather than
        302-ing to an attacker-controlled URL (open-redirect)."""
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            encode_state_with_base_url,
        )

        state = encode_state_with_base_url(
            base_url="http://localhost:3000/",
            original_state="x",
            client_redirect_uri="https://attacker.example.com/steal",
        )

        resp = callback_test_client.get(
            "/callback",
            params={"error": "access_denied", "state": state},
            follow_redirects=False,
        )
        # Must not 3xx — open redirect would defeat the redirect_uri allowlist.
        assert resp.status_code == 400
        assert "attacker.example.com" not in resp.headers.get("location", "")
        assert "access_denied" in resp.text

    def test_idp_error_with_undecryptable_state_falls_back_to_html(
        self, callback_test_client
    ):
        resp = callback_test_client.get(
            "/callback",
            params={
                "error": "server_error",
                "error_description": "boom",
                "state": "not-a-valid-encrypted-state",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 400
        assert "server_error" in resp.text
        assert "boom" in resp.text

    def test_bare_callback_with_no_params_returns_400_not_422(
        self, callback_test_client
    ):
        """An SSO redirect chain that drops the original /authorize query
        params should land on a human-readable 400, not a Pydantic 422."""
        resp = callback_test_client.get("/callback", follow_redirects=False)
        assert resp.status_code == 400
        assert "invalid_request" in resp.text
        assert "Field required" not in resp.text

    def test_success_path_still_redirects_with_code_and_state(
        self, callback_test_client
    ):
        """Regression: the successful (``code``+``state``) flow must still
        redirect back to the trusted client redirect_uri with the original
        state preserved."""
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            encode_state_with_base_url,
        )

        state = encode_state_with_base_url(
            base_url="http://localhost:3000/",
            original_state="orig-state-success",
            client_redirect_uri="http://127.0.0.1:60108/callback",
        )

        resp = callback_test_client.get(
            "/callback",
            params={"code": "auth-code-abc", "state": state},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        location = resp.headers["location"]
        assert location.startswith("http://127.0.0.1:60108/callback?")
        assert "code=auth-code-abc" in location
        assert "state=orig-state-success" in location
