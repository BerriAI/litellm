"""
Tests for Anthropic passthrough endpoint credential handling.
"""


FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing"
FAKE_API_KEY = "sk-ant-api03-fake-key-for-testing"
FAKE_SERVER_API_KEY = "sk-ant-api03-server-key"


def _get_custom_headers_for_anthropic_passthrough(
    x_api_key_header: str,
    auth_header: str,
    server_api_key: str | None,
) -> dict:
    """
    Mirrors the credential handling logic in anthropic_proxy_route.
    
    Priority:
    1. Client x-api-key -> forward as-is (return empty custom_headers)
    2. Client Authorization -> forward as-is (return empty custom_headers)
    3. No client auth -> use server credentials if configured
    4. Nothing -> return empty (let request fail at Anthropic)
    """
    if x_api_key_header or auth_header:
        return {}
    else:
        if server_api_key:
            return {"x-api-key": server_api_key}
        else:
            return {}


class TestAnthropicPassthroughCredentials:
    """
    Tests that verify credential handling priority:
    1. Client x-api-key -> forward as-is
    2. Client Authorization -> forward as-is  
    3. No client auth -> use server credentials (if configured)
    4. Nothing -> forward without credentials
    """

    def test_client_x_api_key_takes_priority(self):
        """Client x-api-key should be forwarded, server credentials not injected."""
        result = _get_custom_headers_for_anthropic_passthrough(
            x_api_key_header=FAKE_API_KEY,
            auth_header="",
            server_api_key=FAKE_SERVER_API_KEY,
        )
        assert result == {}

    def test_client_authorization_takes_priority(self):
        """Client Authorization header should be forwarded, server credentials not injected."""
        result = _get_custom_headers_for_anthropic_passthrough(
            x_api_key_header="",
            auth_header=f"Bearer {FAKE_OAUTH_TOKEN}",
            server_api_key=FAKE_SERVER_API_KEY,
        )
        assert result == {}

    def test_server_credentials_used_when_no_client_auth(self):
        """Server credentials used as fallback when client provides no auth."""
        result = _get_custom_headers_for_anthropic_passthrough(
            x_api_key_header="",
            auth_header="",
            server_api_key=FAKE_SERVER_API_KEY,
        )
        assert result == {"x-api-key": FAKE_SERVER_API_KEY}

    def test_no_credentials_injected_when_server_not_configured(self):
        """No x-api-key injected when client has no auth and server has no credentials."""
        result = _get_custom_headers_for_anthropic_passthrough(
            x_api_key_header="",
            auth_header="",
            server_api_key=None,
        )
        assert result == {}

    def test_both_client_headers_present(self):
        """When both x-api-key and Authorization present, forward as-is."""
        result = _get_custom_headers_for_anthropic_passthrough(
            x_api_key_header=FAKE_API_KEY,
            auth_header=f"Bearer {FAKE_OAUTH_TOKEN}",
            server_api_key=FAKE_SERVER_API_KEY,
        )
        assert result == {}


class TestHeaderForwardingSecurity:
    """Tests for x-litellm-api-key not being forwarded to providers."""

    def test_litellm_api_key_not_forwarded(self):
        """x-litellm-api-key should be stripped when forwarding headers."""
        from litellm.passthrough.utils import HttpPassThroughEndpointHelpers

        request_headers = {
            "x-litellm-api-key": "sk-litellm-secret-key",
            "content-type": "application/json",
            "x-api-key": "sk-ant-api-key",
        }

        result = HttpPassThroughEndpointHelpers.forward_headers_from_request(
            request_headers=request_headers.copy(),
            headers={},
            forward_headers=True,
        )

        assert "x-litellm-api-key" not in result
        assert result.get("content-type") == "application/json"
        assert result.get("x-api-key") == "sk-ant-api-key"

    def test_host_and_content_length_not_forwarded(self):
        """host and content-length should also be stripped."""
        from litellm.passthrough.utils import HttpPassThroughEndpointHelpers

        request_headers = {
            "host": "api.anthropic.com",
            "content-length": "1234",
            "content-type": "application/json",
        }

        result = HttpPassThroughEndpointHelpers.forward_headers_from_request(
            request_headers=request_headers.copy(),
            headers={},
            forward_headers=True,
        )

        assert "host" not in result
        assert "content-length" not in result
        assert result.get("content-type") == "application/json"
