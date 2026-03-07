"""Unit tests for openapi_oauth2_endpoints.py"""

import json
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "../../../../../")

pytest.importorskip("mcp", reason="mcp package not installed; skipping MCP OAuth2 tests")

from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
    _make_state_token,
    _pending_oauth2_states,
    _purge_expired_states,
)

# ---------------------------------------------------------------------------
# _make_state_token
# ---------------------------------------------------------------------------


def test_make_state_token_returns_string():
    token = _make_state_token("server1", "user1", time.time(), "master-key")
    assert isinstance(token, str)
    assert len(token) > 0


def test_make_state_token_is_unique_for_same_inputs():
    """Two calls with identical inputs must produce different tokens (random nonce)."""
    ts = time.time()
    t1 = _make_state_token("server1", "user1", ts, "master-key")
    t2 = _make_state_token("server1", "user1", ts, "master-key")
    assert t1 != t2, "Tokens should differ due to random nonce"


def test_make_state_token_differs_by_server():
    ts = time.time()
    t1 = _make_state_token("server1", "user1", ts, "master-key")
    t2 = _make_state_token("server2", "user1", ts, "master-key")
    assert t1 != t2


# ---------------------------------------------------------------------------
# _purge_expired_states
# ---------------------------------------------------------------------------


def test_purge_expired_states_removes_expired():
    _pending_oauth2_states.clear()
    now = time.time()
    _pending_oauth2_states["expired"] = {"expires_at": now - 1}
    _pending_oauth2_states["valid"] = {"expires_at": now + 600}
    _purge_expired_states()
    assert "expired" not in _pending_oauth2_states
    assert "valid" in _pending_oauth2_states
    _pending_oauth2_states.clear()


# ---------------------------------------------------------------------------
# openapi_oauth2_connect — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_missing_server_raises_404():
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_connect,
    )

    mock_request = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user1"
    mock_user.api_key = None

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr:
        mock_mgr.get_mcp_server_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await openapi_oauth2_connect("nonexistent", mock_request, mock_user)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_connect_missing_client_secret_raises_400():
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_connect,
    )

    mock_request = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user1"
    mock_user.api_key = None

    mock_server = MagicMock()
    mock_server.authorization_url = "https://provider.example/auth"
    mock_server.token_url = "https://provider.example/token"
    mock_server.client_id = "my-client-id"
    mock_server.client_secret = None  # missing

    # master_key is imported inline via `from litellm.proxy.proxy_server import master_key`;
    # patch at the source so the function sees the mock value.
    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy.proxy_server.master_key",
        "sk-test",
        create=True,
    ):
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        with pytest.raises(HTTPException) as exc_info:
            await openapi_oauth2_connect("server1", mock_request, mock_user)
        assert exc_info.value.status_code == 400
        assert "client_secret" in exc_info.value.detail


# ---------------------------------------------------------------------------
# openapi_oauth2_callback — token parsing (provider error body)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_provider_error_in_json_body():
    """HTTP 200 with JSON error body should render error page, not succeed."""
    from fastapi.responses import HTMLResponse

    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_callback,
    )

    state = "test-state-123"
    now = time.time()
    _pending_oauth2_states[state] = {
        "server_id": "server1",
        "user_id": "user1",
        "timestamp": now,
        "expires_at": now + 600,
    }

    mock_server = MagicMock()
    mock_server.token_url = "https://provider.example/token"
    mock_server.client_id = "cid"
    mock_server.client_secret = "csecret"

    # Simulate provider returning HTTP 200 with an error body
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "error": "invalid_grant",
        "error_description": "Code already used",
    }
    mock_response.raise_for_status = MagicMock()  # does not raise

    mock_request = MagicMock()
    mock_request.base_url = "http://localhost:4000"

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.get_request_base_url",
        return_value="http://localhost:4000",
    ), patch(
        "httpx.AsyncClient"
    ) as mock_client_cls:
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        mock_async_client = AsyncMock()
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await openapi_oauth2_callback(
            request=mock_request,
            code="auth-code",
            state=state,
            error=None,
            error_description=None,
        )

    assert isinstance(result, HTMLResponse)
    assert result.status_code == 502
    body_bytes = bytes(result.body) if not isinstance(result.body, bytes) else result.body
    assert b"invalid_grant" in body_bytes or b"provider" in body_bytes.lower()


# ---------------------------------------------------------------------------
# openapi_oauth2_status — basic happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_no_prisma_returns_not_connected():
    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_status,
    )

    mock_user = MagicMock()
    mock_user.user_id = "user1"
    mock_user.api_key = None

    mock_server = MagicMock()
    mock_server.server_name = "GitHub"
    mock_server.name = "github"

    # prisma_client is imported inside the function via
    # `from litellm.proxy.proxy_server import prisma_client`, so we must patch
    # it at the source module rather than on the endpoint module.
    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy.proxy_server.prisma_client",
        None,
        create=True,
    ):
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        result = await openapi_oauth2_status("server1", mock_user)

    raw = result.body
    body = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw))
    assert body["connected"] is False
    assert body["server_id"] == "server1"


# ---------------------------------------------------------------------------
# Refresh token — stored as JSON blob when provider returns one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_stores_refresh_token_as_json():
    """When the provider returns a refresh_token, it is stored as a JSON blob."""
    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_callback,
    )

    state = "test-state-refresh"
    now = time.time()
    _pending_oauth2_states[state] = {
        "server_id": "server1",
        "user_id": "user1",
        "timestamp": now,
        "expires_at": now + 600,
    }

    mock_server = MagicMock()
    mock_server.token_url = "https://provider.example/token"
    mock_server.client_id = "cid"
    mock_server.client_secret = "csecret"
    mock_server.server_name = "TestProvider"
    mock_server.name = "test"

    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "access_token": "ghu_accesstoken123",
        "refresh_token": "ghr_refreshtoken456",
        "token_type": "bearer",
    }
    mock_response.raise_for_status = MagicMock()

    stored_credentials: list = []

    async def fake_store(prisma_client, user_id, server_id, credential):
        stored_credentials.append(credential)

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.get_request_base_url",
        return_value="http://localhost:4000",
    ), patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.store_user_credential",
        side_effect=fake_store,
    ), patch(
        "litellm.proxy.proxy_server.prisma_client",
        MagicMock(),
        create=True,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._invalidate_byok_cred_cache",
        MagicMock(),
    ), patch(
        "httpx.AsyncClient"
    ) as mock_client_cls:
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        mock_async_client = AsyncMock()
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        await openapi_oauth2_callback(
            request=MagicMock(),
            code="auth-code",
            state=state,
            error=None,
            error_description=None,
        )

    assert len(stored_credentials) == 1
    stored = stored_credentials[0]
    parsed = json.loads(stored)
    assert parsed["access_token"] == "ghu_accesstoken123"
    assert parsed["refresh_token"] == "ghr_refreshtoken456"


@pytest.mark.asyncio
async def test_callback_stores_plain_token_when_no_refresh_token():
    """When the provider does not return a refresh_token, the plain access_token is stored."""
    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_callback,
    )

    state = "test-state-no-refresh"
    now = time.time()
    _pending_oauth2_states[state] = {
        "server_id": "server1",
        "user_id": "user1",
        "timestamp": now,
        "expires_at": now + 600,
    }

    mock_server = MagicMock()
    mock_server.token_url = "https://provider.example/token"
    mock_server.client_id = "cid"
    mock_server.client_secret = "csecret"
    mock_server.server_name = "TestProvider"
    mock_server.name = "test"

    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "access_token": "ghu_only_access",
        "token_type": "bearer",
    }
    mock_response.raise_for_status = MagicMock()

    stored_credentials: list = []

    async def fake_store(prisma_client, user_id, server_id, credential):
        stored_credentials.append(credential)

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.get_request_base_url",
        return_value="http://localhost:4000",
    ), patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.store_user_credential",
        side_effect=fake_store,
    ), patch(
        "litellm.proxy.proxy_server.prisma_client",
        MagicMock(),
        create=True,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._invalidate_byok_cred_cache",
        MagicMock(),
    ), patch(
        "httpx.AsyncClient"
    ) as mock_client_cls:
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        mock_async_client = AsyncMock()
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        await openapi_oauth2_callback(
            request=MagicMock(),
            code="auth-code",
            state=state,
            error=None,
            error_description=None,
        )

    assert len(stored_credentials) == 1
    # Plain string — NOT a JSON blob
    assert stored_credentials[0] == "ghu_only_access"


# ---------------------------------------------------------------------------
# _extract_access_token (server.py helper)
# ---------------------------------------------------------------------------


def test_extract_access_token_plain_string():
    """Plain token strings are returned unchanged."""
    from litellm.proxy._experimental.mcp_server.server import _extract_access_token

    assert _extract_access_token("ghp_plaintoken") == "ghp_plaintoken"


def test_extract_access_token_json_blob():
    """JSON blob with access_token + refresh_token → access_token returned."""
    from litellm.proxy._experimental.mcp_server.server import _extract_access_token

    blob = json.dumps({"access_token": "ghu_access", "refresh_token": "ghr_refresh"})
    assert _extract_access_token(blob) == "ghu_access"


def test_extract_access_token_none():
    """None input returns None."""
    from litellm.proxy._experimental.mcp_server.server import _extract_access_token

    assert _extract_access_token(None) is None


def test_extract_access_token_json_without_access_token_key():
    """JSON object without 'access_token' key is returned as-is (treated as plain string)."""
    from litellm.proxy._experimental.mcp_server.server import _extract_access_token

    blob = json.dumps({"some_other_key": "value"})
    # Falls back to raw string since there's no access_token
    assert _extract_access_token(blob) == blob


# ---------------------------------------------------------------------------
# _write_byok_cred_cache — LRU eviction (regression for thundering-herd bug)
# ---------------------------------------------------------------------------


def test_write_byok_cred_cache_evicts_single_oldest_entry():
    """At capacity, _write_byok_cred_cache must evict one (oldest) entry, not clear all."""
    from litellm.proxy._experimental.mcp_server.server import (
        _BYOK_CRED_CACHE_MAX_SIZE,
        _byok_cred_cache,
        _write_byok_cred_cache,
    )

    _byok_cred_cache.clear()

    # Fill to capacity
    for i in range(_BYOK_CRED_CACHE_MAX_SIZE):
        _byok_cred_cache[(f"user{i}", "server")] = ("token", 0.0)

    assert len(_byok_cred_cache) == _BYOK_CRED_CACHE_MAX_SIZE

    # Writing one more entry should evict the oldest, not clear everything
    _write_byok_cred_cache("new_user", "server", "new_token")

    # Cache should still be at max size — only ONE entry was evicted
    assert len(_byok_cred_cache) == _BYOK_CRED_CACHE_MAX_SIZE

    # The new entry must be present
    assert ("new_user", "server") in _byok_cred_cache

    # The oldest entry (user0) should have been evicted
    assert ("user0", "server") not in _byok_cred_cache

    _byok_cred_cache.clear()


# ---------------------------------------------------------------------------
# Regression tests for the three bug fixes in c1fcbf6219
# ---------------------------------------------------------------------------


def test_mcp_server_byok_fields_propagated():
    """Bug fix 1 (related): MCPServer must accept and preserve is_byok,
    byok_description, and byok_api_key_help_url (previously missing from
    load_servers_from_config which caused BYOK OAuth2 servers to lose these
    fields after config load).
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="github",
        name="github",
        server_name="GitHub",
        transport=MCPTransport.sse,
        is_byok=True,
        byok_description=["Read your profile", "List your repos"],
        byok_api_key_help_url="https://github.com/settings/tokens",
    )

    assert server.is_byok is True
    assert server.byok_description == ["Read your profile", "List your repos"]
    assert server.byok_api_key_help_url == "https://github.com/settings/tokens"


def test_mcp_server_byok_fields_default_to_empty():
    """Regression: byok fields default to False/[] (not None) so they don't
    need null-guards in code that iterates byok_description.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="plain",
        name="plain",
        server_name="Plain Server",
        transport=MCPTransport.http,
    )

    assert server.is_byok is False
    assert isinstance(server.byok_description, list)
