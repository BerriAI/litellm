"""
Unit tests for per-user MCP OAuth token storage:
- MCPPerUserTokenCache (NaCl-encrypted Redis cache)
- _validate_token_response (token validation rules)
- _compute_per_user_token_ttl (TTL computation)
- refresh_user_oauth_token (token refresh flow)
"""

import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub out modules that aren't available in the unit-test environment
# so we can import the targets without a full proxy stack.
for _mod in ("orjson",):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (  # noqa: E402
    MCPPerUserTokenCache,
    _compute_per_user_token_ttl,
    mcp_per_user_token_cache,
)
from litellm.types.mcp import MCPAuth, MCPTransport  # noqa: E402
from litellm.types.mcp_server.mcp_server_manager import MCPServer  # noqa: E402


def _import_validate():
    """Lazy import to avoid pulling orjson at collection time."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _validate_token_response,
    )

    return _validate_token_response


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_server(**kwargs) -> MCPServer:
    defaults: Dict[str, Any] = {
        "server_id": "slack-test",
        "name": "Slack",
        "server_name": "slack",
        "url": "https://slack-mcp.example.com/mcp",
        "transport": MCPTransport.http,
        "auth_type": MCPAuth.oauth2,
        "client_id": "SLACK_CLIENT_ID",
        "client_secret": "SLACK_CLIENT_SECRET",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "authorization_url": "https://slack.com/oauth/v2/authorize",
    }
    defaults.update(kwargs)
    return MCPServer(**defaults)


# ── _validate_token_response ──────────────────────────────────────────────────


class TestValidateTokenResponse:
    def test_passes_when_all_rules_match(self):
        _validate_token_response = _import_validate()
        token_response = {
            "access_token": "xoxb-123",
            "enterprise_id": "E04XXXXXX",
            "team": {"id": "T123", "name": "Acme"},
        }
        # Should not raise
        _validate_token_response(
            token_response=token_response,
            validation_rules={"enterprise_id": "E04XXXXXX"},
            server_id="slack-test",
        )

    def test_raises_on_mismatch(self):
        from fastapi import HTTPException

        _validate_token_response = _import_validate()
        token_response = {"access_token": "xoxb-123", "enterprise_id": "E99999999"}
        with pytest.raises(HTTPException) as exc_info:
            _validate_token_response(
                token_response=token_response,
                validation_rules={"enterprise_id": "E04XXXXXX"},
                server_id="slack-test",
            )
        assert exc_info.value.status_code == 403
        detail = exc_info.value.detail
        assert detail["error"] == "token_validation_failed"
        assert detail["field"] == "enterprise_id"

    def test_raises_when_field_absent(self):
        from fastapi import HTTPException

        _validate_token_response = _import_validate()
        token_response = {"access_token": "xoxb-123"}
        with pytest.raises(HTTPException) as exc_info:
            _validate_token_response(
                token_response=token_response,
                validation_rules={"enterprise_id": "E04XXXXXX"},
                server_id="slack-test",
            )
        assert exc_info.value.status_code == 403
        # Absent field should produce a distinct "absent" message, not str(None)
        assert "absent" in exc_info.value.detail["message"]

    def test_absent_field_does_not_match_string_none(self):
        """str(None)='None' must NOT match the string rule value 'None'."""
        from fastapi import HTTPException

        _validate_token_response = _import_validate()
        token_response = {"access_token": "tok"}  # enterprise_id absent
        # Even if admin writes validation_rules={"enterprise_id": "None"}, absent
        # field should raise, not pass.
        with pytest.raises(HTTPException) as exc_info:
            _validate_token_response(
                token_response=token_response,
                validation_rules={"enterprise_id": "None"},
                server_id="slack-test",
            )
        assert exc_info.value.status_code == 403
        assert "absent" in exc_info.value.detail["message"]

    def test_dot_notation_nested_field(self):
        _validate_token_response = _import_validate()
        token_response = {
            "access_token": "xoxb-123",
            "team": {"enterprise_id": "E04XXXXXX"},
        }
        # Should not raise — dot-notation traverses nested dict
        _validate_token_response(
            token_response=token_response,
            validation_rules={"team.enterprise_id": "E04XXXXXX"},
            server_id="slack-test",
        )

    def test_dot_notation_mismatch(self):
        from fastapi import HTTPException

        _validate_token_response = _import_validate()
        token_response = {
            "access_token": "xoxb-123",
            "team": {"enterprise_id": "WRONG"},
        }
        with pytest.raises(HTTPException) as exc_info:
            _validate_token_response(
                token_response=token_response,
                validation_rules={"team.enterprise_id": "E04XXXXXX"},
                server_id="slack-test",
            )
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["field"] == "team.enterprise_id"

    def test_numeric_value_string_coercion(self):
        """Numeric values in token response should match string rules."""
        _validate_token_response = _import_validate()
        token_response = {"access_token": "tok", "org_id": 12345}
        # Should not raise — str(12345) == "12345"
        _validate_token_response(
            token_response=token_response,
            validation_rules={"org_id": "12345"},
            server_id="test",
        )

    def test_multiple_rules_all_must_match(self):
        from fastapi import HTTPException

        _validate_token_response = _import_validate()
        token_response = {
            "access_token": "tok",
            "enterprise_id": "E04XXXXXX",
            "cloud_id": "WRONG_CLOUD",
        }
        with pytest.raises(HTTPException):
            _validate_token_response(
                token_response=token_response,
                validation_rules={
                    "enterprise_id": "E04XXXXXX",
                    "cloud_id": "abc-123",
                },
                server_id="atlassian",
            )


# ── _compute_per_user_token_ttl ──────────────────────────────────────────────


class TestComputePerUserTokenTtl:
    def test_uses_server_override_when_set(self):
        server = _make_server(token_storage_ttl_seconds=7200)
        assert _compute_per_user_token_ttl(server, expires_in=99999) == 7200

    def test_uses_expires_in_minus_buffer(self):
        server = _make_server()
        # Default buffer is 60s
        ttl = _compute_per_user_token_ttl(server, expires_in=3600)
        assert ttl == 3600 - 60

    def test_minimum_ttl_is_1(self):
        server = _make_server()
        # expires_in smaller than buffer → clamp to 1
        ttl = _compute_per_user_token_ttl(server, expires_in=30)
        assert ttl == 1

    def test_default_ttl_when_expires_in_none(self):
        from litellm.constants import MCP_PER_USER_TOKEN_DEFAULT_TTL

        server = _make_server()
        ttl = _compute_per_user_token_ttl(server, expires_in=None)
        assert ttl == MCP_PER_USER_TOKEN_DEFAULT_TTL


# ── MCPPerUserTokenCache ──────────────────────────────────────────────────────


class TestMCPPerUserTokenCache:
    """Tests for Redis-backed per-user token cache.

    Patches ``user_api_key_cache`` to avoid needing a real Redis instance.
    Patches ``encrypt_value_helper`` / ``decrypt_value_helper`` to verify
    encryption is applied before Redis writes and decryption after reads.
    """

    @pytest.fixture
    def cache(self):
        return MCPPerUserTokenCache()

    @pytest.fixture
    def mock_dual_cache(self):
        dc = MagicMock()
        dc.async_get_cache = AsyncMock(return_value=None)
        dc.async_set_cache = AsyncMock()
        return dc

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, cache, mock_dual_cache):
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.oauth2_token_cache.decrypt_value_helper"
            ) as mock_decrypt,
            patch("litellm.proxy.proxy_server.user_api_key_cache", mock_dual_cache),
        ):
            mock_dual_cache.async_get_cache.return_value = None
            result = await cache.get("alice", "slack-test")
        assert result is None
        mock_decrypt.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_decrypts_cached_value(self, cache, mock_dual_cache):
        fake_encrypted = "encrypted_blob_abc123"
        fake_plaintext = "xoxb-slack-token"
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.oauth2_token_cache.decrypt_value_helper",
                return_value=fake_plaintext,
            ) as mock_decrypt,
            patch("litellm.proxy.proxy_server.user_api_key_cache", mock_dual_cache),
        ):
            mock_dual_cache.async_get_cache.return_value = fake_encrypted
            result = await cache.get("alice", "slack-test")

        assert result == fake_plaintext
        mock_decrypt.assert_called_once_with(
            fake_encrypted,
            key="mcp_per_user_token",
            exception_type="debug",
        )

    @pytest.mark.asyncio
    async def test_set_encrypts_before_storing(self, cache, mock_dual_cache):
        fake_encrypted = "encrypted_blob_xyz"
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.oauth2_token_cache.encrypt_value_helper",
                return_value=fake_encrypted,
            ) as mock_encrypt,
            patch("litellm.proxy.proxy_server.user_api_key_cache", mock_dual_cache),
        ):
            await cache.set("alice", "slack-test", "xoxb-token", ttl=3540)

        mock_encrypt.assert_called_once_with("xoxb-token")
        mock_dual_cache.async_set_cache.assert_called_once()
        call_kwargs = mock_dual_cache.async_set_cache.call_args
        assert call_kwargs[0][1] == fake_encrypted  # encrypted value stored
        assert call_kwargs[1]["ttl"] == 3540

    @pytest.mark.asyncio
    async def test_set_uses_correct_cache_key(self, cache, mock_dual_cache):
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.oauth2_token_cache.encrypt_value_helper",
                return_value="enc",
            ),
            patch("litellm.proxy.proxy_server.user_api_key_cache", mock_dual_cache),
        ):
            await cache.set("bob", "github-server", "ghp_token", ttl=3600)

        key_used = mock_dual_cache.async_set_cache.call_args[0][0]
        assert key_used == "mcp:per_user_token:bob:github-server"

    @pytest.mark.asyncio
    async def test_delete_calls_async_delete_cache(self, cache, mock_dual_cache):
        mock_dual_cache.async_delete_cache = AsyncMock()
        with patch("litellm.proxy.proxy_server.user_api_key_cache", mock_dual_cache):
            await cache.delete("alice", "slack-test")

        mock_dual_cache.async_delete_cache.assert_called_once_with(
            "mcp:per_user_token:alice:slack-test"
        )
        mock_dual_cache.async_set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_returns_none_on_decrypt_failure(self, cache, mock_dual_cache):
        """Cache misses and decrypt errors should both return None without raising."""
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.oauth2_token_cache.decrypt_value_helper",
                return_value=None,  # decrypt returns None on failure
            ),
            patch("litellm.proxy.proxy_server.user_api_key_cache", mock_dual_cache),
        ):
            mock_dual_cache.async_get_cache.return_value = "bad_encrypted_data"
            result = await cache.get("alice", "slack-test")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_is_noop_on_cache_error(self, cache, mock_dual_cache):
        """Errors in the cache layer must not propagate to the caller."""
        mock_dual_cache.async_set_cache.side_effect = RuntimeError("Redis down")
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.oauth2_token_cache.encrypt_value_helper",
                return_value="enc",
            ),
            patch("litellm.proxy.proxy_server.user_api_key_cache", mock_dual_cache),
        ):
            # Should not raise
            await cache.set("alice", "slack-test", "token", ttl=3600)


# ── refresh_user_oauth_token ──────────────────────────────────────────────────


class TestRefreshUserOauthToken:
    """Tests for the DB-level token refresh helper."""

    @pytest.fixture
    def server(self):
        return _make_server()

    @pytest.fixture
    def cred(self):
        return {
            "type": "oauth2",
            "access_token": "OLD_TOKEN",
            "refresh_token": "REFRESH_TOKEN_123",
            "expires_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        }

    @pytest.mark.asyncio
    async def test_returns_none_when_no_refresh_token(self, server):
        from litellm.proxy._experimental.mcp_server.db import refresh_user_oauth_token

        cred = {"type": "oauth2", "access_token": "OLD"}  # no refresh_token
        result = await refresh_user_oauth_token(
            prisma_client=MagicMock(),
            user_id="alice",
            server=server,
            cred=cred,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_token_url(self, cred):
        from litellm.proxy._experimental.mcp_server.db import refresh_user_oauth_token

        server = _make_server(token_url=None)
        result = await refresh_user_oauth_token(
            prisma_client=MagicMock(),
            user_id="alice",
            server=server,
            cred=cred,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self, server, cred):
        from litellm.proxy._experimental.mcp_server.db import refresh_user_oauth_token

        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection refused")

        with patch(
            "litellm.proxy._experimental.mcp_server.db.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await refresh_user_oauth_token(
                prisma_client=MagicMock(),
                user_id="alice",
                server=server,
                cred=cred,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_stores_and_returns_new_credential(self, server, cred):
        from litellm.proxy._experimental.mcp_server.db import refresh_user_oauth_token

        new_token_response = MagicMock()
        new_token_response.json.return_value = {
            "access_token": "NEW_TOKEN",
            "expires_in": 3600,
            "refresh_token": "NEW_REFRESH",
            "scope": "channels:read chat:write",
        }
        new_token_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = new_token_response

        stored_cred = {
            "type": "oauth2",
            "access_token": "NEW_TOKEN",
            "refresh_token": "NEW_REFRESH",
        }
        mock_prisma = AsyncMock()

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.db.get_async_httpx_client",
                return_value=mock_client,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.db.store_user_oauth_credential",
                new_callable=AsyncMock,
            ) as mock_store,
            patch(
                "litellm.proxy._experimental.mcp_server.db.get_user_oauth_credential",
                new_callable=AsyncMock,
                return_value=stored_cred,
            ),
        ):
            result = await refresh_user_oauth_token(
                prisma_client=mock_prisma,
                user_id="alice",
                server=server,
                cred=cred,
            )

        assert result == stored_cred
        mock_store.assert_called_once()
        call_kwargs = mock_store.call_args[1]
        assert call_kwargs["access_token"] == "NEW_TOKEN"
        assert call_kwargs["refresh_token"] == "NEW_REFRESH"
        assert call_kwargs["expires_in"] == 3600
        assert call_kwargs["scopes"] == ["channels:read", "chat:write"]
        # Refresh path must skip the BYOK guard (row is already OAuth2)
        assert call_kwargs.get("skip_byok_guard") is True

    @pytest.mark.asyncio
    async def test_falls_back_to_old_refresh_token_when_not_rotated(self, server, cred):
        """When provider doesn't return a new refresh_token, keep the old one."""
        from litellm.proxy._experimental.mcp_server.db import refresh_user_oauth_token

        new_token_response = MagicMock()
        new_token_response.json.return_value = {
            "access_token": "NEW_TOKEN",
            "expires_in": 3600,
            # No refresh_token in response
        }
        new_token_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = new_token_response

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.db.get_async_httpx_client",
                return_value=mock_client,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.db.store_user_oauth_credential",
                new_callable=AsyncMock,
            ) as mock_store,
            patch(
                "litellm.proxy._experimental.mcp_server.db.get_user_oauth_credential",
                new_callable=AsyncMock,
                return_value={"type": "oauth2", "access_token": "NEW_TOKEN"},
            ),
        ):
            await refresh_user_oauth_token(
                prisma_client=AsyncMock(),
                user_id="alice",
                server=server,
                cred=cred,
            )

        call_kwargs = mock_store.call_args[1]
        # Old refresh_token preserved when provider doesn't rotate
        assert call_kwargs["refresh_token"] == "REFRESH_TOKEN_123"


# ── MCPServer new fields ──────────────────────────────────────────────────────


class TestMCPServerNewFields:
    def test_token_validation_default_none(self):
        server = _make_server()
        assert server.token_validation is None

    def test_token_validation_set(self):
        server = _make_server(token_validation={"enterprise_id": "E04XXXXXX"})
        assert server.token_validation == {"enterprise_id": "E04XXXXXX"}

    def test_token_storage_ttl_default_none(self):
        server = _make_server()
        assert server.token_storage_ttl_seconds is None

    def test_token_storage_ttl_set(self):
        server = _make_server(token_storage_ttl_seconds=7200)
        assert server.token_storage_ttl_seconds == 7200

    def test_needs_user_oauth_token_true_for_oauth2_without_m2m(self):
        server = _make_server(auth_type=MCPAuth.oauth2)
        assert server.needs_user_oauth_token is True

    def test_needs_user_oauth_token_false_for_m2m(self):
        server = _make_server(
            auth_type=MCPAuth.oauth2,
            oauth2_flow="client_credentials",
        )
        assert server.needs_user_oauth_token is False
