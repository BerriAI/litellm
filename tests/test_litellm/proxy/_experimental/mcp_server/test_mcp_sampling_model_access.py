"""
Tests for MCP sampling handler model-access enforcement.

Verifies that handle_sampling_create_message and _check_model_access
enforce the same model-permission checks as regular /chat/completions
calls, preventing a malicious upstream MCP server from requesting
inference on models the caller's API key is not authorized to use.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _check_model_access,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_api_key_auth(*, models=None, team_id=None, team_model_aliases=None):
    """Build a minimal UserAPIKeyAuth-like object for tests."""
    auth = MagicMock()
    auth.models = models or []
    auth.team_id = team_id
    auth.team_model_aliases = team_model_aliases or {}
    auth.access_group_ids = []
    return auth


# ---------------------------------------------------------------------------
# _check_model_access
# ---------------------------------------------------------------------------


class TestCheckModelAccess:
    """Tests for the _check_model_access helper that gates sampling requests."""

    @pytest.mark.asyncio
    async def test_should_return_none_when_no_auth_context(self):
        """No auth context means no restriction — pass through."""
        result = await _check_model_access("gpt-4o", user_api_key_auth=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_should_allow_model_when_key_has_access(self):
        """Key with explicit model access should be allowed."""
        auth = _make_user_api_key_auth(models=["gpt-4o", "gpt-3.5-turbo"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_check:
            result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        assert result is None
        mock_check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_deny_model_when_key_lacks_access(self):
        """Key without model access should be denied with ErrorData."""
        from litellm.proxy._types import ProxyException

        auth = _make_user_api_key_auth(models=["gpt-3.5-turbo"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            side_effect=ProxyException(
                message="key not allowed to access model",
                type="key_model_access_denied",
                param="model",
                code=401,
            ),
        ):
            result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        # Should return ErrorData, not raise
        assert result is not None
        assert result.code == -1
        assert "Model access denied" in result.message
        assert "gpt-4o" in result.message

    @pytest.mark.asyncio
    async def test_should_allow_wildcard_model_access(self):
        """Key with wildcard model access should allow any model."""
        auth = _make_user_api_key_auth(models=["*"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await _check_model_access(
                "claude-3-opus-20240229", user_api_key_auth=auth
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_should_deny_expensive_model_requested_by_malicious_server(self):
        """Simulates the attack: malicious MCP server hints at an expensive model
        the caller's key is restricted from using."""
        from litellm.proxy._types import ProxyException

        # Key only has access to cheap models
        auth = _make_user_api_key_auth(models=["gpt-3.5-turbo"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            side_effect=ProxyException(
                message="key not allowed to access model. This key can only access models=['gpt-3.5-turbo']. Tried to access claude-3-opus-20240229",
                type="key_model_access_denied",
                param="model",
                code=401,
            ),
        ):
            result = await _check_model_access(
                "claude-3-opus-20240229", user_api_key_auth=auth
            )

        assert result is not None
        assert result.code == -1
        assert "claude-3-opus-20240229" in result.message

