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


def _make_user_api_key_auth(
    *,
    models=None,
    team_id=None,
    team_model_aliases=None,
    api_key="sk-test-key",
    token=None,
    user_role=None,
):
    """Build a minimal UserAPIKeyAuth-like object for tests."""
    auth = MagicMock()
    auth.models = models or []
    auth.team_id = team_id
    auth.team_model_aliases = team_model_aliases or {}
    auth.access_group_ids = []
    auth.api_key = api_key
    auth.token = token
    auth.user_role = user_role
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

    @pytest.mark.asyncio
    async def test_should_deny_empty_oauth_passthrough_placeholder(self):
        """Regression: process_mcp_request() returns an empty UserAPIKeyAuth()
        for OAuth2 upstream-token passthrough.  The None check alone is not
        sufficient — the empty placeholder is truthy but has no api_key, no
        token, and an empty models list.  can_key_call_model() would treat
        that as all-model access, letting an OAuth-only user trigger sampling
        calls on any proxy model without a LiteLLM key or budget."""
        # Simulate the empty placeholder from process_mcp_request()
        auth = _make_user_api_key_auth(
            models=[],
            api_key=None,
            token=None,
            user_role=None,
        )

        result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        # Must be denied — not passed through to can_key_call_model
        assert result is not None
        assert result.code == -1
        assert "sampling requires a valid LiteLLM" in result.message

    @pytest.mark.asyncio
    async def test_should_allow_proxy_admin_even_without_api_key(self):
        """Proxy admins may not have a traditional api_key but should still
        be allowed to use sampling."""
        auth = _make_user_api_key_auth(
            models=[],
            api_key=None,
            token=None,
            user_role="proxy_admin",
        )

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        assert result is None


# ---------------------------------------------------------------------------
# handle_sampling_create_message — auth + budget gating
# ---------------------------------------------------------------------------


class TestSamplingAuthAndBudgetGating:

    @pytest.mark.asyncio
    async def test_should_deny_when_no_auth_context(self):
        """Sampling must reject calls with no user_api_key_auth."""
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        params = MagicMock()
        params.modelPreferences = None
        params.messages = []
        params.systemPrompt = None
        params.maxTokens = 100
        params.temperature = None
        params.stopSequences = None
        params.tools = None
        params.toolChoice = None
        params.metadata = None

        result = await handle_sampling_create_message(
            context=MagicMock(),
            params=params,
            default_model="gpt-4o",
            user_api_key_auth=None,
        )

        assert result is not None
        assert result.code == -1
        assert "authenticated" in result.message.lower()

    @pytest.mark.asyncio
    async def test_should_run_budget_checks(self):
        """Sampling must call _run_budget_checks after model access check."""
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        auth = _make_user_api_key_auth(models=["gpt-4o"])
        params = MagicMock()
        params.modelPreferences = None
        params.messages = []
        params.systemPrompt = None
        params.maxTokens = 100
        params.temperature = None
        params.stopSequences = None
        params.tools = None
        params.toolChoice = None
        params.metadata = None

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._check_model_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_budget_checks",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_budget,
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._resolve_model_from_preferences",
                return_value="gpt-4o",
            ),
            patch(
                "litellm.proxy.proxy_server.llm_router",
                new=None,
            ),
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content="hi", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    model="gpt-4o",
                ),
            ),
        ):
            await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o",
                user_api_key_auth=auth,
            )

        mock_budget.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_deny_over_budget_caller(self):
        """When _run_budget_checks returns ErrorData, sampling must return it."""
        from mcp.types import ErrorData
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        auth = _make_user_api_key_auth(models=["gpt-4o"])
        params = MagicMock()
        params.modelPreferences = None
        params.messages = []
        params.systemPrompt = None
        params.maxTokens = 100
        params.temperature = None
        params.stopSequences = None
        params.tools = None
        params.toolChoice = None
        params.metadata = None

        budget_error = ErrorData(code=-1, message="ExceededBudget: over limit")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._check_model_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_budget_checks",
                new_callable=AsyncMock,
                return_value=budget_error,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._resolve_model_from_preferences",
                return_value="gpt-4o",
            ),
        ):
            result = await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o",
                user_api_key_auth=auth,
            )

        assert result is budget_error
        assert "ExceededBudget" in result.message
