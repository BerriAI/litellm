"""
Auth Integration Tests
=======================

End-to-end tests that exercise the full auth pipeline (user_api_key_auth)
for different auth types: custom auth, key-based auth.

These tests mock at the boundary (DB, cache) but let the real auth logic run,
catching regressions where unit-level mocks might miss interaction bugs.

Background: The PR chain (#22164 -> #22662 -> b44755db) showed that unit tests
on individual functions weren't enough — the bug was in how functions composed.
These integration tests cover the full flow.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import common_checks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_request(route: str, method: str = "POST") -> MagicMock:
    """Create a mock FastAPI Request targeting a specific route."""
    req = MagicMock(spec=Request)
    req.url.path = route
    req.method = method
    req.query_params = {}
    req.headers = {}
    return req


# ---------------------------------------------------------------------------
# 1. Custom auth: full pipeline integration
# ---------------------------------------------------------------------------


class TestCustomAuthIntegration:
    """
    Tests that exercise _run_post_custom_auth_checks through
    realistic scenarios, verifying the interaction between:
    - custom auth callback
    - post-auth checks (end_user lookup, team lookup, etc.)
    - common_checks gating
    """

    @pytest.mark.asyncio
    async def test_custom_route_allowed_by_default(self):
        """
        A custom user-defined route (e.g. /ldap/ngs/ready) must pass
        through custom auth without being rejected as admin-only.

        This is the exact scenario that PR #22164 broke.
        """
        from litellm.proxy.auth.user_api_key_auth import _run_post_custom_auth_checks

        valid_token = UserAPIKeyAuth(
            token="sk-custom-key",
            user_id="ldap-user",
        )

        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common, patch(
            "litellm.proxy.auth.user_api_key_auth.get_user_object",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_get_user, patch(
            "litellm.proxy.proxy_server.general_settings",
            {},
        ):
            mock_common.return_value = True
            result = await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=MagicMock(),
                request_data={},
                route="/ldap/ngs/ready",
                parent_otel_span=None,
            )

            # common_checks must NOT be called (backwards compat)
            mock_common.assert_not_called()
            # user lookup should have been attempted
            mock_get_user.assert_called_once()
            # Token should pass through unchanged
            assert result.token == "sk-custom-key"

    @pytest.mark.asyncio
    async def test_custom_auth_with_opt_in_enforces_budget(self):
        """
        When custom_auth_run_common_checks=True, budget enforcement
        via common_checks actually runs.
        """
        from litellm.proxy.auth.user_api_key_auth import _run_post_custom_auth_checks

        valid_token = UserAPIKeyAuth(
            token="sk-custom-key",
            user_id="budget-user",
        )

        budget_error = HTTPException(
            status_code=400,
            detail="Budget exceeded",
        )

        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common, patch(
            "litellm.proxy.auth.user_api_key_auth.get_user_object",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ):
            mock_common.side_effect = budget_error

            with pytest.raises(HTTPException) as exc_info:
                await _run_post_custom_auth_checks(
                    valid_token=valid_token,
                    request=MagicMock(),
                    request_data={},
                    route="/chat/completions",
                    parent_otel_span=None,
                )

            assert exc_info.value.status_code == 400
            assert "Budget exceeded" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_custom_auth_without_opt_in_ignores_budget(self):
        """
        Without the opt-in flag, even if common_checks would reject
        (budget exceeded), custom auth still passes.
        """
        from litellm.proxy.auth.user_api_key_auth import _run_post_custom_auth_checks

        valid_token = UserAPIKeyAuth(
            token="sk-custom-key",
            user_id="budget-user",
        )

        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common, patch(
            "litellm.proxy.auth.user_api_key_auth.get_user_object",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "litellm.proxy.proxy_server.general_settings",
            {},
        ):
            # Even if common_checks would raise, it should never be called
            mock_common.side_effect = HTTPException(
                status_code=400, detail="Budget exceeded"
            )

            result = await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=MagicMock(),
                request_data={},
                route="/chat/completions",
                parent_otel_span=None,
            )

            mock_common.assert_not_called()
            assert result.token == "sk-custom-key"

    @pytest.mark.asyncio
    async def test_custom_auth_with_end_user_id(self):
        """
        Custom auth returning an end_user_id triggers the end-user
        lookup, and budget limits from the DB object are applied to
        the valid_token.
        """
        from litellm.proxy.auth.user_api_key_auth import _run_post_custom_auth_checks

        valid_token = UserAPIKeyAuth(
            token="sk-custom-key",
            end_user_id="eu-123",
        )

        # Simulate _lookup_end_user_and_apply_budget returning a token
        # with allowed_model_region set (as it would from a real DB lookup)
        patched_token = valid_token.model_copy()
        patched_token.allowed_model_region = "us"

        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common, patch(
            "litellm.proxy.auth.user_api_key_auth._lookup_end_user_and_apply_budget",
            new_callable=AsyncMock,
            return_value=(patched_token, MagicMock()),
        ) as mock_lookup, patch(
            "litellm.proxy.proxy_server.general_settings",
            {},
        ):
            mock_common.return_value = True
            result = await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=MagicMock(),
                request_data={},
                route="/v1/chat/completions",
                parent_otel_span=None,
            )

            # Verify the lookup was called with the end_user_id
            mock_lookup.assert_called_once()
            assert result.end_user_id == "eu-123"
            # Verify budget info from DB lookup was propagated
            assert result.allowed_model_region == "us"

    @pytest.mark.asyncio
    async def test_custom_auth_with_team_id_triggers_team_lookup(self):
        """
        When custom auth sets a team_id on the token, the post-auth
        checks should attempt team object lookup.
        """
        from litellm.proxy.auth.user_api_key_auth import _run_post_custom_auth_checks

        valid_token = UserAPIKeyAuth(
            token="sk-custom-key",
            team_id="team-abc",
        )

        mock_team = MagicMock()
        mock_team.blocked = False

        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common, patch(
            "litellm.proxy.auth.user_api_key_auth.get_team_object",
            new_callable=AsyncMock,
            return_value=mock_team,
        ) as mock_get_team, patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ), patch(
            "litellm.proxy.proxy_server.prisma_client",
            MagicMock(),
        ), patch(
            "litellm.proxy.proxy_server.user_api_key_cache",
            MagicMock(),
        ):
            mock_common.return_value = True
            await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=MagicMock(),
                request_data={},
                route="/chat/completions",
                parent_otel_span=None,
            )

            mock_get_team.assert_called_once()
            mock_common.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Key-based auth: common_checks integration
# ---------------------------------------------------------------------------


class TestKeyBasedAuthCommonChecks:
    """
    Tests that common_checks correctly enforces route authorization
    for standard key-based auth flows.
    """

    @pytest.mark.asyncio
    async def test_llm_route_allowed_for_regular_user(self):
        """Regular users can call LLM API routes through common_checks."""
        user_obj = LiteLLM_UserTable(
            user_id="user1",
            user_role=LitellmUserRoles.INTERNAL_USER.value,
        )
        valid_token = UserAPIKeyAuth(
            token="sk-key",
            user_id="user1",
            user_role=LitellmUserRoles.INTERNAL_USER.value,
        )

        result = await common_checks(
            request_body={"model": "gpt-4"},
            team_object=None,
            user_object=user_obj,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=MagicMock(),
            valid_token=valid_token,
            request=_make_mock_request("/chat/completions"),
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_admin_route_denied_for_regular_user(self):
        """Regular users cannot call admin-only routes through common_checks."""
        from litellm.proxy.auth.auth_checks import common_checks

        user_obj = LiteLLM_UserTable(
            user_id="user1",
            user_role=LitellmUserRoles.INTERNAL_USER.value,
        )
        valid_token = UserAPIKeyAuth(
            token="sk-key",
            user_id="user1",
            user_role=LitellmUserRoles.INTERNAL_USER.value,
        )

        with pytest.raises(Exception):
            await common_checks(
                request_body={},
                team_object=None,
                user_object=user_obj,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/config/update",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=valid_token,
                request=_make_mock_request("/config/update"),
            )

    @pytest.mark.asyncio
    async def test_admin_can_access_any_route(self):
        """Proxy admin can call any route through common_checks."""
        from litellm.proxy.auth.auth_checks import common_checks

        user_obj = LiteLLM_UserTable(
            user_id="admin1",
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
        )
        valid_token = UserAPIKeyAuth(
            token="sk-admin",
            user_id="admin1",
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
        )

        for route in ["/chat/completions", "/config/update", "/key/generate"]:
            result = await common_checks(
                request_body={},
                team_object=None,
                user_object=user_obj,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=valid_token,
                request=_make_mock_request(route),
            )
            assert result is True


# ---------------------------------------------------------------------------
# 3. Cross-cutting: ensure custom_auth_run_common_checks flag isolation
# ---------------------------------------------------------------------------


class TestFlagIsolation:
    """
    Verify that the custom_auth_run_common_checks flag ONLY affects
    custom auth flows and does NOT interfere with key-based or JWT auth.
    """

    @pytest.mark.asyncio
    async def test_flag_does_not_affect_common_checks_directly(self):
        """
        common_checks() itself should not read the flag —
        it's the caller (_run_post_custom_auth_checks) that gates the call.
        """
        from litellm.proxy.auth.auth_checks import common_checks

        user_obj = LiteLLM_UserTable(
            user_id="user1",
            user_role=LitellmUserRoles.INTERNAL_USER.value,
        )
        valid_token = UserAPIKeyAuth(
            token="sk-key",
            user_id="user1",
            user_role=LitellmUserRoles.INTERNAL_USER.value,
        )

        # common_checks should work the same regardless of the flag
        for flag_value in [True, False]:
            result = await common_checks(
                request_body={},
                team_object=None,
                user_object=user_obj,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={"custom_auth_run_common_checks": flag_value},
                route="/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=valid_token,
                request=_make_mock_request("/chat/completions"),
            )
            assert result is True
