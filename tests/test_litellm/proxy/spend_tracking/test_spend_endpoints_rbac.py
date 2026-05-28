"""
Unit tests for cross-tenant isolation fix on /spend/keys and /spend/users.

Regression tests for https://github.com/BerriAI/litellm/issues/28864

Before the fix, both endpoints performed an unconstrained `find_all` for every
authenticated caller, leaking every key and every user to any INTERNAL_USER.
After the fix, non-admin callers only receive their own keys / user row, and
a non-admin who explicitly requests a different user's data via the ``user_id``
query param receives HTTP 403.
"""
from __future__ import annotations

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# Make sure the repo root is on the path so relative imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    spend_key_fn,
    spend_user_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_caller(role: LitellmUserRoles, user_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_role=role, user_id=user_id)


def _make_prisma_mock() -> MagicMock:
    """Return a minimal mock that satisfies the prisma_client contract."""
    mock = MagicMock()
    mock.get_data = AsyncMock(return_value=[])
    return mock


# ---------------------------------------------------------------------------
# /spend/keys
# ---------------------------------------------------------------------------

class TestSpendKeyFnRbac:
    """spend_key_fn RBAC isolation tests."""

    @pytest.mark.asyncio
    async def test_internal_user_only_sees_own_keys(self):
        """INTERNAL_USER must query with user_id=caller, not a bare find_all."""
        mock_prisma = _make_prisma_mock()

        with patch(
            "litellm.proxy.spend_tracking.spend_management_endpoints.prisma_client",
            mock_prisma,
            create=True,
        ), patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            caller = _make_caller(LitellmUserRoles.INTERNAL_USER, "alice")
            await spend_key_fn(user_api_key_dict=caller)

        mock_prisma.get_data.assert_called_once()
        call_kwargs = mock_prisma.get_data.call_args[1]
        assert call_kwargs.get("user_id") == "alice", (
            "Non-admin callers must have user_id scoped to their own ID"
        )

    @pytest.mark.asyncio
    async def test_internal_user_view_only_only_sees_own_keys(self):
        """INTERNAL_USER_VIEW_ONLY must also be scoped."""
        mock_prisma = _make_prisma_mock()

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(
                LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, "bob"
            )
            await spend_key_fn(user_api_key_dict=caller)

        call_kwargs = mock_prisma.get_data.call_args[1]
        assert call_kwargs.get("user_id") == "bob"

    @pytest.mark.asyncio
    async def test_admin_sees_all_keys(self):
        """PROXY_ADMIN must call get_data *without* a user_id filter."""
        mock_prisma = _make_prisma_mock()

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(LitellmUserRoles.PROXY_ADMIN, "admin")
            await spend_key_fn(user_api_key_dict=caller)

        call_kwargs = mock_prisma.get_data.call_args[1]
        # Admins should NOT have their query filtered by user_id
        assert call_kwargs.get("user_id") is None, (
            "Admin callers should not have a user_id restriction"
        )


# ---------------------------------------------------------------------------
# /spend/users
# ---------------------------------------------------------------------------

class TestSpendUserFnRbac:
    """spend_user_fn RBAC isolation tests."""

    @pytest.mark.asyncio
    async def test_internal_user_no_param_sees_own_row(self):
        """Without user_id param, non-admin sees their own row via find_unique."""
        mock_prisma = _make_prisma_mock()
        mock_prisma.get_data.return_value = {"user_id": "alice"}

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(LitellmUserRoles.INTERNAL_USER, "alice")
            result = await spend_user_fn(user_id=None, user_api_key_dict=caller)

        # Should have queried find_unique for the caller's own user_id
        call_kwargs = mock_prisma.get_data.call_args[1]
        assert call_kwargs.get("user_id") == "alice"
        assert call_kwargs.get("query_type") == "find_unique"

    @pytest.mark.asyncio
    async def test_internal_user_same_user_id_allowed(self):
        """Non-admin providing their own user_id is fine."""
        mock_prisma = _make_prisma_mock()
        mock_prisma.get_data.return_value = {"user_id": "alice"}

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(LitellmUserRoles.INTERNAL_USER, "alice")
            # Should NOT raise
            await spend_user_fn(user_id="alice", user_api_key_dict=caller)

    @pytest.mark.asyncio
    async def test_internal_user_cross_tenant_user_id_raises_403(self):
        """Non-admin requesting another user's data must get 403."""
        mock_prisma = _make_prisma_mock()

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(LitellmUserRoles.INTERNAL_USER, "alice")
            with pytest.raises(HTTPException) as exc_info:
                await spend_user_fn(user_id="bob", user_api_key_dict=caller)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_internal_user_view_only_cross_tenant_raises_403(self):
        """INTERNAL_USER_VIEW_ONLY cannot read another user's row."""
        mock_prisma = _make_prisma_mock()

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(
                LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, "carol"
            )
            with pytest.raises(HTTPException) as exc_info:
                await spend_user_fn(user_id="dave", user_api_key_dict=caller)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_can_request_any_user(self):
        """PROXY_ADMIN is not restricted; any user_id query param is forwarded."""
        mock_prisma = _make_prisma_mock()
        mock_prisma.get_data.return_value = {"user_id": "carol"}

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(LitellmUserRoles.PROXY_ADMIN, "admin")
            # Requesting another user should work without 403
            await spend_user_fn(user_id="carol", user_api_key_dict=caller)

        call_kwargs = mock_prisma.get_data.call_args[1]
        assert call_kwargs.get("user_id") == "carol"

    @pytest.mark.asyncio
    async def test_admin_no_param_sees_all_users(self):
        """Admin with no user_id param triggers find_all."""
        mock_prisma = _make_prisma_mock()
        mock_prisma.get_data.return_value = []

        with patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            caller = _make_caller(LitellmUserRoles.PROXY_ADMIN, "admin")
            await spend_user_fn(user_id=None, user_api_key_dict=caller)

        call_kwargs = mock_prisma.get_data.call_args[1]
        assert call_kwargs.get("query_type") == "find_all"
