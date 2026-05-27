"""Tests for LIT-3301: restrict team virtual keys to team-scoped spend data.

These tests cover ``_is_admin_view_safe`` and the ``view_spend_logs``
``team_id`` enforcement to guarantee a team virtual key (one whose ``team_id``
is set) can never bypass team scope, even when the underlying user has the
``PROXY_ADMIN`` role.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    _is_admin_view_safe,
    view_spend_logs,
)


class TestIsAdminViewSafe:
    """``_is_admin_view_safe`` must reject team-scoped keys (LIT-3301)."""

    def test_personal_admin_key_is_admin(self) -> None:
        uak = UserAPIKeyAuth(
            api_key="sk-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id=None,
        )
        assert _is_admin_view_safe(uak) is True

    def test_personal_admin_view_only_key_is_admin(self) -> None:
        uak = UserAPIKeyAuth(
            api_key="sk-admin-view",
            user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            team_id=None,
        )
        assert _is_admin_view_safe(uak) is True

    def test_team_admin_key_is_not_admin_scoped(self) -> None:
        """Team virtual keys never get cross-team admin scope (LIT-3301)."""
        uak = UserAPIKeyAuth(
            api_key="sk-team-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id="team-A",
        )
        assert _is_admin_view_safe(uak) is False

    def test_team_admin_view_only_key_is_not_admin_scoped(self) -> None:
        uak = UserAPIKeyAuth(
            api_key="sk-team-admin-view",
            user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            team_id="team-B",
        )
        assert _is_admin_view_safe(uak) is False

    def test_internal_user_is_not_admin_scoped(self) -> None:
        uak = UserAPIKeyAuth(
            api_key="sk-internal",
            user_role=LitellmUserRoles.INTERNAL_USER,
            team_id=None,
        )
        assert _is_admin_view_safe(uak) is False

    def test_missing_user_role_is_not_admin_scoped(self) -> None:
        uak = UserAPIKeyAuth(api_key="sk-none")
        assert _is_admin_view_safe(uak) is False


@pytest.mark.asyncio
class TestViewSpendLogsTeamScope:
    """``view_spend_logs`` (legacy ``/spend/logs``) must inject ``team_id`` into
    every filter when the calling key is team-scoped (LIT-3301)."""

    async def test_team_admin_key_filters_by_team_id_with_date_range(self) -> None:
        team_uak = UserAPIKeyAuth(
            api_key="sk-team-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id="team-A",
        )

        mock_prisma = MagicMock()
        mock_prisma.hash_token = MagicMock(side_effect=lambda token: "hashed-" + token)
        mock_prisma.db.litellm_spendlogs.find_many = AsyncMock(return_value=[])
        mock_prisma.db.litellm_spendlogs.group_by = AsyncMock(return_value=[])

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
            create=True,
        ):
            await view_spend_logs(
                api_key=None,
                user_id=None,
                request_id=None,
                start_date="2026-01-01",
                end_date="2026-01-02",
                summarize=False,
                user_api_key_dict=team_uak,
            )

        mock_prisma.db.litellm_spendlogs.find_many.assert_awaited_once()
        call = mock_prisma.db.litellm_spendlogs.find_many.call_args
        where = call.kwargs.get("where") or (call.args[0] if call.args else {})
        assert where.get("team_id") == "team-A"

    async def test_team_admin_key_filters_by_team_id_without_date_range(self) -> None:
        team_uak = UserAPIKeyAuth(
            api_key="sk-team-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id="team-A",
        )

        mock_prisma = MagicMock()
        mock_prisma.hash_token = MagicMock(side_effect=lambda token: "hashed-" + token)
        mock_prisma.db.litellm_spendlogs.find_many = AsyncMock(return_value=[])
        mock_prisma.get_data = AsyncMock(return_value=[])

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
            create=True,
        ):
            await view_spend_logs(
                api_key=None,
                user_id=None,
                request_id=None,
                start_date=None,
                end_date=None,
                summarize=True,
                user_api_key_dict=team_uak,
            )

        mock_prisma.get_data.assert_not_awaited()
        mock_prisma.db.litellm_spendlogs.find_many.assert_awaited_once()
        call = mock_prisma.db.litellm_spendlogs.find_many.call_args
        where = call.kwargs.get("where") or (call.args[0] if call.args else {})
        assert where.get("team_id") == "team-A"

    async def test_personal_admin_key_unchanged_no_team_filter(self) -> None:
        """Regression guard: a personal admin key keeps its existing wide scope."""
        admin_uak = UserAPIKeyAuth(
            api_key="sk-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id=None,
        )

        mock_prisma = MagicMock()
        mock_prisma.hash_token = MagicMock(side_effect=lambda token: "hashed-" + token)
        mock_prisma.db.litellm_spendlogs.find_many = AsyncMock(return_value=[])

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
            create=True,
        ):
            await view_spend_logs(
                api_key=None,
                user_id=None,
                request_id=None,
                start_date="2026-01-01",
                end_date="2026-01-02",
                summarize=False,
                user_api_key_dict=admin_uak,
            )

        mock_prisma.db.litellm_spendlogs.find_many.assert_awaited_once()
        call = mock_prisma.db.litellm_spendlogs.find_many.call_args
        where = call.kwargs.get("where") or (call.args[0] if call.args else {})
        assert "team_id" not in where
