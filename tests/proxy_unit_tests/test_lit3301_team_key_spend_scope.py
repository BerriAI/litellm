"""Tests for LIT-3301: restrict team virtual keys to team-scoped spend data.

These tests cover ``_is_admin_view_safe`` and the ``view_spend_logs``
``team_id`` enforcement to guarantee a team virtual key (one whose ``team_id``
is set) can never bypass team scope, even when the underlying user has the
``PROXY_ADMIN`` role.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
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



# -----------------------------------------------------------------------------
# ui_view_spend_logs team_id enforcement (LIT-3301 follow-up)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUIViewSpendLogsTeamScope:
    """``ui_view_spend_logs`` (``/spend/logs/ui`` and ``/spend/logs/v2``) must
    force ``where_conditions['team_id']`` to the calling key's ``team_id`` for
    every team virtual key, regardless of:
    - the ``team_id`` query parameter being omitted, and
    - the underlying user role (PROXY_ADMIN previously bypassed the scope).
    """

    @staticmethod
    def _make_request(path: str = "/spend/logs/ui"):
        req = MagicMock()
        req.url.path = path
        return req

    async def _drive(
        self,
        team_uak,
        *,
        query_team_id=None,
        mock_prisma=None,
    ):
        from litellm.proxy.spend_tracking.spend_management_endpoints import (
            ui_view_spend_logs,
        )

        if mock_prisma is None:
            mock_prisma = MagicMock()
        mock_prisma.db.litellm_spendlogs.count = AsyncMock(return_value=0)
        mock_prisma.db.query_raw = AsyncMock(return_value=[])

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
            create=True,
        ):
            try:
                await ui_view_spend_logs(
                    api_key=None,
                    user_id=None,
                    request_id=None,
                    start_date="2026-01-01 00:00:00",
                    end_date="2026-01-02 00:00:00",
                    team_id=query_team_id,
                    model=None,
                    model_id=None,
                    end_user=None,
                    status_filter=None,
                    min_spend=None,
                    max_spend=None,
                    key_alias=None,
                    error_code=None,
                    error_message=None,
                    sort_by="startTime",
                    sort_order="desc",
                    page=1,
                    page_size=50,
                    user_api_key_dict=team_uak,
                    request=self._make_request(),
                )
            except (HTTPException, ProxyException) as e:
                return None, e, mock_prisma
        return True, None, mock_prisma

    async def test_team_admin_key_no_query_param_locks_to_team(self) -> None:
        """The Greptile-flagged path: PROXY_ADMIN team key, no team_id query
        param. Must end up with the calling key's ``team_id`` in the
        Prisma ``count`` where clause."""
        team_uak = UserAPIKeyAuth(
            api_key="sk-team-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id="team-A",
        )
        ok, exc, prisma = await self._drive(team_uak, query_team_id=None)
        assert ok and exc is None
        prisma.db.litellm_spendlogs.count.assert_awaited()
        where = prisma.db.litellm_spendlogs.count.call_args.kwargs.get("where", {})
        assert where.get("team_id") == "team-A"

    async def test_team_admin_key_with_matching_query_param_ok(self) -> None:
        team_uak = UserAPIKeyAuth(
            api_key="sk-team-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id="team-A",
        )
        ok, exc, prisma = await self._drive(team_uak, query_team_id="team-A")
        assert ok and exc is None
        where = prisma.db.litellm_spendlogs.count.call_args.kwargs.get("where", {})
        assert where.get("team_id") == "team-A"

    async def test_team_admin_key_with_mismatched_query_param_403(self) -> None:
        team_uak = UserAPIKeyAuth(
            api_key="sk-team-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id="team-A",
        )
        ok, exc, _ = await self._drive(team_uak, query_team_id="team-B")
        assert ok is None
        assert exc is not None
        # The endpoint wraps the underlying HTTPException(403) into a
        # litellm ProxyException; either type is acceptable.
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        assert int(code) == 403

    async def test_personal_admin_key_unchanged(self) -> None:
        """Regression guard: personal admin key keeps wide scope when no
        team_id query parameter is supplied."""
        admin_uak = UserAPIKeyAuth(
            api_key="sk-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            team_id=None,
        )
        ok, exc, prisma = await self._drive(admin_uak, query_team_id=None)
        assert ok and exc is None
        where = prisma.db.litellm_spendlogs.count.call_args.kwargs.get("where", {})
        assert "team_id" not in where
