"""
Tests for applying default team params during team creation
and loading default_team_params from DB on startup.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy._types import (
    NewTeamRequest,
    UserAPIKeyAuth,
    LitellmUserRoles,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    _get_default_team_param,
)
from litellm.proxy.proxy_server import ProxyConfig


# ---------------------------------------------------------------------------
# _update_config_fields: default_team_params loaded from DB on startup
# ---------------------------------------------------------------------------


class TestConfigFieldsDefaultTeamParams:
    """Tests that _update_config_fields applies default_team_params from DB."""

    def _make_proxy_config(self) -> ProxyConfig:
        return ProxyConfig()

    def test_default_team_params_applied_from_db(self, monkeypatch):
        """default_team_params in DB is set on litellm module during config load."""
        monkeypatch.setattr(litellm, "default_team_params", None)

        pc = self._make_proxy_config()
        db_settings = {
            "default_team_params": {
                "max_budget": 500.0,
                "budget_duration": "30d",
                "tpm_limit": 1000,
                "rpm_limit": 200,
                "team_member_permissions": ["/key/generate", "/key/delete"],
            }
        }

        pc._update_config_fields(
            current_config={},
            param_name="litellm_settings",
            db_param_value=db_settings,
        )

        assert litellm.default_team_params == db_settings["default_team_params"]

    def test_default_team_params_merged_into_config_dict(self):
        """DB default_team_params ends up in the returned config dict."""
        pc = self._make_proxy_config()
        config = {"litellm_settings": {"cache": False}}
        db_settings = {
            "default_team_params": {
                "max_budget": 100.0,
            }
        }

        result = pc._update_config_fields(
            current_config=config,
            param_name="litellm_settings",
            db_param_value=db_settings,
        )

        assert result["litellm_settings"]["default_team_params"] == {"max_budget": 100.0}
        # Existing keys preserved
        assert result["litellm_settings"]["cache"] is False

    def test_default_team_params_not_applied_when_absent(self, monkeypatch):
        """When DB litellm_settings has no default_team_params, it stays None."""
        monkeypatch.setattr(litellm, "default_team_params", None)

        pc = self._make_proxy_config()
        pc._update_config_fields(
            current_config={},
            param_name="litellm_settings",
            db_param_value={"cache": True},
        )

        assert litellm.default_team_params is None

    def test_default_team_params_overrides_yaml_value(self, monkeypatch):
        """DB value for default_team_params overrides YAML value via deep merge."""
        monkeypatch.setattr(litellm, "default_team_params", None)

        pc = self._make_proxy_config()
        config = {
            "litellm_settings": {
                "default_team_params": {
                    "max_budget": 50.0,
                    "tpm_limit": 100,
                }
            }
        }
        db_settings = {
            "default_team_params": {
                "max_budget": 200.0,
                "rpm_limit": 500,
            }
        }

        result = pc._update_config_fields(
            current_config=config,
            param_name="litellm_settings",
            db_param_value=db_settings,
        )

        merged = result["litellm_settings"]["default_team_params"]
        # DB value wins for max_budget
        assert merged["max_budget"] == 200.0
        # DB adds rpm_limit
        assert merged["rpm_limit"] == 500
        # YAML tpm_limit preserved (not in DB)
        assert merged["tpm_limit"] == 100

        # setattr should have applied the DB value
        assert litellm.default_team_params == db_settings["default_team_params"]


# ---------------------------------------------------------------------------
# new_team: default params applied to team creation
#
# We test the defaults-application logic by calling new_team with
# prisma_client patched at the proxy_server module level (where the
# endpoint imports it from).
# ---------------------------------------------------------------------------


class TestNewTeamDefaultParamsApplied:
    """Tests that /team/new applies defaults from litellm.default_team_params."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Set up common mocks for team creation tests."""
        mock_prisma = AsyncMock()
        mock_prisma.insert_data = AsyncMock(
            return_value=MagicMock(
                team_id="test-team-id",
                team_alias="test-team",
            )
        )
        mock_prisma.get_generic_data = AsyncMock(return_value=None)
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_teamtable = MagicMock()
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)

        monkeypatch.setattr(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        )

        # Reset default_team_settings to avoid legacy fallback interference
        monkeypatch.setattr(litellm, "default_team_settings", None)

    def _make_admin_auth(self) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            user_id="admin-user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

    @pytest.mark.asyncio
    async def test_all_defaults_applied_when_not_provided(self, monkeypatch):
        """When no budget/rate/permission fields are in the request, all defaults apply."""
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        monkeypatch.setattr(
            litellm,
            "default_team_params",
            {
                "max_budget": 100.0,
                "budget_duration": "30d",
                "tpm_limit": 200,
                "rpm_limit": 500,
                "team_member_permissions": ["/key/generate", "/key/update"],
            },
        )

        data = NewTeamRequest(team_alias="my-team")
        auth = self._make_admin_auth()

        try:
            await new_team(
                data=data,
                user_api_key_dict=auth,
                http_request=MagicMock(),
            )
        except Exception:
            pass  # May fail on downstream mocks, that's OK

        # Verify defaults were set on the data object
        assert data.max_budget == 100.0
        assert data.budget_duration == "30d"
        assert data.tpm_limit == 200
        assert data.rpm_limit == 500
        assert data.team_member_permissions == ["/key/generate", "/key/update"]

    @pytest.mark.asyncio
    async def test_explicit_values_not_overridden(self, monkeypatch):
        """When request provides explicit values, defaults do not override them."""
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        monkeypatch.setattr(
            litellm,
            "default_team_params",
            {
                "max_budget": 100.0,
                "budget_duration": "30d",
                "tpm_limit": 200,
                "rpm_limit": 500,
                "team_member_permissions": ["/key/generate"],
            },
        )

        data = NewTeamRequest(
            team_alias="my-team",
            max_budget=50.0,
            budget_duration="7d",
            tpm_limit=999,
            rpm_limit=888,
            team_member_permissions=["/key/delete"],
        )
        auth = self._make_admin_auth()

        try:
            await new_team(
                data=data,
                user_api_key_dict=auth,
                http_request=MagicMock(),
            )
        except Exception:
            pass

        # Explicit values preserved
        assert data.max_budget == 50.0
        assert data.budget_duration == "7d"
        assert data.tpm_limit == 999
        assert data.rpm_limit == 888
        assert data.team_member_permissions == ["/key/delete"]

    @pytest.mark.asyncio
    async def test_partial_defaults_applied(self, monkeypatch):
        """Only missing fields get defaults; provided fields are untouched."""
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        monkeypatch.setattr(
            litellm,
            "default_team_params",
            {
                "max_budget": 100.0,
                "budget_duration": "30d",
                "tpm_limit": 200,
                "rpm_limit": 500,
            },
        )

        data = NewTeamRequest(
            team_alias="my-team",
            max_budget=75.0,  # explicit
            # budget_duration, tpm_limit, rpm_limit not set → defaults apply
        )
        auth = self._make_admin_auth()

        try:
            await new_team(
                data=data,
                user_api_key_dict=auth,
                http_request=MagicMock(),
            )
        except Exception:
            pass

        assert data.max_budget == 75.0  # explicit, not overridden
        assert data.budget_duration == "30d"  # default applied
        assert data.tpm_limit == 200  # default applied
        assert data.rpm_limit == 500  # default applied

    @pytest.mark.asyncio
    async def test_no_defaults_when_config_is_none(self, monkeypatch):
        """When default_team_params is None, no defaults applied."""
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        monkeypatch.setattr(litellm, "default_team_params", None)

        data = NewTeamRequest(team_alias="my-team")
        auth = self._make_admin_auth()

        try:
            await new_team(
                data=data,
                user_api_key_dict=auth,
                http_request=MagicMock(),
            )
        except Exception:
            pass

        assert data.max_budget is None
        assert data.budget_duration is None
        assert data.tpm_limit is None
        assert data.rpm_limit is None
        assert data.team_member_permissions is None

    @pytest.mark.asyncio
    async def test_legacy_default_team_settings_fallback(self, monkeypatch):
        """Legacy default_team_settings YAML config applies max_budget as fallback."""
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        monkeypatch.setattr(litellm, "default_team_params", None)
        monkeypatch.setattr(
            litellm,
            "default_team_settings",
            [{"team_id": "default", "max_budget": 999.0}],
        )

        data = NewTeamRequest(team_alias="my-team")
        auth = self._make_admin_auth()

        try:
            await new_team(
                data=data,
                user_api_key_dict=auth,
                http_request=MagicMock(),
            )
        except Exception:
            pass

        assert data.max_budget == 999.0

    @pytest.mark.asyncio
    async def test_default_team_params_takes_priority_over_legacy(self, monkeypatch):
        """default_team_params max_budget takes priority over legacy default_team_settings."""
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        monkeypatch.setattr(
            litellm,
            "default_team_params",
            {"max_budget": 100.0},
        )
        monkeypatch.setattr(
            litellm,
            "default_team_settings",
            [{"team_id": "default", "max_budget": 999.0}],
        )

        data = NewTeamRequest(team_alias="my-team")
        auth = self._make_admin_auth()

        try:
            await new_team(
                data=data,
                user_api_key_dict=auth,
                http_request=MagicMock(),
            )
        except Exception:
            pass

        # default_team_params wins (100.0), legacy fallback (999.0) not used
        assert data.max_budget == 100.0


# ---------------------------------------------------------------------------
# _update_litellm_setting: setattr ordering
# ---------------------------------------------------------------------------


class TestUpdateLitellmSettingOrdering:
    """Tests that _update_litellm_setting sets in-memory value AFTER get_config,
    so stale DB values from LITELLM_SETTINGS_SAFE_DB_OVERRIDES don't overwrite it."""

    @pytest.mark.asyncio
    async def test_setattr_not_overwritten_by_get_config(self, monkeypatch):
        """The new in-memory value survives get_config() which may load stale DB values."""
        from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
            _update_litellm_setting,
        )
        from litellm.types.proxy.management_endpoints.ui_sso import (
            DefaultTeamSSOParams,
        )

        # Simulate stale DB state: get_config returns old default_team_params
        stale_value = {"max_budget": 50.0}
        monkeypatch.setattr(litellm, "default_team_params", stale_value)

        # get_config will overwrite litellm.default_team_params with stale DB value
        async def mock_get_config():
            # Simulate what _update_config_from_db does for safe overrides
            litellm.default_team_params = stale_value
            return {
                "litellm_settings": {
                    "default_team_params": stale_value,
                }
            }

        saved_configs = []

        async def mock_save_config(new_config=None):
            saved_configs.append(new_config)

        from litellm.proxy.proxy_server import proxy_config

        monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
        monkeypatch.setattr(proxy_config, "save_config", mock_save_config)
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.store_model_in_db", True
        )

        # New settings to save
        new_settings = DefaultTeamSSOParams(
            max_budget=200.0,
            budget_duration="7d",
            rpm_limit=1000,
        )

        result = await _update_litellm_setting(
            settings=new_settings,
            settings_key="default_team_params",
            success_message="Updated",
        )

        # In-memory value should be the NEW value, not the stale one
        expected = new_settings.model_dump(exclude_none=True)
        assert litellm.default_team_params == expected

        # Saved config should contain the new value
        assert len(saved_configs) == 1
        saved_settings = saved_configs[0]["litellm_settings"]["default_team_params"]
        assert saved_settings == expected

        # Return value should reflect the new settings
        assert result["settings"] == expected

    @pytest.mark.asyncio
    async def test_requires_store_model_in_db(self, monkeypatch):
        """Raises HTTPException when store_model_in_db is not True."""
        from fastapi import HTTPException

        from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
            _update_litellm_setting,
        )
        from litellm.types.proxy.management_endpoints.ui_sso import (
            DefaultTeamSSOParams,
        )

        monkeypatch.setattr(
            "litellm.proxy.proxy_server.store_model_in_db", False
        )

        with pytest.raises(HTTPException) as exc_info:
            await _update_litellm_setting(
                settings=DefaultTeamSSOParams(max_budget=100.0),
                settings_key="default_team_params",
                success_message="Updated",
            )

        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# LITELLM_SETTINGS_SAFE_DB_OVERRIDES contains default_team_params
# ---------------------------------------------------------------------------


class TestSafeDbOverrides:
    """Verify default_team_params is in the safe overrides list."""

    def test_default_team_params_in_safe_overrides(self):
        from litellm.constants import LITELLM_SETTINGS_SAFE_DB_OVERRIDES

        assert "default_team_params" in LITELLM_SETTINGS_SAFE_DB_OVERRIDES

    def test_default_internal_user_params_in_safe_overrides(self):
        """Sanity: default_internal_user_params was already in the list."""
        from litellm.constants import LITELLM_SETTINGS_SAFE_DB_OVERRIDES

        assert "default_internal_user_params" in LITELLM_SETTINGS_SAFE_DB_OVERRIDES


# ---------------------------------------------------------------------------
# POST /team/permissions/bulk_update
# ---------------------------------------------------------------------------


class TestBulkUpdateTeamMemberPermissions:
    """Tests for the bulk_update_team_member_permissions endpoint."""

    def _make_team(self, team_id: str, permissions: list):
        """Create a mock team object."""
        team = MagicMock()
        team.team_id = team_id
        team.team_member_permissions = permissions
        return team

    def _admin_key_dict(self):
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

        return UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
            api_key="sk-1234",
        )

    def _non_admin_key_dict(self):
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

        return UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER.value,
            api_key="sk-user",
        )

    # --- apply_to_all_teams tests ---

    @pytest.mark.asyncio
    async def test_all_teams_appends_preserving_existing(self, monkeypatch):
        """apply_to_all_teams: permissions are merged, not overwritten."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        team_a = self._make_team("team-a", ["/key/generate"])
        team_b = self._make_team("team-b", ["/key/delete", "/key/update"])

        mock_batcher = MagicMock()
        mock_batcher.commit = AsyncMock(return_value=None)

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_a, team_b])
        mock_prisma.db.batch_ = MagicMock(return_value=mock_batcher)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"], apply_to_all_teams=True
        )
        result = await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert result["teams_updated"] == 2
        calls = mock_batcher.litellm_teamtable.update.call_args_list
        assert len(calls) == 2

        team_a_call = [c for c in calls if c.kwargs["where"]["team_id"] == "team-a"][0]
        assert "/key/generate" in team_a_call.kwargs["data"]["team_member_permissions"]
        assert "/team/daily/activity" in team_a_call.kwargs["data"]["team_member_permissions"]

        team_b_call = [c for c in calls if c.kwargs["where"]["team_id"] == "team-b"][0]
        assert "/key/delete" in team_b_call.kwargs["data"]["team_member_permissions"]
        assert "/key/update" in team_b_call.kwargs["data"]["team_member_permissions"]

    @pytest.mark.asyncio
    async def test_all_teams_skips_teams_that_already_have_permission(self, monkeypatch):
        """apply_to_all_teams: teams that already have the permission are skipped."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        team_has = self._make_team("team-has", ["/team/daily/activity", "/key/update"])
        team_missing = self._make_team("team-missing", ["/key/generate"])

        mock_batcher = MagicMock()
        mock_batcher.commit = AsyncMock(return_value=None)

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_has, team_missing])
        mock_prisma.db.batch_ = MagicMock(return_value=mock_batcher)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"], apply_to_all_teams=True
        )
        result = await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert result["teams_updated"] == 1
        calls = mock_batcher.litellm_teamtable.update.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["where"]["team_id"] == "team-missing"

    @pytest.mark.asyncio
    async def test_all_teams_pagination(self, monkeypatch):
        """apply_to_all_teams: cursor-based pagination processes multiple pages."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        page1 = [self._make_team(f"team-{i}", []) for i in range(500)]
        page2 = [self._make_team(f"team-{i}", []) for i in range(500, 502)]

        mock_batcher = MagicMock()
        mock_batcher.commit = AsyncMock(return_value=None)

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_teamtable.find_many = AsyncMock(side_effect=[page1, page2])
        mock_prisma.db.batch_ = MagicMock(return_value=mock_batcher)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"], apply_to_all_teams=True
        )
        result = await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert result["teams_updated"] == 502
        find_calls = mock_prisma.db.litellm_teamtable.find_many.call_args_list
        assert len(find_calls) == 2
        assert find_calls[1].kwargs["cursor"] == {"team_id": "team-499"}
        assert mock_batcher.commit.call_count == 2

    # --- team_ids tests ---

    @pytest.mark.asyncio
    async def test_team_ids_updates_only_specified_teams(self, monkeypatch):
        """team_ids: only the specified teams are fetched and updated."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        team_a = self._make_team("team-a", ["/key/generate"])
        team_b = self._make_team("team-b", ["/key/delete"])

        mock_batcher = MagicMock()
        mock_batcher.commit = AsyncMock(return_value=None)

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_a, team_b])
        mock_prisma.db.batch_ = MagicMock(return_value=mock_batcher)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"], team_ids=["team-a", "team-b"]
        )
        result = await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert result["teams_updated"] == 2

        # Verify find_many was called with the team_ids filter
        find_call = mock_prisma.db.litellm_teamtable.find_many.call_args
        assert find_call.kwargs["where"] == {"team_id": {"in": ["team-a", "team-b"]}}

    @pytest.mark.asyncio
    async def test_team_ids_skips_teams_that_already_have_permission(self, monkeypatch):
        """team_ids: teams that already have the permission are skipped."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        team_has = self._make_team("team-has", ["/team/daily/activity"])
        team_missing = self._make_team("team-missing", [])

        mock_batcher = MagicMock()
        mock_batcher.commit = AsyncMock(return_value=None)

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_has, team_missing])
        mock_prisma.db.batch_ = MagicMock(return_value=mock_batcher)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"], team_ids=["team-has", "team-missing"]
        )
        result = await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert result["teams_updated"] == 1
        calls = mock_batcher.litellm_teamtable.update.call_args_list
        assert calls[0].kwargs["where"]["team_id"] == "team-missing"

    @pytest.mark.asyncio
    async def test_team_ids_returns_404_for_missing_teams(self, monkeypatch):
        """If any provided team_ids don't exist, return 404."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        team_a = self._make_team("team-a", ["/key/generate"])

        mock_prisma = MagicMock()
        # Only team-a exists, team-b does not
        mock_prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_a])
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"], team_ids=["team-a", "team-b"]
        )

        with pytest.raises(HTTPException) as exc_info:
            await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert exc_info.value.status_code == 404
        assert "team-b" in str(exc_info.value.detail)

    # --- validation tests ---

    @pytest.mark.asyncio
    async def test_rejects_when_no_team_ids_and_no_apply_all(self, monkeypatch):
        """Must provide team_ids or set apply_to_all_teams=True."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        mock_prisma = MagicMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(permissions=["/team/daily/activity"])

        with pytest.raises(HTTPException) as exc_info:
            await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_when_both_team_ids_and_apply_all(self, monkeypatch):
        """Cannot set both team_ids and apply_to_all_teams."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        mock_prisma = MagicMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"],
            team_ids=["team-a"],
            apply_to_all_teams=True,
        )

        with pytest.raises(HTTPException) as exc_info:
            await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_permissions_list_is_noop(self, monkeypatch):
        """Passing an empty permissions list returns immediately with 0 updated."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        mock_prisma = MagicMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(permissions=[])
        result = await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._admin_key_dict())

        assert result["teams_updated"] == 0
        mock_prisma.db.litellm_teamtable.find_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_admin_gets_403(self, monkeypatch):
        """Non-admin users are rejected with 403."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            bulk_update_team_member_permissions,
        )
        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        mock_prisma = MagicMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        data = BulkUpdateTeamMemberPermissionsRequest(
            permissions=["/team/daily/activity"], apply_to_all_teams=True
        )

        with pytest.raises(HTTPException) as exc_info:
            await bulk_update_team_member_permissions(data=data, user_api_key_dict=self._non_admin_key_dict())

        assert exc_info.value.status_code == 403

    def test_invalid_permission_rejected_by_pydantic(self):
        """Invalid permission strings are rejected at the type level by Pydantic."""
        from pydantic import ValidationError

        from litellm.types.proxy.management_endpoints.team_endpoints import (
            BulkUpdateTeamMemberPermissionsRequest,
        )

        with pytest.raises(ValidationError):
            BulkUpdateTeamMemberPermissionsRequest(permissions=["/not/a/real/permission"])
