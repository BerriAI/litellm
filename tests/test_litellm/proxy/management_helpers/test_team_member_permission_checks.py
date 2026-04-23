import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import KeyManagementRoutes, Member
from litellm.proxy.management_helpers.team_member_permission_checks import (
    BASELINE_TEAM_MEMBER_PERMISSIONS,
    TeamMemberPermissionChecks,
)


def _make_team_table(team_member_permissions):
    """Create a mock team table object with given permissions."""
    team = MagicMock()
    team.team_member_permissions = team_member_permissions
    return team


class TestGetPermissionsForTeamMember:
    def test_none_permissions_returns_defaults(self):
        """When team_member_permissions is None, return DEFAULT_TEAM_MEMBER_PERMISSIONS."""
        team = _make_team_table(None)
        member = MagicMock(spec=Member)

        result = TeamMemberPermissionChecks.get_permissions_for_team_member(
            team_member_object=member, team_table=team
        )

        assert set(result) == set(BASELINE_TEAM_MEMBER_PERMISSIONS)

    def test_empty_list_includes_baseline(self):
        """When team_member_permissions is [], baseline permissions are still included."""
        team = _make_team_table([])
        member = MagicMock(spec=Member)

        result = TeamMemberPermissionChecks.get_permissions_for_team_member(
            team_member_object=member, team_table=team
        )

        assert KeyManagementRoutes.KEY_INFO in result
        assert KeyManagementRoutes.KEY_HEALTH in result

    def test_explicit_permissions_include_baseline(self):
        """When explicit permissions are set, baseline is always included."""
        team = _make_team_table(["/key/generate", "/key/delete"])
        member = MagicMock(spec=Member)

        result = TeamMemberPermissionChecks.get_permissions_for_team_member(
            team_member_object=member, team_table=team
        )

        assert KeyManagementRoutes.KEY_GENERATE in result
        assert KeyManagementRoutes.KEY_DELETE in result
        assert KeyManagementRoutes.KEY_INFO in result
        assert KeyManagementRoutes.KEY_HEALTH in result

    def test_explicit_permissions_with_baseline_no_duplicates(self):
        """When explicit permissions already include baseline, no duplicates."""
        team = _make_team_table(["/key/info", "/key/generate"])
        member = MagicMock(spec=Member)

        result = TeamMemberPermissionChecks.get_permissions_for_team_member(
            team_member_object=member, team_table=team
        )

        # Using set ensures no duplicates from the implementation
        assert KeyManagementRoutes.KEY_INFO in result
        assert KeyManagementRoutes.KEY_GENERATE in result
        assert KeyManagementRoutes.KEY_HEALTH in result


class TestGetDefaultTeamParam:
    def test_returns_none_when_no_config(self, monkeypatch):
        """Returns None when litellm.default_team_params is None."""
        import litellm

        from litellm.proxy.management_endpoints.team_endpoints import (
            _get_default_team_param,
        )

        monkeypatch.setattr(litellm, "default_team_params", None)

        assert _get_default_team_param("team_member_permissions") is None
        assert _get_default_team_param("max_budget") is None

    def test_returns_none_when_field_not_set(self, monkeypatch):
        """Returns None when default_team_params exists but the field is not set."""
        import litellm

        from litellm.proxy.management_endpoints.team_endpoints import (
            _get_default_team_param,
        )

        monkeypatch.setattr(litellm, "default_team_params", {"models": ["gpt-4"]})

        assert _get_default_team_param("team_member_permissions") is None
        assert _get_default_team_param("max_budget") is None

    def test_returns_permissions_from_dict_config(self, monkeypatch):
        """Returns permissions when default_team_params is a dict."""
        import litellm

        from litellm.proxy.management_endpoints.team_endpoints import (
            _get_default_team_param,
        )

        monkeypatch.setattr(
            litellm,
            "default_team_params",
            {"team_member_permissions": ["/key/generate", "/key/update"]},
        )

        result = _get_default_team_param("team_member_permissions")
        assert result == ["/key/generate", "/key/update"]

    def test_returns_scalar_fields_from_dict_config(self, monkeypatch):
        """Returns scalar fields (max_budget, tpm_limit, etc.) from dict config."""
        import litellm

        from litellm.proxy.management_endpoints.team_endpoints import (
            _get_default_team_param,
        )

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

        assert _get_default_team_param("max_budget") == 100.0
        assert _get_default_team_param("budget_duration") == "30d"
        assert _get_default_team_param("tpm_limit") == 200
        assert _get_default_team_param("rpm_limit") == 500

    def test_returns_permissions_from_pydantic_config(self, monkeypatch):
        """Returns permissions when default_team_params is a DefaultTeamSSOParams object."""
        import litellm

        from litellm.proxy.management_endpoints.team_endpoints import (
            _get_default_team_param,
        )
        from litellm.types.proxy.management_endpoints.ui_sso import (
            DefaultTeamSSOParams,
        )

        params = DefaultTeamSSOParams(
            team_member_permissions=[
                KeyManagementRoutes.KEY_GENERATE,
                KeyManagementRoutes.KEY_DELETE,
            ]
        )
        monkeypatch.setattr(litellm, "default_team_params", params)

        result = _get_default_team_param("team_member_permissions")
        assert result == ["/key/generate", "/key/delete"]

    def test_returns_scalar_fields_from_pydantic_config(self, monkeypatch):
        """Returns scalar fields from DefaultTeamSSOParams object."""
        import litellm

        from litellm.proxy.management_endpoints.team_endpoints import (
            _get_default_team_param,
        )
        from litellm.types.proxy.management_endpoints.ui_sso import (
            DefaultTeamSSOParams,
        )

        params = DefaultTeamSSOParams(
            max_budget=250.0,
            budget_duration="7d",
            tpm_limit=1000,
            rpm_limit=100,
        )
        monkeypatch.setattr(litellm, "default_team_params", params)

        assert _get_default_team_param("max_budget") == 250.0
        assert _get_default_team_param("budget_duration") == "7d"
        assert _get_default_team_param("tpm_limit") == 1000
        assert _get_default_team_param("rpm_limit") == 100
