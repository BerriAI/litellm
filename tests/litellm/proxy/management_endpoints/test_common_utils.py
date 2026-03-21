"""
Tests for litellm/proxy/management_endpoints/common_utils.py

Specifically tests that _update_metadata_fields does not trigger premium
user checks when premium fields are present but empty.

Related: https://github.com/BerriAI/litellm/issues/20534
"""

from unittest.mock import patch

import pytest

from litellm.proxy.management_endpoints.common_utils import (
    _has_non_empty_value,
    _update_metadata_fields,
)


class TestHasNonEmptyValue:
    """Tests for the _has_non_empty_value helper."""

    def test_none_is_empty(self):
        assert _has_non_empty_value(None) is False

    def test_empty_list_is_empty(self):
        assert _has_non_empty_value([]) is False

    def test_empty_string_is_empty(self):
        assert _has_non_empty_value("") is False

    def test_blank_string_is_empty(self):
        assert _has_non_empty_value("   ") is False

    def test_non_empty_list_has_value(self):
        assert _has_non_empty_value(["policy-a"]) is True

    def test_non_empty_string_has_value(self):
        assert _has_non_empty_value("30d") is True

    def test_dict_has_value(self):
        assert _has_non_empty_value({"key": "val"}) is True

    def test_empty_dict_has_value(self):
        # empty dict is not None/list/str, so it counts as non-empty
        assert _has_non_empty_value({}) is True


class TestUpdateMetadataFieldsPremiumCheck:
    """
    Tests that _update_metadata_fields skips premium user checks for empty
    values but still enforces them for real values.

    Issue: The UI sends the full form on every team update, including premium
    fields like `policies: []`. The backend was treating these empty values
    as premium feature usage and returning 403.
    """

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_policies_skips_premium_check(self, mock_check):
        """policies: [] should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_alias": "my-team",
            "policies": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_guardrails_skips_premium_check(self, mock_check):
        """guardrails: [] should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "guardrails": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_string_team_member_key_duration_skips_premium_check(
        self, mock_check
    ):
        """team_member_key_duration: '' should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_member_key_duration": "",
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_full_ui_payload_with_empty_premium_fields_skips_premium_check(
        self, mock_check
    ):
        """A realistic UI payload with all empty premium fields should not 403."""
        updated_kv = {
            "team_id": "team-123",
            "team_alias": "renamed-team",
            "models": ["gpt-4o"],
            "max_budget": 200,
            "policies": [],
            "guardrails": [],
            "logging": [],
            "team_member_key_duration": "",
            "prompts": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_policies_triggers_premium_check(self, mock_check):
        """policies: ['real-policy'] SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "policies": ["real-policy"],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_guardrails_triggers_premium_check(self, mock_check):
        """guardrails: ['my-guardrail'] SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "guardrails": ["my-guardrail"],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_team_member_key_duration_triggers_premium_check(
        self, mock_check
    ):
        """team_member_key_duration: '30d' SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_member_key_duration": "30d",
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()


class TestCheckMemberPermission:
    """Tests for check_member_permission and _find_member_in_team."""

    def _make_user_api_key_dict(self, user_id="user-1", user_role=None):
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

        return UserAPIKeyAuth(
            user_id=user_id,
            user_role=user_role or LitellmUserRoles.INTERNAL_USER,
            api_key="sk-test",
        )

    def _make_team(self, members):
        from litellm.proxy._types import LiteLLM_TeamTable

        return LiteLLM_TeamTable(
            team_id="team-1",
            members_with_roles=members,
        )

    def test_proxy_admin_always_allowed(self):
        from litellm.proxy._types import LitellmUserRoles
        from litellm.proxy.management_endpoints.common_utils import (
            check_member_permission,
        )

        user = self._make_user_api_key_dict(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        team = self._make_team([])
        assert check_member_permission(user, team, "mcp:create") is True

    def test_team_admin_always_allowed(self):
        from litellm.proxy._types import Member
        from litellm.proxy.management_endpoints.common_utils import (
            check_member_permission,
        )

        user = self._make_user_api_key_dict(user_id="admin-1")
        team = self._make_team(
            [Member(user_id="admin-1", role="admin")]
        )
        assert check_member_permission(user, team, "mcp:create") is True

    def test_member_with_permission_allowed(self):
        from litellm.proxy._types import Member
        from litellm.proxy.management_endpoints.common_utils import (
            check_member_permission,
        )

        user = self._make_user_api_key_dict(user_id="member-1")
        team = self._make_team(
            [Member(user_id="member-1", role="user", extra_permissions=["mcp:create"])]
        )
        assert check_member_permission(user, team, "mcp:create") is True

    def test_member_without_permission_denied(self):
        from litellm.proxy._types import Member
        from litellm.proxy.management_endpoints.common_utils import (
            check_member_permission,
        )

        user = self._make_user_api_key_dict(user_id="member-1")
        team = self._make_team(
            [Member(user_id="member-1", role="user", extra_permissions=["mcp:read"])]
        )
        assert check_member_permission(user, team, "mcp:create") is False

    def test_member_with_no_permissions_denied(self):
        from litellm.proxy._types import Member
        from litellm.proxy.management_endpoints.common_utils import (
            check_member_permission,
        )

        user = self._make_user_api_key_dict(user_id="member-1")
        team = self._make_team(
            [Member(user_id="member-1", role="user")]
        )
        assert check_member_permission(user, team, "mcp:create") is False

    def test_user_not_in_team_denied(self):
        from litellm.proxy._types import Member
        from litellm.proxy.management_endpoints.common_utils import (
            check_member_permission,
        )

        user = self._make_user_api_key_dict(user_id="outsider")
        team = self._make_team(
            [Member(user_id="member-1", role="user", extra_permissions=["mcp:create"])]
        )
        assert check_member_permission(user, team, "mcp:create") is False


class TestMemberExtraPermissionsSerialization:
    """Verify Member with extra_permissions round-trips through JSON."""

    def test_member_with_permissions_roundtrip(self):
        import json

        from litellm.proxy._types import Member

        original = Member(
            user_id="user-1",
            role="user",
            extra_permissions=["mcp:create", "mcp:delete"],
        )
        serialized = json.dumps(original.model_dump())
        deserialized = Member(**json.loads(serialized))
        assert deserialized.extra_permissions == ["mcp:create", "mcp:delete"]
        assert deserialized.user_id == "user-1"
        assert deserialized.role == "user"

    def test_member_without_permissions_roundtrip(self):
        import json

        from litellm.proxy._types import Member

        original = Member(user_id="user-1", role="admin")
        serialized = json.dumps(original.model_dump())
        deserialized = Member(**json.loads(serialized))
        assert deserialized.extra_permissions is None
        assert deserialized.role == "admin"
