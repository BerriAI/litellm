"""
Tests for litellm/proxy/management_endpoints/common_utils.py

Covers the fix for GitHub issue #20304:
Empty guardrails/policies arrays sent by the UI should NOT trigger the
enterprise (premium) license check, but should still be applied so that
users can intentionally clear previously-set fields.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    Member,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_team_admin,
    _org_admin_can_invite_user,
    _set_object_metadata_field,
    _team_admin_can_invite_user,
    _update_metadata_fields,
    _user_has_admin_privileges,
    _user_has_admin_view,
    admin_can_invite_user,
)


class TestUpdateMetadataFieldsEmptyCollections:
    """
    Regression tests for issue #20304.

    The UI sends empty arrays (`[]`) for enterprise-only fields like
    guardrails, policies, and logging even when the user hasn't configured
    these features.  The backend must not treat empty collections as an
    intent to use the feature, and therefore must not trigger the premium
    license check.

    However, empty collections must still be written into metadata so that
    users can intentionally clear a previously-set field (e.g. removing all
    guardrails by sending `guardrails: []`).
    """

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_list_does_not_trigger_premium_check(self, mock_premium_check):
        """Empty lists for premium fields must not trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": [],
            "policies": [],
            "logging": [],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_list_still_updates_metadata(self, mock_premium_check):
        """
        Empty lists must still be moved into metadata so users can clear
        previously-set fields (e.g. remove all guardrails).
        """
        updated_kv = {
            "team_id": "test-team",
            "guardrails": [],
            "policies": [],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        # The fields should have been moved into metadata
        assert (
            "guardrails" not in updated_kv
        ), "guardrails should be popped from top-level"
        assert "policies" not in updated_kv, "policies should be popped from top-level"
        assert updated_kv["metadata"]["guardrails"] == []
        assert updated_kv["metadata"]["policies"] == []

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_dict_does_not_trigger_premium_check(self, mock_premium_check):
        """Empty dicts for premium fields must not trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "secret_manager_settings": {},
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_empty_dict_still_updates_metadata(self, mock_premium_check):
        """
        Empty dicts must still be moved into metadata so users can clear
        previously-set fields.
        """
        updated_kv = {
            "team_id": "test-team",
            "secret_manager_settings": {},
        }
        _update_metadata_fields(updated_kv=updated_kv)
        assert (
            "secret_manager_settings" not in updated_kv
        ), "secret_manager_settings should be popped from top-level"
        assert updated_kv["metadata"]["secret_manager_settings"] == {}

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_none_value_does_not_trigger_premium_check(self, mock_premium_check):
        """None values for premium fields should be silently ignored."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": None,
            "policies": None,
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_absent_fields_do_not_trigger_premium_check(self, mock_premium_check):
        """Fields not present in the dict should not trigger premium check."""
        updated_kv = {
            "team_id": "test-team",
            "team_alias": "example-team",
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_non_empty_list_triggers_premium_check(self, mock_premium_check):
        """Non-empty lists for premium fields should trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": ["my-guardrail"],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_non_empty_value_triggers_premium_check(self, mock_premium_check):
        """Non-empty string values for premium fields should trigger the premium check."""
        updated_kv = {
            "team_id": "test-team",
            "tags": ["production"],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_non_empty_list_updates_metadata(self, mock_premium_check):
        """Non-empty lists should be moved into metadata."""
        updated_kv = {
            "team_id": "test-team",
            "guardrails": ["my-guardrail"],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        assert "guardrails" not in updated_kv
        assert updated_kv["metadata"]["guardrails"] == ["my-guardrail"]

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_false_boolean_does_not_trigger_premium_check(self, mock_premium_check):
        """
        Regression #30285: /team/update sends disable_global_guardrails=False
        (the UI's unchanged default). A falsy boolean must not trigger the
        premium check, so non-premium users are not wrongly 403'd.
        """
        updated_kv = {"team_id": "test-team", "disable_global_guardrails": False}
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_false_boolean_still_updates_metadata(self, mock_premium_check):
        """A falsy boolean must still be moved into metadata so it persists."""
        updated_kv = {"team_id": "test-team", "disable_global_guardrails": False}
        _update_metadata_fields(updated_kv=updated_kv)
        assert "disable_global_guardrails" not in updated_kv
        assert updated_kv["metadata"]["disable_global_guardrails"] is False

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_true_boolean_triggers_premium_check(self, mock_premium_check):
        """Control: enabling the premium feature (True) still requires a license."""
        updated_kv = {"team_id": "test-team", "disable_global_guardrails": True}
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_called()

    @patch("litellm.proxy.management_endpoints.common_utils._premium_user_check")
    def test_ui_typical_payload_does_not_trigger_premium_check(
        self, mock_premium_check
    ):
        """
        Simulate the exact payload the UI sends when no enterprise features
        are configured.  This must NOT trigger the premium check.
        """
        # This is the payload structure the UI sends (from issue #20304)
        updated_kv = {
            "team_id": "67848772-1a8b-4343-938c-17e60f1db860",
            "team_alias": "example-team",
            "models": ["gpt-4"],
            "metadata": {
                "guardrails": [],
                "logging": [],
            },
            "policies": [],
        }
        _update_metadata_fields(updated_kv=updated_kv)
        mock_premium_check.assert_not_called()


class TestUserHasAdminView:
    """Tests for _user_has_admin_view function."""

    @pytest.mark.parametrize(
        "user_role,expected",
        [
            (LitellmUserRoles.PROXY_ADMIN, True),
            (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, True),
            (LitellmUserRoles.INTERNAL_USER, False),
            (LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, False),
        ],
    )
    def test_user_has_admin_view_by_role(self, user_role, expected):
        """Parametrized test: admin roles return True, non-admin return False."""
        mock_auth = MagicMock()
        mock_auth.user_role = user_role
        assert _user_has_admin_view(mock_auth) == expected

    def test_user_has_admin_view_with_user_api_key_auth(self):
        """Test with actual UserAPIKeyAuth object."""
        auth_admin = UserAPIKeyAuth(
            user_id="u1",
            api_key="sk-xxx",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        auth_user = UserAPIKeyAuth(
            user_id="u2",
            api_key="sk-yyy",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        assert _user_has_admin_view(auth_admin) is True
        assert _user_has_admin_view(auth_user) is False


class TestIsUserTeamAdmin:
    """Tests for _is_user_team_admin function."""

    @pytest.mark.parametrize(
        "members_with_roles,user_id,expected",
        [
            (
                [Member(user_id="u1", role="admin")],
                "u1",
                True,
            ),
            (
                [Member(user_id="u1", role="user")],
                "u1",
                False,
            ),
            (
                [
                    Member(user_id="u2", role="admin"),
                    Member(user_id="u1", role="admin"),
                ],
                "u1",
                True,
            ),
            ([], "u1", False),
        ],
    )
    def test_is_user_team_admin_parametrized(
        self, members_with_roles, user_id, expected
    ):
        """Parametrized test: user is team admin only when in members_with_roles with admin role."""
        mock_auth = MagicMock()
        mock_auth.user_id = user_id
        team = LiteLLM_TeamTable(
            team_id="team-1",
            members_with_roles=members_with_roles,
        )
        assert _is_user_team_admin(mock_auth, team) == expected

    def test_is_user_team_admin_user_not_in_team(self):
        """Test returns False when user is not in team members."""
        auth = UserAPIKeyAuth(user_id="u99", api_key="sk-x", user_role=None)
        team = LiteLLM_TeamTable(
            team_id="team-1",
            members_with_roles=[Member(user_id="u1", role="admin")],
        )
        assert _is_user_team_admin(auth, team) is False


class TestOrgAdminCanInviteUser:
    """Tests for _org_admin_can_invite_user function."""

    def _make_membership(self, org_id: str, user_role: str):
        now = datetime.now(timezone.utc)
        return LiteLLM_OrganizationMembershipTable(
            user_id="u",
            organization_id=org_id,
            user_role=user_role,
            created_at=now,
            updated_at=now,
        )

    @pytest.mark.parametrize(
        "admin_orgs,target_orgs,expected",
        [
            (["org1"], ["org1"], True),
            (["org1", "org2"], ["org2"], True),
            (["org1"], ["org2"], False),
            ([], ["org1"], False),
            (["org1"], [], False),
        ],
    )
    def test_org_admin_can_invite_user_parametrized(
        self, admin_orgs, target_orgs, expected
    ):
        """Parametrized test: can invite when target is in org where admin has ORG_ADMIN role."""
        admin_user = LiteLLM_UserTable(
            user_id="admin",
            organization_memberships=[
                self._make_membership(oid, LitellmUserRoles.ORG_ADMIN.value)
                for oid in admin_orgs
            ],
        )
        target_user = LiteLLM_UserTable(
            user_id="target",
            organization_memberships=[
                self._make_membership(oid, LitellmUserRoles.INTERNAL_USER.value)
                for oid in target_orgs
            ],
        )
        assert _org_admin_can_invite_user(admin_user, target_user) == expected

    def test_org_admin_can_invite_user_no_shared_org(self):
        """Test returns False when admin has no org admin role."""
        admin_user = LiteLLM_UserTable(
            user_id="admin",
            organization_memberships=[
                self._make_membership("org1", LitellmUserRoles.INTERNAL_USER.value),
            ],
        )
        target_user = LiteLLM_UserTable(
            user_id="target",
            organization_memberships=[
                self._make_membership("org1", LitellmUserRoles.INTERNAL_USER.value),
            ],
        )
        assert _org_admin_can_invite_user(admin_user, target_user) is False


class TestTeamAdminCanInviteUser:
    """Tests for _team_admin_can_invite_user async function."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "admin_teams,target_teams,user_is_admin_in,expected",
        [
            (["t1"], ["t1"], ["t1"], True),
            (["t1", "t2"], ["t2"], ["t1", "t2"], True),
            (["t1"], ["t2"], ["t1"], False),
        ],
    )
    async def test_team_admin_can_invite_user_parametrized(
        self, admin_teams, target_teams, user_is_admin_in, expected
    ):
        """Parametrized test: can invite when target shares a team where user is admin."""
        mock_prisma = MagicMock()
        mock_auth = MagicMock()
        mock_auth.user_id = "admin"

        admin_user = LiteLLM_UserTable(user_id="admin", teams=admin_teams)
        target_user = LiteLLM_UserTable(user_id="target", teams=target_teams)

        def make_team(tid, is_admin):
            m = [{"user_id": "admin", "role": "admin"}] if is_admin else []
            obj = MagicMock()
            obj.team_id = tid
            obj.model_dump = lambda: {"team_id": tid, "members_with_roles": m}
            return obj

        teams = [make_team(tid, tid in user_is_admin_in) for tid in admin_teams]
        mock_prisma.db.litellm_teamtable.find_many = AsyncMock(return_value=teams)

        result = await _team_admin_can_invite_user(
            user_api_key_dict=mock_auth,
            admin_user_obj=admin_user,
            target_user_obj=target_user,
            prisma_client=mock_prisma,
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_team_admin_can_invite_user_no_shared_team(self):
        """Test returns False when admin and target share no team."""
        mock_prisma = MagicMock()
        mock_auth = MagicMock()
        mock_auth.user_id = "admin"
        admin_user = LiteLLM_UserTable(user_id="admin", teams=[])
        target_user = LiteLLM_UserTable(user_id="target", teams=["t1"])

        result = await _team_admin_can_invite_user(
            user_api_key_dict=mock_auth,
            admin_user_obj=admin_user,
            target_user_obj=target_user,
            prisma_client=mock_prisma,
        )
        assert result is False


class TestUserHasAdminPrivileges:
    """Tests for _user_has_admin_privileges async function."""

    @pytest.mark.asyncio
    async def test_proxy_admin_has_privileges(self):
        """Proxy admin always has admin privileges."""
        auth = UserAPIKeyAuth(
            user_id="admin",
            api_key="sk-x",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        result = await _user_has_admin_privileges(
            user_api_key_dict=auth,
            prisma_client=None,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_non_admin_no_prisma_returns_false(self):
        """Non-admin with no prisma connection has no privileges."""
        auth = UserAPIKeyAuth(
            user_id="user1",
            api_key="sk-x",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        result = await _user_has_admin_privileges(
            user_api_key_dict=auth,
            prisma_client=None,
        )
        assert result is False


class TestAdminCanInviteUser:
    """Tests for admin_can_invite_user async function."""

    @pytest.mark.asyncio
    async def test_proxy_admin_can_invite_any_user(self):
        """Proxy admin can invite any user regardless of org/team."""
        auth = UserAPIKeyAuth(
            user_id="admin",
            api_key="sk-x",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        result = await admin_can_invite_user(
            target_user_id="any-user",
            user_api_key_dict=auth,
            prisma_client=None,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_non_admin_cannot_invite_without_prisma(self):
        """Non-admin with no prisma cannot invite."""
        auth = UserAPIKeyAuth(
            user_id="user1",
            api_key="sk-x",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        result = await admin_can_invite_user(
            target_user_id="other-user",
            user_api_key_dict=auth,
            prisma_client=None,
        )
        assert result is False


class TestSetObjectMetadataField:
    """Tests for _set_object_metadata_field function."""

    @pytest.mark.parametrize(
        "field_name,value,should_call_premium",
        [
            ("guardrails", ["g1"], True),
            ("model_rpm_limit", {"gpt-4": 10}, False),
        ],
    )
    def test_set_object_metadata_field_parametrized(
        self, field_name, value, should_call_premium
    ):
        """Parametrized test: premium fields trigger _premium_user_check."""
        team = LiteLLM_TeamTable(team_id="t1", metadata={})
        with patch(
            "litellm.proxy.management_endpoints.common_utils._premium_user_check"
        ) as mock_premium:
            _set_object_metadata_field(team, field_name, value)
            if should_call_premium:
                mock_premium.assert_called_once()
            else:
                mock_premium.assert_not_called()
        assert team.metadata[field_name] == value

    def test_set_object_metadata_field_initializes_metadata_if_none(self):
        """Test initializes metadata dict when object has None."""
        team = LiteLLM_TeamTable(team_id="t1", metadata=None)
        with patch(
            "litellm.proxy.management_endpoints.common_utils._premium_user_check"
        ):
            _set_object_metadata_field(team, "model_rpm_limit", {"x": 1})
        assert team.metadata == {"model_rpm_limit": {"x": 1}}

    def test_mcp_rpm_limit_is_hoisted_into_metadata(self):
        """
        Per-MCP-server rpm limits are stored in the metadata JSON column, not a
        dedicated DB column. The key/team management endpoints rely on
        LiteLLM_ManagementEndpoint_MetadataFields to move the request field into
        metadata; this regression guards that mcp_rpm_limit is in that list and
        round-trips through the same loop the endpoints use.
        """
        from litellm.proxy._types import LiteLLM_ManagementEndpoint_MetadataFields

        assert "mcp_rpm_limit" in LiteLLM_ManagementEndpoint_MetadataFields

        from types import SimpleNamespace

        team = LiteLLM_TeamTable(team_id="t1", metadata={})
        mcp_rpm_limit = {"github": 100}
        data = SimpleNamespace(mcp_rpm_limit=mcp_rpm_limit)

        with patch(
            "litellm.proxy.management_endpoints.common_utils._premium_user_check"
        ):
            for field in LiteLLM_ManagementEndpoint_MetadataFields:
                if getattr(data, field, None) is not None:
                    _set_object_metadata_field(team, field, getattr(data, field))

        assert team.metadata["mcp_rpm_limit"] == mcp_rpm_limit


class TestRequireCallerUserIdForNonAdmin:
    """
    Security regression: service-account keys (user_id=None) must not bypass
    the non-admin scoping branch on analytics endpoints.
    """

    def test_returns_user_id_when_present(self):
        from litellm.proxy.management_endpoints.common_utils import (
            require_caller_user_id_for_non_admin,
        )

        key_dict = UserAPIKeyAuth(
            user_id="user-abc",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        assert require_caller_user_id_for_non_admin(key_dict) == "user-abc"

    def test_raises_403_when_user_id_is_none(self):
        from fastapi import HTTPException

        from litellm.proxy.management_endpoints.common_utils import (
            require_caller_user_id_for_non_admin,
        )

        # Simulates a service-account key (user_id forced to None at key creation)
        service_account_key = UserAPIKeyAuth(
            user_id=None,
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        with pytest.raises(HTTPException) as exc_info:
            require_caller_user_id_for_non_admin(service_account_key)

        assert exc_info.value.status_code == 403
        assert "Service-account keys" in str(exc_info.value.detail)


class TestValidateFiniteSpend:
    """`validate_finite_spend` rejects NaN/±inf so a non-finite spend cannot
    bypass `spend >= max_budget` enforcement (NaN/-inf compare false)."""

    def test_none_is_allowed(self):
        from litellm.proxy.management_endpoints.common_utils import (
            validate_finite_spend,
        )

        assert validate_finite_spend(None) is None

    def test_finite_value_is_allowed(self):
        from litellm.proxy.management_endpoints.common_utils import (
            validate_finite_spend,
        )

        assert validate_finite_spend(0.0) is None
        assert validate_finite_spend(12.5) is None
        # Negative spend is intentionally allowed. Admins may set a negative
        # spend counter to grant an entity extra allowance for the current
        # budget period only (e.g. a large one-time spend grant), effectively
        # raising their headroom without raising the recurring budget ceiling.
        # Future changes should continue to allow negative spend counters.
        assert validate_finite_spend(-50.0) is None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_is_rejected(self, bad):
        from fastapi import HTTPException

        from litellm.proxy.management_endpoints.common_utils import (
            validate_finite_spend,
        )

        with pytest.raises(HTTPException) as exc_info:
            validate_finite_spend(bad)
        assert exc_info.value.status_code == 400


class TestValidateFiniteSpendErrorDetail:
    """The 400 for non-finite spend must carry the exact {"error": <msg>} body."""

    def test_rejection_detail_is_exact(self):
        from fastapi import HTTPException

        from litellm.proxy.management_endpoints.common_utils import (
            validate_finite_spend,
        )

        with pytest.raises(HTTPException) as exc_info:
            validate_finite_spend(float("nan"))

        assert exc_info.value.detail == {
            "error": "spend must be a finite number. Received: nan"
        }


class TestRequireCallerUserIdErrorDetail:
    """The 403 for a service-account key must carry the exact error body."""

    def test_rejection_detail_is_exact(self):
        from fastapi import HTTPException

        from litellm.proxy.management_endpoints.common_utils import (
            require_caller_user_id_for_non_admin,
        )

        service_account_key = UserAPIKeyAuth(
            user_id=None,
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        with pytest.raises(HTTPException) as exc_info:
            require_caller_user_id_for_non_admin(service_account_key)

        assert exc_info.value.detail == {
            "error": "Service-account keys cannot query user analytics. Use a user-bound key, or call as a proxy admin."
        }


class TestCheckPassthroughRoutesCallerPermission:
    """Only proxy admins may set allowed_passthrough_routes (top-level or under
    metadata); non-admins get a 403 naming the entity."""

    def _non_admin(self):
        return UserAPIKeyAuth(
            user_id="u1", api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER
        )

    def test_top_level_routes_rejected_with_default_entity(self):
        from fastapi import HTTPException
        from pydantic import BaseModel

        from litellm.proxy.management_endpoints.common_utils import (
            _check_passthrough_routes_caller_permission,
        )

        class _RouteData(BaseModel):
            allowed_passthrough_routes: list | None = None
            metadata: dict | None = None

        data = _RouteData(allowed_passthrough_routes=["/v1/foo"])
        with pytest.raises(HTTPException) as exc_info:
            _check_passthrough_routes_caller_permission(data, self._non_admin())

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == {
            "error": "Only proxy admins can set `allowed_passthrough_routes` on a key."
        }

    def test_metadata_routes_rejected_with_default_entity(self):
        from fastapi import HTTPException
        from pydantic import BaseModel

        from litellm.proxy.management_endpoints.common_utils import (
            _check_passthrough_routes_caller_permission,
        )

        class _RouteData(BaseModel):
            allowed_passthrough_routes: list | None = None
            metadata: dict | None = None

        data = _RouteData(metadata={"allowed_passthrough_routes": ["/v1/foo"]})
        with pytest.raises(HTTPException) as exc_info:
            _check_passthrough_routes_caller_permission(data, self._non_admin())

        assert exc_info.value.detail == {
            "error": "Only proxy admins can set `metadata.allowed_passthrough_routes` on a key."
        }

    def test_tolerates_data_missing_passthrough_and_metadata_fields(self):
        from pydantic import BaseModel

        from litellm.proxy.management_endpoints.common_utils import (
            _check_passthrough_routes_caller_permission,
        )

        class _Bare(BaseModel):
            unrelated: str = "x"

        assert (
            _check_passthrough_routes_caller_permission(_Bare(), self._non_admin())
            is None
        )


class TestIsUserOrgAdminForTeam:
    """The caller must be looked up with its exact identity; a nulled or omitted
    lookup argument would silently mis-resolve org-admin status."""

    @pytest.mark.asyncio
    async def test_get_user_object_called_with_caller_identity(self):
        from litellm.proxy.management_endpoints.common_utils import (
            _is_user_org_admin_for_team,
        )

        team = LiteLLM_TeamTable(
            team_id="t1", organization_id="org1", members_with_roles=[]
        )
        key = UserAPIKeyAuth(
            user_id="u1", api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER
        )
        fake_prisma, fake_cache, fake_logging = MagicMock(), MagicMock(), MagicMock()
        mock_get_user = AsyncMock(return_value=None)

        with patch(
            "litellm.proxy.proxy_server.prisma_client", fake_prisma
        ), patch(
            "litellm.proxy.proxy_server.user_api_key_cache", fake_cache
        ), patch(
            "litellm.proxy.proxy_server.proxy_logging_obj", fake_logging
        ), patch(
            "litellm.proxy.auth.auth_checks.get_user_object", mock_get_user
        ):
            result = await _is_user_org_admin_for_team(key, team)

        assert result is False
        mock_get_user.assert_awaited_once_with(
            user_id="u1",
            prisma_client=fake_prisma,
            user_api_key_cache=fake_cache,
            user_id_upsert=False,
            proxy_logging_obj=fake_logging,
        )


class TestTeamMemberHasPermission:
    def test_requires_caller_to_be_a_team_member(self):
        from litellm.proxy.management_endpoints.common_utils import (
            _team_member_has_permission,
        )

        team = LiteLLM_TeamTable(
            team_id="t1",
            team_member_permissions=["/key/generate"],
            members_with_roles=[Member(user_id="someone-else", role="user")],
        )
        key = UserAPIKeyAuth(
            user_id="u1", api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER
        )
        assert _team_member_has_permission(key, team, "/key/generate") is False


class TestUserHasAdminPrivilegesGuard:
    @pytest.mark.asyncio
    async def test_no_user_lookup_when_prisma_is_none(self):
        """With no DB the guard short-circuits before any user lookup."""
        auth = UserAPIKeyAuth(
            user_id="user1", api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER
        )
        mock_get_user = AsyncMock(return_value=None)
        with patch("litellm.proxy.auth.auth_checks.get_user_object", mock_get_user):
            result = await _user_has_admin_privileges(
                user_api_key_dict=auth, prisma_client=None
            )
        assert result is False
        mock_get_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_org_admin_membership_grants_privileges(self):
        """With DB + user_id present, an ORG_ADMIN membership yields True."""
        auth = UserAPIKeyAuth(
            user_id="user1", api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER
        )
        now = datetime.now(timezone.utc)
        user_obj = LiteLLM_UserTable(
            user_id="user1",
            organization_memberships=[
                LiteLLM_OrganizationMembershipTable(
                    user_id="user1",
                    organization_id="org1",
                    user_role=LitellmUserRoles.ORG_ADMIN.value,
                    created_at=now,
                    updated_at=now,
                )
            ],
        )
        mock_get_user = AsyncMock(return_value=user_obj)
        with patch("litellm.proxy.auth.auth_checks.get_user_object", mock_get_user):
            result = await _user_has_admin_privileges(
                user_api_key_dict=auth, prisma_client=MagicMock()
            )
        assert result is True


class TestAdminCanInviteUserGuard:
    @pytest.mark.asyncio
    async def test_no_user_lookup_when_prisma_is_none(self):
        auth = UserAPIKeyAuth(
            user_id="admin1", api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER
        )
        mock_get_user = AsyncMock(return_value=None)
        with patch("litellm.proxy.auth.auth_checks.get_user_object", mock_get_user):
            result = await admin_can_invite_user(
                target_user_id="target1",
                user_api_key_dict=auth,
                prisma_client=None,
            )
        assert result is False
        mock_get_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_org_admin_can_invite_user_in_shared_org(self):
        now = datetime.now(timezone.utc)
        auth = UserAPIKeyAuth(
            user_id="admin1", api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER
        )

        def membership(role):
            return LiteLLM_OrganizationMembershipTable(
                user_id="x",
                organization_id="org1",
                user_role=role,
                created_at=now,
                updated_at=now,
            )

        admin_obj = LiteLLM_UserTable(
            user_id="admin1",
            organization_memberships=[membership(LitellmUserRoles.ORG_ADMIN.value)],
        )
        target_obj = LiteLLM_UserTable(
            user_id="target1",
            organization_memberships=[membership(LitellmUserRoles.INTERNAL_USER.value)],
        )
        mock_get_user = AsyncMock(side_effect=[admin_obj, target_obj])
        with patch("litellm.proxy.auth.auth_checks.get_user_object", mock_get_user):
            result = await admin_can_invite_user(
                target_user_id="target1",
                user_api_key_dict=auth,
                prisma_client=MagicMock(),
            )
        assert result is True


class TestTeamAdminCanInviteUserQuery:
    @pytest.mark.asyncio
    async def test_find_many_queries_admin_teams_with_exact_where(self):
        mock_prisma = MagicMock()
        mock_auth = MagicMock()
        mock_auth.user_id = "admin"
        admin_user = LiteLLM_UserTable(user_id="admin", teams=["t1", "t2"])
        target_user = LiteLLM_UserTable(user_id="target", teams=["t2"])

        def make_team(tid):
            obj = MagicMock()
            obj.team_id = tid
            obj.model_dump = lambda: {
                "team_id": tid,
                "members_with_roles": [{"user_id": "admin", "role": "admin"}],
            }
            return obj

        find_many = AsyncMock(return_value=[make_team("t1"), make_team("t2")])
        mock_prisma.db.litellm_teamtable.find_many = find_many

        await _team_admin_can_invite_user(
            user_api_key_dict=mock_auth,
            admin_user_obj=admin_user,
            target_user_obj=target_user,
            prisma_client=mock_prisma,
        )

        find_many.assert_awaited_once_with(where={"team_id": {"in": ["t1", "t2"]}})


class TestSetObjectMetadataFieldPremiumArg:
    def test_premium_check_receives_the_field_name(self):
        team = LiteLLM_TeamTable(team_id="t1", metadata={})
        with patch(
            "litellm.proxy.management_endpoints.common_utils._premium_user_check"
        ) as mock_premium:
            _set_object_metadata_field(team, "guardrails", ["g1"])
            mock_premium.assert_called_once_with("guardrails")


class TestUpdateMetadataFieldMove:
    def test_none_valued_field_is_not_moved_into_metadata(self):
        """A None value must leave the field untouched (guard requires non-None)."""
        from litellm.proxy.management_endpoints.common_utils import (
            _update_metadata_field,
        )

        updated_kv = {"guardrails": None}
        _update_metadata_field(updated_kv=updated_kv, field_name="guardrails")
        assert updated_kv == {"guardrails": None}

    def test_set_premium_field_is_moved_into_metadata(self):
        updated_kv = {"guardrails": ["g1"]}
        with patch(
            "litellm.proxy.management_endpoints.common_utils._premium_user_check"
        ):
            _update_metadata_fields(updated_kv)
        assert "guardrails" not in updated_kv
        assert updated_kv["metadata"]["guardrails"] == ["g1"]
