"""Exhaustive tests for the pure access-decision function.

Each test names one specific reason a team-admin patch should be denied (or
allowed). Together they pin the security contract: changing this code with
the tests in place should fail a case named after what you broke.
"""

from __future__ import annotations

import pytest

from litellm.models.credentials import CredentialInfo
from litellm.proxy.credential_endpoints.access_decision import (
    Allow,
    Deny,
    decide_credential_patch,
)

_EXISTING_INFO = {
    "credential_type": "logging",
    "description": "tenant Langfuse",
    "host": "https://cloud.langfuse.com",
    "access": {"teams": ["team-A", "team-B"], "orgs": ["org-1"], "global": False},
}


def _info(value):
    return None if value is None else CredentialInfo.model_validate(value)


def _decision(
    *,
    is_proxy_admin: bool = False,
    caller_team_admin_ids: frozenset[str] = frozenset({"team-T"}),
    existing_info=_EXISTING_INFO,
    patch_info=None,
    patch_values=None,
    patch_name_changed: bool = False,
):
    return decide_credential_patch(
        is_proxy_admin=is_proxy_admin,
        caller_team_admin_ids=caller_team_admin_ids,
        existing_info=_info(existing_info),
        patch_info=_info(patch_info),
        patch_values=patch_values,
        patch_name_changed=patch_name_changed,
    )


class TestProxyAdminAllow:
    def test_proxy_admin_allowed_on_value_change(self):
        d = _decision(
            is_proxy_admin=True,
            patch_values={"api_key": "rotated"},
            patch_info={"credential_type": "logging"},
        )
        assert isinstance(d, Allow)

    def test_proxy_admin_allowed_on_global_flip(self):
        d = _decision(
            is_proxy_admin=True,
            patch_info={"access": {"global": True}},
        )
        assert isinstance(d, Allow)

    def test_proxy_admin_allowed_on_rename(self):
        d = _decision(is_proxy_admin=True, patch_name_changed=True)
        assert isinstance(d, Allow)


class TestTeamAdminAllow:
    def test_appending_own_team_id(self):
        d = _decision(
            patch_info={
                "access": {"teams": ["team-A", "team-B", "team-T"]},
            },
        )
        assert isinstance(d, Allow)

    def test_appending_only_own_team_id_when_no_prior_teams(self):
        existing = {**_EXISTING_INFO, "access": {"global": False}}
        d = _decision(
            existing_info=existing,
            patch_info={"access": {"teams": ["team-T"]}},
        )
        assert isinstance(d, Allow)

    def test_idempotent_when_already_granted(self):
        existing = {**_EXISTING_INFO, "access": {"teams": ["team-T"]}}
        d = _decision(
            existing_info=existing,
            patch_info={"access": {"teams": ["team-T"]}},
        )
        assert isinstance(d, Allow)


class TestTeamAdminDeny:
    def test_not_team_admin_anywhere(self):
        d = _decision(
            caller_team_admin_ids=frozenset(),
            patch_info={"access": {"teams": ["team-T"]}},
        )
        assert isinstance(d, Deny)
        assert "proxy admin" in d.reason

    def test_rename(self):
        d = _decision(
            patch_name_changed=True,
            patch_info={"access": {"teams": ["team-T"]}},
        )
        assert isinstance(d, Deny)
        assert "credential_name" in d.reason

    def test_changing_credential_values(self):
        d = _decision(
            patch_values={"api_key": "stolen"},
            patch_info={"access": {"teams": ["team-T"]}},
        )
        assert isinstance(d, Deny)
        assert "credential_values" in d.reason

    def test_empty_patch(self):
        d = _decision(patch_info=None)
        assert isinstance(d, Deny)

    @pytest.mark.parametrize(
        "field", ["credential_type", "description", "host", "endpoint"]
    )
    def test_changing_immutable_info_field(self, field):
        d = _decision(patch_info={field: "x", "access": {"teams": ["team-T"]}})
        assert isinstance(d, Deny)
        assert field in d.reason

    def test_patch_info_with_unknown_keys(self):
        d = _decision(patch_info={"weird_field": 1})
        assert isinstance(d, Deny)
        assert "weird_field" in d.reason

    def test_flipping_global(self):
        d = _decision(patch_info={"access": {"global": True}})
        assert isinstance(d, Deny)
        assert "global" in d.reason

    def test_editing_orgs(self):
        d = _decision(patch_info={"access": {"orgs": ["org-new"]}})
        assert isinstance(d, Deny)
        assert "orgs" in d.reason

    def test_no_op_resend_of_existing_global_is_allowed(self):
        """The UI's Edit-access modal always sends the FULL access object
        (so unchecking a team revokes). A non-admin re-sending the existing
        global=false alongside their team patch must NOT be rejected as if
        they were trying to flip the toggle.

        Before this fix the decider checked only "is global_ in the patch?"
        which broke every UI save that included the unchanged global toggle
        plus a team edit.
        """
        d = _decision(
            patch_info={
                "access": {
                    "global": False,
                    "teams": ["team-A", "team-B", "team-T"],
                    "orgs": ["org-1"],
                }
            }
        )
        assert isinstance(d, Allow)

    def test_no_op_resend_of_existing_orgs_is_allowed(self):
        """Same shape as global: a patch that includes the unchanged orgs
        list alongside a team edit must pass."""
        d = _decision(
            patch_info={
                "access": {
                    "global": False,
                    "teams": ["team-A", "team-B", "team-T"],
                    "orgs": ["org-1"],
                }
            }
        )
        assert isinstance(d, Allow)

    def test_attempt_to_flip_global_when_existing_is_false(self):
        """Direct flip from stored False to True still rejected."""
        d = _decision(
            patch_info={"access": {"global": True, "teams": ["team-A", "team-B"]}}
        )
        assert isinstance(d, Deny)
        assert "global" in d.reason

    def test_attempt_to_flip_global_when_existing_is_true(self):
        """Direct flip from stored True to False still rejected (it's still
        a global mutation; only proxy-admin can change destination-wide reach)."""
        existing = {
            **_EXISTING_INFO,
            "access": {**_EXISTING_INFO["access"], "global": True},
        }
        d = _decision(
            existing_info=existing,
            patch_info={"access": {"global": False, "teams": ["team-A", "team-B"]}},
        )
        assert isinstance(d, Deny)
        assert "global" in d.reason

    def test_attempt_to_change_orgs_is_rejected(self):
        """Adding an org_id different from stored is still rejected."""
        d = _decision(
            patch_info={
                "access": {"orgs": ["org-1", "org-2"], "teams": ["team-A", "team-B"]}
            }
        )
        assert isinstance(d, Deny)
        assert "orgs" in d.reason

    def test_adding_foreign_team_id(self):
        """foreign team_ids in the patch ARE caller input -- safe to echo."""
        d = _decision(
            patch_info={
                "access": {"teams": ["team-A", "team-B", "team-foreign"]},
            },
        )
        assert isinstance(d, Deny)
        assert d.from_user_input is True
        assert "team-foreign" in d.reason

    def test_removing_foreign_team_grant(self):
        """team-admin may NOT remove a team they don't admin.

        Reason intentionally does NOT echo the stored team_id (it's not
        caller-typed; surfacing it would leak access list membership).
        """
        d = _decision(
            patch_info={
                "access": {"teams": ["team-A", "team-T"]},
            },
        )
        assert isinstance(d, Deny)
        assert d.from_user_input is False
        assert "team-B" not in d.reason
        assert "may only revoke" in d.reason

    def test_replacing_teams_wholesale_with_foreign_remaining(self):
        """Wholesale replacement that removes foreign grants is rejected.

        The stored team_ids that were dropped must not appear in the reason --
        they're access list contents, not caller input.
        """
        d = _decision(patch_info={"access": {"teams": ["team-T"]}})
        assert isinstance(d, Deny)
        assert d.from_user_input is False
        assert "team-A" not in d.reason
        assert "team-B" not in d.reason


class TestTeamAdminRevoke:
    """A team-admin may revoke their OWN team's grant; never another's."""

    def test_revoking_own_team_is_allowed(self):
        existing = {
            **_EXISTING_INFO,
            "access": {"teams": ["team-A", "team-T"]},
        }
        d = _decision(
            existing_info=existing,
            patch_info={"access": {"teams": ["team-A"]}},
        )
        assert isinstance(d, Allow)

    def test_revoking_own_team_when_only_grant(self):
        """Saving an empty teams list when the caller was the sole grant."""
        existing = {**_EXISTING_INFO, "access": {"teams": ["team-T"]}}
        d = _decision(
            existing_info=existing,
            patch_info={"access": {"teams": []}},
        )
        assert isinstance(d, Allow)

    def test_revoke_attempt_on_foreign_team_denied(self):
        """A patch that removes a foreign team is still rejected, even if
        the caller is also revoking their own. The foreign team_id MUST
        NOT appear in the reason (stored access list content)."""
        existing = {
            **_EXISTING_INFO,
            "access": {"teams": ["team-A", "team-B", "team-T"]},
        }
        d = _decision(
            existing_info=existing,
            patch_info={"access": {"teams": ["team-A"]}},  # drops team-B AND team-T
        )
        assert isinstance(d, Deny)
        assert d.from_user_input is False
        assert "team-B" not in d.reason


class TestTeamAdminMultipleTeams:
    def test_can_add_multiple_own_team_ids(self):
        d = _decision(
            caller_team_admin_ids=frozenset({"team-T1", "team-T2"}),
            patch_info={
                "access": {"teams": ["team-A", "team-B", "team-T1", "team-T2"]},
            },
        )
        assert isinstance(d, Allow)

    def test_one_own_one_foreign_is_deny(self):
        d = _decision(
            caller_team_admin_ids=frozenset({"team-T1"}),
            patch_info={
                "access": {"teams": ["team-A", "team-B", "team-T1", "team-T2"]},
            },
        )
        assert isinstance(d, Deny)
        assert "team-T2" in d.reason
