"""
Tests for managed-resource tenant isolation helpers.
"""

import pytest

from litellm.llms.base_llm.managed_resources.isolation import (
    build_owner_filter,
    can_access_resource,
    get_managed_resource_owner_id,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


# ---------------------------------------------------------------------------
# build_owner_filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
)
def test_owner_filter_admin_unscoped(role):
    assert build_owner_filter(UserAPIKeyAuth(user_role=role)) == {}


def test_owner_filter_user_scoped_to_user_id():
    user = UserAPIKeyAuth(user_id="alice")
    assert build_owner_filter(user) == {"created_by": "alice"}


def test_owner_filter_service_account_scoped_to_team():
    service_account = UserAPIKeyAuth(team_id="team-eng")
    assert build_owner_filter(service_account) == {"team_id": "team-eng"}


def test_owner_filter_user_with_team_returns_or_filter():
    """List view must mirror `can_access_resource`: a user-keyed caller in a
    team can also access team-shared resources, so the listing returns both
    their own records and team records via an OR filter."""
    user = UserAPIKeyAuth(user_id="alice", team_id="team-eng")
    assert build_owner_filter(user) == {
        "OR": [
            {"created_by": "alice"},
            {"team_id": "team-eng"},
        ]
    }


def test_owner_filter_no_identity_returns_none():
    """A caller with no admin role and no identifying ids must be denied so
    the listing path can refuse the query rather than fall through to an
    unscoped fetch."""
    assert build_owner_filter(UserAPIKeyAuth()) is None


def test_owner_filter_falls_back_to_end_user_id():
    """Custom-auth callers (only end_user_id set) scope to their end-user id,
    matching the ownership stamped at create time."""
    end_user = UserAPIKeyAuth(end_user_id="customer-1")
    assert build_owner_filter(end_user) == {"created_by": "customer-1"}


def test_owner_filter_user_id_takes_precedence_over_end_user_id():
    user = UserAPIKeyAuth(user_id="alice", end_user_id="customer-1")
    assert build_owner_filter(user) == {"created_by": "alice"}


def test_owner_filter_end_user_with_team_returns_or_filter():
    end_user = UserAPIKeyAuth(end_user_id="customer-1", team_id="team-eng")
    assert build_owner_filter(end_user) == {
        "OR": [
            {"created_by": "customer-1"},
            {"team_id": "team-eng"},
        ]
    }


# ---------------------------------------------------------------------------
# get_managed_resource_owner_id
# ---------------------------------------------------------------------------


def test_owner_id_prefers_user_id():
    auth = UserAPIKeyAuth(user_id="alice", end_user_id="customer-1")
    assert get_managed_resource_owner_id(auth) == "alice"


def test_owner_id_falls_back_to_end_user_id():
    auth = UserAPIKeyAuth(end_user_id="customer-1")
    assert get_managed_resource_owner_id(auth) == "customer-1"


def test_owner_id_none_when_no_identity():
    assert get_managed_resource_owner_id(UserAPIKeyAuth()) is None


# ---------------------------------------------------------------------------
# can_access_resource
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
)
@pytest.mark.parametrize(
    "created_by,resource_team_id",
    [("alice", "team-eng"), (None, None)],
)
def test_access_admin_can_read_any_resource(role, created_by, resource_team_id):
    admin = UserAPIKeyAuth(user_role=role)
    assert (
        can_access_resource(
            admin, created_by=created_by, resource_team_id=resource_team_id
        )
        is True
    )


@pytest.mark.parametrize(
    "user_id,created_by,expected",
    [
        ("alice", "alice", True),
        ("alice", "bob", False),
        ("alice", None, False),
    ],
)
def test_access_user_id_match(user_id, created_by, expected):
    user = UserAPIKeyAuth(user_id=user_id)
    assert (
        can_access_resource(user, created_by=created_by, resource_team_id=None)
        is expected
    )


@pytest.mark.parametrize(
    "end_user_id,created_by,expected",
    [
        ("customer-1", "customer-1", True),
        ("customer-1", "customer-2", False),
        ("customer-1", None, False),
    ],
)
def test_access_end_user_id_match_when_no_user_id(end_user_id, created_by, expected):
    """Custom-auth callers (only end_user_id) can read resources they
    created (created_by == end_user_id), but not others' or ownerless rows."""
    caller = UserAPIKeyAuth(end_user_id=end_user_id)
    assert (
        can_access_resource(caller, created_by=created_by, resource_team_id=None)
        is expected
    )


def test_access_user_id_match_ignores_end_user_id():
    """When user_id is set it is the owner identity; a created_by that only
    matches end_user_id must NOT grant access."""
    caller = UserAPIKeyAuth(user_id="alice", end_user_id="customer-1")
    assert (
        can_access_resource(caller, created_by="customer-1", resource_team_id=None)
        is False
    )


@pytest.mark.parametrize(
    "caller_team_id,resource_team_id,expected",
    [
        ("team-eng", "team-eng", True),
        ("team-eng", "team-sales", False),
        ("team-eng", None, False),
    ],
)
def test_access_service_account_team_id_match(
    caller_team_id, resource_team_id, expected
):
    service_account = UserAPIKeyAuth(team_id=caller_team_id)
    assert (
        can_access_resource(
            service_account, created_by=None, resource_team_id=resource_team_id
        )
        is expected
    )


def test_access_user_can_see_team_match_when_no_user_id_match():
    """Falls through to the team check when user_id doesn't match — lets a
    team member read a resource created by a sibling service account in the
    same team."""
    user = UserAPIKeyAuth(user_id="alice", team_id="team-eng")
    assert (
        can_access_resource(user, created_by="service-bot", resource_team_id="team-eng")
        is True
    )


def test_access_service_account_denied_user_resource_in_different_team():
    service_account = UserAPIKeyAuth(team_id="team-eng")
    assert (
        can_access_resource(
            service_account, created_by="bob", resource_team_id="team-sales"
        )
        is False
    )


@pytest.mark.parametrize(
    "created_by,resource_team_id",
    [
        (None, None),
        ("anybody", None),
        (None, "any-team"),
        ("anybody", "any-team"),
    ],
)
def test_access_identity_less_caller_always_denied(created_by, resource_team_id):
    """The original `None == None` bypass — a caller with no admin role and
    no identifying ids is denied against every resource regardless of how
    the resource was tagged."""
    nobody = UserAPIKeyAuth()
    assert (
        can_access_resource(
            nobody, created_by=created_by, resource_team_id=resource_team_id
        )
        is False
    )
