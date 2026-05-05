"""
Tests for managed-resource tenant isolation helpers.
"""

import pytest

from litellm.llms.base_llm.managed_resources.isolation import (
    build_owner_filter,
    can_access_resource,
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
