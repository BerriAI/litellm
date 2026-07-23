"""The shared visibility predicate for admin-owned logging destinations.

This is the single chokepoint the list endpoint, the assignment validator, and
the request-time resolver all route through, so a mutation here would let a
non-admin see or route to a destination outside their scope. Each case is written
to fail if the corresponding branch is flipped.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.models.credentials import CredentialAccess, CredentialInfo
from litellm.proxy.management_endpoints.logging_exporter_access import (
    access_grants,
    identity_scope,
    is_destination_visible,
    parse_credential_info,
)


# --- parse_credential_info: fail closed on bad input -----------------------


def test_parse_none_for_non_dict():
    assert parse_credential_info(None) is None
    assert parse_credential_info("not a dict") is None
    assert parse_credential_info(["a"]) is None


def test_parse_typed_access_and_auto_enable():
    info = parse_credential_info(
        {
            "credential_type": "logging",
            "description": "arize",
            "auto_enable": True,
            "access": {"global": True, "teams": ["t1"], "orgs": ["o1"]},
        }
    )
    assert info is not None
    assert info.credential_type == "logging"
    assert info.auto_enable is True
    assert info.access is not None
    assert info.access.global_ is True
    assert info.access.teams == ("t1",)
    assert info.access.orgs == ("o1",)


def test_parse_missing_access_is_none_not_error():
    info = parse_credential_info({"credential_type": "logging"})
    assert info is not None
    assert info.access is None
    assert info.auto_enable is False


def test_parse_malformed_access_fails_closed():
    """A stored access with an unknown field is rejected by the strict read model;
    the parse must return None (invisible) rather than raise or grant."""
    assert parse_credential_info({"access": {"legacy_field": "x"}}) is None
    assert parse_credential_info({"access": "not-an-object"}) is None


# --- access_grants: the primitive ------------------------------------------


def _access(**kw) -> CredentialAccess:
    return CredentialAccess.model_validate(kw)


def test_access_grants_global_reaches_empty_scope():
    assert access_grants(_access(**{"global": True}), frozenset(), frozenset()) is True


def test_access_grants_none_denies():
    assert access_grants(None, frozenset({"t1"}), frozenset({"o1"})) is False


def test_access_grants_team_match():
    a = _access(teams=["t1", "t2"])
    assert access_grants(a, frozenset({"t2"}), frozenset()) is True
    assert access_grants(a, frozenset({"t3"}), frozenset()) is False


def test_access_grants_org_match():
    a = _access(orgs=["o1"])
    assert access_grants(a, frozenset(), frozenset({"o1"})) is True
    assert access_grants(a, frozenset(), frozenset({"o2"})) is False


def test_access_grants_disjoint_denies():
    a = _access(teams=["t1"], orgs=["o1"])
    assert access_grants(a, frozenset({"t9"}), frozenset({"o9"})) is False


def test_access_grants_not_global_when_false():
    """global=False must not short-circuit to visible."""
    a = _access(**{"global": False})
    assert access_grants(a, frozenset({"t1"}), frozenset({"o1"})) is False


# --- is_destination_visible: auto_enable scoped by access ------------------
#
# auto_enable=True means "selected automatically" but the scope of that
# automatic selection is controlled by access:
#   - explicit grants (global/teams/orgs) → caller must be within the grant
#   - empty access (no grants at all)     → proxy-wide fallback (visible to all)
# This lets admins create a team-scoped auto-exporter without it leaking to
# every other team on the proxy.


def test_visible_auto_enable_empty_access_is_proxy_wide():
    """auto_enable=True with no access grants is proxy-wide: visible to all admins."""
    info = CredentialInfo(credential_type="logging", auto_enable=True)
    assert is_destination_visible(info, frozenset(), frozenset()) is True
    assert is_destination_visible(info, frozenset({"any-team"}), frozenset()) is True
    assert is_destination_visible(info, frozenset(), frozenset({"any-org"})) is True


def test_visible_auto_enable_global_access_is_proxy_wide():
    """auto_enable=True + access.global=True is proxy-wide."""
    info = CredentialInfo(credential_type="logging", auto_enable=True, access=_access(global_=True))
    assert is_destination_visible(info, frozenset({"t1"}), frozenset()) is True
    assert is_destination_visible(info, frozenset(), frozenset()) is True


def test_visible_auto_enable_team_scoped():
    """auto_enable=True + access.teams=[t1] is visible only to t1 admins."""
    info = CredentialInfo(credential_type="logging", auto_enable=True, access=_access(teams=["t1"]))
    assert is_destination_visible(info, frozenset({"t1"}), frozenset()) is True
    assert is_destination_visible(info, frozenset({"t2"}), frozenset()) is False
    assert is_destination_visible(info, frozenset(), frozenset()) is False


def test_visible_auto_enable_org_scoped():
    """auto_enable=True + access.orgs=[o1] is visible only to o1 admins."""
    info = CredentialInfo(credential_type="logging", auto_enable=True, access=_access(orgs=["o1"]))
    assert is_destination_visible(info, frozenset(), frozenset({"o1"})) is True
    assert is_destination_visible(info, frozenset(), frozenset({"o2"})) is False


def test_visible_delegates_to_access_when_not_auto_enable():
    info = CredentialInfo(credential_type="logging", access=_access(teams=["t1"]))
    assert is_destination_visible(info, frozenset({"t1"}), frozenset()) is True
    assert is_destination_visible(info, frozenset({"t2"}), frozenset()) is False


def test_visible_denies_when_neither():
    info = CredentialInfo(credential_type="logging")
    assert is_destination_visible(info, frozenset({"t1"}), frozenset({"o1"})) is False


# --- identity_scope --------------------------------------------------------


def test_identity_scope_single_elements():
    teams, orgs = identity_scope("t1", "o1")
    assert teams == frozenset({"t1"})
    assert orgs == frozenset({"o1"})


def test_identity_scope_empty_for_none():
    teams, orgs = identity_scope(None, None)
    assert teams == frozenset()
    assert orgs == frozenset()
