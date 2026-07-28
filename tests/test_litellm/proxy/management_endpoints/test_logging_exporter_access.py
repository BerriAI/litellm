"""The request-time routing predicate for admin-owned logging destinations.

``access_grants`` is the chokepoint the resolver routes through at call time, so a
mutation here would route an identity's traces to a destination outside its scope.
Each case is written to fail if the corresponding branch is flipped.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.models.credentials import CredentialAccess, CredentialInfo
from litellm.proxy.management_endpoints.logging_exporter_access import (
    access_grants,
    identity_scope,
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


# --- routing scope decided entirely by access -------------------------------
#
# Routing is access-only. auto_enable does not widen it: an empty-access
# destination fires for no one regardless of auto_enable (empty access =
# deny-all). Proxy-wide routing must be requested explicitly with
# access.global=True.


def test_empty_access_is_deny_all_even_with_auto_enable():
    """Empty access grants no one, even when auto_enable=True: not proxy-wide."""
    info = CredentialInfo(credential_type="logging", auto_enable=True)
    assert access_grants(info.access, frozenset(), frozenset()) is False
    assert access_grants(info.access, frozenset({"any-team"}), frozenset()) is False
    assert access_grants(info.access, frozenset(), frozenset({"any-org"})) is False


def test_global_access_is_proxy_wide():
    """access.global=True is proxy-wide regardless of auto_enable."""
    info = CredentialInfo(credential_type="logging", auto_enable=True, access=_access(global_=True))
    assert access_grants(info.access, frozenset({"t1"}), frozenset()) is True
    assert access_grants(info.access, frozenset(), frozenset()) is True
    manual = CredentialInfo(credential_type="logging", access=_access(global_=True))
    assert access_grants(manual.access, frozenset(), frozenset()) is True


def test_auto_enable_team_scoped():
    """auto_enable=True + access.teams=[t1] fires only for t1 identities."""
    info = CredentialInfo(credential_type="logging", auto_enable=True, access=_access(teams=["t1"]))
    assert access_grants(info.access, frozenset({"t1"}), frozenset()) is True
    assert access_grants(info.access, frozenset({"t2"}), frozenset()) is False
    assert access_grants(info.access, frozenset(), frozenset()) is False


def test_auto_enable_org_scoped():
    """auto_enable=True + access.orgs=[o1] fires only for o1 identities."""
    info = CredentialInfo(credential_type="logging", auto_enable=True, access=_access(orgs=["o1"]))
    assert access_grants(info.access, frozenset(), frozenset({"o1"})) is True
    assert access_grants(info.access, frozenset(), frozenset({"o2"})) is False


def test_access_scoped_when_not_auto_enable():
    info = CredentialInfo(credential_type="logging", access=_access(teams=["t1"]))
    assert access_grants(info.access, frozenset({"t1"}), frozenset()) is True
    assert access_grants(info.access, frozenset({"t2"}), frozenset()) is False


def test_denies_when_no_access():
    info = CredentialInfo(credential_type="logging")
    assert access_grants(info.access, frozenset({"t1"}), frozenset({"o1"})) is False


# --- identity_scope --------------------------------------------------------


def test_identity_scope_single_elements():
    teams, orgs = identity_scope("t1", "o1")
    assert teams == frozenset({"t1"})
    assert orgs == frozenset({"o1"})


def test_identity_scope_empty_for_none():
    teams, orgs = identity_scope(None, None)
    assert teams == frozenset()
    assert orgs == frozenset()
