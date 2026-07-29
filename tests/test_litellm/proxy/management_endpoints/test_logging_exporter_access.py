"""The request-time routing predicate for admin-owned logging destinations.

``access_grants`` is the chokepoint the resolver routes through at call time, so a
mutation here would route an identity's traces to a destination outside its scope.
Each case is written to fail if the corresponding branch is flipped.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.models.credentials import CredentialAccess, CredentialInfo, CredentialItem
from litellm.proxy.management_endpoints.logging_exporter_access import (
    access_grants,
    identity_scope,
    parse_credential_info,
    resolved_logging_exporter_names,
)


# --- parse_credential_info: fail closed on bad input -----------------------


def test_parse_none_for_non_dict():
    assert parse_credential_info(None) is None
    assert parse_credential_info("not a dict") is None
    assert parse_credential_info(["a"]) is None


def test_parse_typed_access():
    info = parse_credential_info(
        {
            "credential_type": "logging",
            "description": "arize",
            "access": {"global": True, "teams": ["t1"], "orgs": ["o1"]},
        }
    )
    assert info is not None
    assert info.credential_type == "logging"
    assert info.access is not None
    assert info.access.global_ is True
    assert info.access.teams == ("t1",)
    assert info.access.orgs == ("o1",)


def test_parse_missing_access_is_none_not_error():
    info = parse_credential_info({"credential_type": "logging"})
    assert info is not None
    assert info.access is None


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
def test_empty_access_is_deny_all():
    """Empty access grants no one: not proxy-wide."""
    info = CredentialInfo(credential_type="logging")
    assert access_grants(info.access, frozenset(), frozenset()) is False
    assert access_grants(info.access, frozenset({"any-team"}), frozenset()) is False
    assert access_grants(info.access, frozenset(), frozenset({"any-org"})) is False


def test_global_access_is_proxy_wide():
    """access.global=True reaches every identity."""
    info = CredentialInfo(credential_type="logging", access=_access(global_=True))
    assert access_grants(info.access, frozenset({"t1"}), frozenset()) is True
    assert access_grants(info.access, frozenset(), frozenset()) is True


def test_access_team_scoped():
    """access.teams=[t1] fires only for t1 identities."""
    info = CredentialInfo(credential_type="logging", access=_access(teams=["t1"]))
    assert access_grants(info.access, frozenset({"t1"}), frozenset()) is True
    assert access_grants(info.access, frozenset({"t2"}), frozenset()) is False
    assert access_grants(info.access, frozenset(), frozenset()) is False


def test_access_org_scoped():
    """access.orgs=[o1] fires only for o1 identities."""
    info = CredentialInfo(credential_type="logging", access=_access(orgs=["o1"]))
    assert access_grants(info.access, frozenset(), frozenset({"o1"})) is True
    assert access_grants(info.access, frozenset(), frozenset({"o2"})) is False


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


# --- resolved_logging_exporter_names: the /team/info + /organization/info disclosure --


def _cred(name, access=None, ctype="logging"):
    info = {"credential_type": ctype}
    if access is not None:
        info["access"] = access
    return CredentialItem(credential_name=name, credential_values={}, credential_info=info)


def test_resolved_names_are_access_only(monkeypatch):
    """A destination name appears iff its access grants the (team_id, org_id).
    Included: team-granted, org-granted, global. Excluded: empty-access,
    granted-but-not-logging (provider) credentials, access for another team."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            _cred("team-granted", access={"teams": ["t1"]}),
            _cred("team-other", access={"teams": ["other"]}),
            _cred("org-granted", access={"orgs": ["o1"]}),
            _cred("empty-access"),
            _cred("global-access", access={"global": True}),
            _cred("provider", access={"global": True}, ctype=None),
        ],
    )
    names = resolved_logging_exporter_names("t1", "o1")
    assert names == ("team-granted", "org-granted", "global-access")


def test_resolved_names_empty_scope_gets_global_only(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            _cred("global-access", access={"global": True}),
            _cred("team-scoped", access={"teams": ["t1"]}),
        ],
    )
    assert resolved_logging_exporter_names(None, None) == ("global-access",)


def test_resolved_names_empty_registry(monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [])
    assert resolved_logging_exporter_names("t1", "o1") == ()
