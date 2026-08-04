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
    destination_for_credential,
    identity_scope,
    is_logging_credential,
    parse_credential_info,
    resolved_logging_exporter_names,
)

import pytest

from litellm.integrations.otel.model.config import is_otel_v2_enabled


@pytest.fixture(autouse=True)
def _reset_otel_v2_flag_cache():
    """``is_otel_v2_enabled`` is lru-cached; clear it around each test so ``LITELLM_OTEL_V2``
    toggles take effect and don't leak between tests."""
    is_otel_v2_enabled.cache_clear()
    yield
    is_otel_v2_enabled.cache_clear()


# --- is_logging_credential: the access-validation + merge gate ---------------


def test_is_logging_credential_true_for_logging_type():
    assert is_logging_credential({"credential_type": "logging", "description": "arize"}) is True


def test_is_logging_credential_true_even_with_malformed_access():
    """A destination carrying an invalid access shape is still a logging destination.
    The gate must route it into validate_credential_access (which rejects it), not skip
    validation because the strict access model can't parse it."""
    assert is_logging_credential({"credential_type": "logging", "access": {"nonsense_field": True}}) is True


def test_is_logging_credential_false_for_provider_and_malformed():
    assert is_logging_credential({"custom_llm_provider": "openai"}) is False
    assert is_logging_credential({"custom_llm_provider": "openai", "access": {"global": True}}) is False
    assert is_logging_credential(None) is False
    assert is_logging_credential("nope") is False


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


def _cred(name, access=None, ctype="logging", buildable=True, endpoint=None):
    info = {"credential_type": ctype}
    if access is not None:
        info["access"] = access
    values = {}
    if buildable:
        # a generic OTLP backend with an endpoint builds a destination, so disclosure
        # (which now mirrors the resolver's buildability) includes it. The endpoint is
        # per-name by default so these fixtures are distinct destinations; disclosure
        # dedupes ones that resolve to the same target, which is covered separately.
        info["description"] = "generic"
        values = {"otel_endpoint": endpoint or f"http://collector.example/{name}/v1/traces"}
    return CredentialItem(credential_name=name, credential_values=values, credential_info=info)


def test_resolved_names_are_access_only(monkeypatch):
    """A destination name appears iff its access grants the (team_id, org_id).
    Included: team-granted, org-granted, global. Excluded: empty-access,
    granted-but-not-logging (provider) credentials, access for another team."""
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
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
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
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
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    monkeypatch.setattr(litellm, "credential_list", [])
    assert resolved_logging_exporter_names("t1", "o1") == ()


def test_resolved_names_gated_off_when_v2_disabled(monkeypatch):
    """Disclosure mirrors the resolver, which returns nothing with the v2 flag off.
    A granting destination is disclosed only when ``LITELLM_OTEL_V2`` is enabled, so
    ``/team/info`` never claims traces are exported while the feature is inert."""
    monkeypatch.setattr(litellm, "credential_list", [_cred("team-granted", access={"teams": ["t1"]})])

    monkeypatch.setenv("LITELLM_OTEL_V2", "false")
    is_otel_v2_enabled.cache_clear()
    assert resolved_logging_exporter_names("t1", None) == ()

    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    assert resolved_logging_exporter_names("t1", None) == ("team-granted",)


def test_resolved_names_excludes_unbuildable(monkeypatch):
    """Disclosure mirrors the resolver's buildability, not access alone: a granted
    destination that names no backend, or a backend with incomplete values, resolves to
    nothing at request time and must not be advertised on /team/info."""
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            _cred("buildable-generic", access={"teams": ["t1"]}),
            _cred("no-backend", access={"teams": ["t1"]}, buildable=False),
            CredentialItem(
                credential_name="langfuse-missing-secret",
                credential_values={"langfuse_public_key": "pk-only"},
                credential_info={
                    "credential_type": "logging",
                    "access": {"teams": ["t1"]},
                    "description": "langfuse_otel",
                },
            ),
        ],
    )
    assert resolved_logging_exporter_names("t1", None) == ("buildable-generic",)


def test_backend_without_a_preset_routes_under_generic():
    """Regression: a backend outside ``PRESET_BY_CALLBACK`` is routed as ``generic``.

    Only a registered name gets an ``OpenTelemetryV2`` logger, and that logger is what
    emits the gen-AI span to its destinations. Routing an unregistered name under itself
    delivered the proxy-internal spans but never the LLM call. Registered names keep
    their own routing so each backend's attribute vocabulary is preserved."""
    from litellm.integrations.otel.presets import PRESET_BY_CALLBACK

    unknown = CredentialItem(
        credential_name="self-hosted",
        credential_values={"otel_endpoint": "http://collector.example/v1/traces"},
        credential_info={"credential_type": "logging", "description": "honeycomb", "access": {"global": True}},
    )
    resolved = destination_for_credential(unknown)
    assert resolved is not None
    assert resolved[0] == "generic"
    assert resolved[1].endpoint == "http://collector.example/v1/traces"

    for registered in ("arize", "langfuse_otel", "generic"):
        assert registered in PRESET_BY_CALLBACK
    known = CredentialItem(
        credential_name="lf",
        credential_values={"langfuse_public_key": "pk", "langfuse_secret_key": "sk"},
        credential_info={"credential_type": "logging", "description": "langfuse_otel", "access": {"global": True}},
    )
    known_resolved = destination_for_credential(known)
    assert known_resolved is not None
    assert known_resolved[0] == "langfuse_otel"


def test_resolved_names_keep_every_grant_sharing_one_target(monkeypatch):
    """Regression: two credentials resolving to one export target are both named.

    Disclosure once kept the first credential per target while the resolver's dict
    comprehension kept the last, so /team/info named a credential whose backend the
    request path never activated. Both grant the team, so both are disclosed and there
    is no winner to disagree about; the resolver still collapses the shared target."""
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    same = "http://collector.example/shared/v1/traces"
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            _cred("dup-one", access={"teams": ["t1"]}, endpoint=same),
            _cred("dup-two", access={"teams": ["t1"]}, endpoint=same),
            _cred("distinct", access={"teams": ["t1"]}),
        ],
    )
    assert resolved_logging_exporter_names("t1", None) == ("dup-one", "dup-two", "distinct")


@pytest.mark.asyncio
async def test_disclosure_agrees_with_the_resolver_on_a_shared_target(monkeypatch):
    """Regression: the disclosed names and the resolver's selection are derived from the
    same grants, so no credential is disclosed that the resolver dropped entirely.

    Pins the two sides together: the resolver collapses the duplicate target to a single
    export, and every name it kept a destination for is disclosed."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import _resolve_logging_exporters

    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    same = "http://collector.example/shared/v1/traces"
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            _cred("dup-one", access={"global": True}, endpoint=same),
            _cred("dup-two", access={"global": True}, endpoint=same),
        ],
    )

    destinations, _backends = await _resolve_logging_exporters(UserAPIKeyAuth(api_key="k"))

    assert len(destinations) == 1
    assert resolved_logging_exporter_names(None, None) == ("dup-one", "dup-two")
