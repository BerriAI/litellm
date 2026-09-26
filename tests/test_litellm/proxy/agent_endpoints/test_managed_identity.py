from typing import Final

import pytest

from litellm.proxy.agent_endpoints.managed_identity import classify_agent_subject
from litellm.types.proxy.agent_identity import (
    AgentExecutionMode,
    AgentIdentityBinding,
    AgentIdentityFailure,
    AgentSubject,
)

TENANT: Final = "11111111-1111-4111-8111-111111111111"
CLIENT: Final = "22222222-2222-4222-8222-222222222222"
PRINCIPAL: Final = "33333333-3333-4333-8333-333333333333"
HUMAN: Final = "44444444-4444-4444-8444-444444444444"
ISSUER: Final = f"https://login.microsoftonline.com/{TENANT}/v2.0"
BINDING: Final = AgentIdentityBinding(
    agent_id="agent-one",
    provider="microsoft_entra",
    tenant_id=TENANT,
    client_id=CLIENT,
    service_principal_id=PRINCIPAL,
    issuer=ISSUER,
    required_roles=("Agent.Invoke",),
    required_scopes=("user_impersonation",),
    revision="binding-one",
)


def claims(**overrides: object) -> dict[str, object]:
    return {"iss": ISSUER, "tid": TENANT, "azp": CLIENT, "oid": PRINCIPAL, "roles": ["Agent.Invoke"], **overrides}


def test_autonomous_identity_needs_no_human_and_checks_the_pinned_principal() -> None:
    result: Final = classify_agent_subject(BINDING, claims(), "autonomous")
    assert result == AgentSubject(kind="application", oid=PRINCIPAL, mode="autonomous")
    assert isinstance(classify_agent_subject(BINDING, claims(oid=HUMAN), "autonomous"), AgentIdentityFailure)


@pytest.mark.parametrize(
    "overrides",
    [
        {"iss": "https://untrusted.example"},
        {"tid": CLIENT},
        {"azp": TENANT},
        {"roles": []},
        {"idtyp": "user"},
        {"scp": "user_impersonation"},
        {"scp": 1},
        {"oid": None},
    ],
)
def test_application_rejects_mismatched_or_contradictory_verified_claims(overrides: dict[str, object]) -> None:
    assert isinstance(classify_agent_subject(BINDING, claims(**overrides), "both"), AgentIdentityFailure)


def test_delegated_profile_identifies_a_subject_without_asserting_that_it_is_human() -> None:
    result: Final = classify_agent_subject(BINDING, claims(oid=HUMAN, scp="user_impersonation"), "delegated")
    assert result == AgentSubject(kind="delegated_subject", oid=HUMAN, mode="delegated")


@pytest.mark.parametrize(
    "overrides",
    [
        {"scp": "unrelated"},
        {"scp": ""},
        {"idtyp": "app"},
        {"xms_sub_fct": "2 13 15"},
        {"xms_sub_fct": [13]},
    ],
)
def test_delegated_profile_rejects_unknown_scope_and_known_nonhuman_subjects(overrides: dict[str, object]) -> None:
    assert isinstance(
        classify_agent_subject(BINDING, claims(**{"oid": HUMAN, "scp": "user_impersonation", **overrides}), "both"),
        AgentIdentityFailure,
    )


def test_allowed_mode_cannot_be_selected_by_the_caller() -> None:
    assert isinstance(classify_agent_subject(BINDING, claims(), "delegated"), AgentIdentityFailure)
    assert isinstance(
        classify_agent_subject(BINDING, claims(oid=HUMAN, scp="user_impersonation"), "autonomous"),
        AgentIdentityFailure,
    )


def test_native_facet_absence_does_not_establish_human_identity() -> None:
    result: Final = classify_agent_subject(
        BINDING, claims(oid=HUMAN, scp="user_impersonation", xms_sub_fct="113"), "both"
    )
    assert isinstance(result, AgentSubject)
    assert result.kind == "delegated_subject"


@pytest.mark.parametrize("roles", ["Agent.Invoke", [42], None])
def test_malformed_application_roles_are_rejected(roles: object) -> None:
    result: Final = classify_agent_subject(BINDING, claims(roles=roles), "autonomous")
    assert isinstance(result, AgentIdentityFailure)
    assert "Invalid application roles" in result.message


def test_entra_binding_normalizes_identifiers_and_rejects_invalid_configuration() -> None:
    from pydantic import ValidationError
    from litellm.types.proxy.agent_identity import EntraIdentityConfig

    identifier = "ABCDEF00-1234-4234-9234-123456789ABC"
    config = EntraIdentityConfig(provider="microsoft_entra", tenant_id=identifier, client_id=identifier)
    assert config.tenant_id == identifier.lower()
    assert config.client_id == identifier.lower()
    assert config.service_principal_id is None
    assert config.issuer == f"https://login.microsoftonline.com/{config.tenant_id}/v2.0"
    with pytest.raises(ValidationError):
        EntraIdentityConfig(provider="microsoft_entra", tenant_id="invalid", client_id=identifier)


@pytest.mark.parametrize("mode", ["delegated", "both"])
def test_empty_required_scopes_allow_valid_delegated_scope(mode: AgentExecutionMode) -> None:
    binding: Final = BINDING.model_copy(update={"required_scopes": ()})
    result: Final = classify_agent_subject(binding, claims(oid=HUMAN, scp="custom_scope"), mode)
    assert result == AgentSubject(kind="delegated_subject", oid=HUMAN, mode="delegated")


@pytest.mark.parametrize("scope", [None, "", " \t ", 42])
def test_empty_requirements_do_not_make_a_scope_less_human_token_valid(scope: object) -> None:
    binding: Final = BINDING.model_copy(update={"required_scopes": ()})
    result: Final = classify_agent_subject(binding, claims(oid=HUMAN, scp=scope), "both")
    assert isinstance(result, AgentIdentityFailure)
