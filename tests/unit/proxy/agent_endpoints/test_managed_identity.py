from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.agent_endpoints.managed_identity import classify_agent_subject, managed_write_fields
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import (
    AgentExecutionMode,
    AgentIdentityBinding,
    AgentIdentityFailure,
    AgentSubject,
    EntraIdentityConfig,
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


def managed_agent() -> AgentResponse:
    return AgentResponse(
        agent_id="agent-one", agent_name="Research", agent_card_params={}, identity=BINDING, identity_managed=True
    )


def test_unbinding_keeps_managed_state_and_disables_agent() -> None:
    result: Final = managed_write_fields({"identity": None, "enabled": True}, managed_agent(), "admin")
    assert not isinstance(result, AgentIdentityFailure)
    assert result["identity_managed"] is True
    assert result["enabled"] is False
    assert result["identity"]["update"]["active"] is False
    assert result["identity"]["update"]["last_authenticated_at"] is None
    assert result["identity"]["update"]["revision"] != BINDING.revision


def test_rename_does_not_rewrite_binding_or_evidence() -> None:
    assert managed_write_fields({"agent_name": "Renamed"}, managed_agent(), "admin") == {}


def test_autonomous_binding_requires_enterprise_application_object_id() -> None:
    result: Final = managed_write_fields(
        {"identity": {"provider": "microsoft_entra", "tenant_id": TENANT, "client_id": CLIENT}}, None, "admin"
    )
    assert isinstance(result, AgentIdentityFailure)
    assert "service-principal" in result.message


def test_rebinding_clears_evidence_and_uses_atomic_nested_write() -> None:
    result: Final = managed_write_fields(
        {
            "identity": {
                "provider": "microsoft_entra",
                "tenant_id": TENANT,
                "client_id": CLIENT,
                "service_principal_id": PRINCIPAL,
            }
        },
        managed_agent(),
        "admin",
    )
    assert not isinstance(result, AgentIdentityFailure)
    assert result["identity_managed"] is True
    assert "upsert" in result["identity"]
    assert result["identity"]["upsert"]["update"]["revision"] != BINDING.revision
    assert result["identity"]["upsert"]["update"]["last_authenticated_at"] is None


def test_unbound_identity_can_be_reactivated_with_the_same_application() -> None:
    disabled: Final = managed_agent().model_copy(
        update={"identity": BINDING.model_copy(update={"active": False}), "enabled": False}
    )
    configuration: Final = BINDING.model_dump(
        exclude={"agent_id", "issuer", "revision", "last_authenticated_at", "active"}
    )
    result: Final = managed_write_fields({"identity": configuration, "enabled": True}, disabled, "admin")
    assert not isinstance(result, AgentIdentityFailure)
    assert result["enabled"] is True
    assert result["identity"]["upsert"]["update"]["active"] is True
    assert result["identity"]["upsert"]["update"]["revision"] != BINDING.revision


def test_each_application_binding_records_its_history_atomically() -> None:
    configuration: Final = BINDING.model_dump(
        exclude={"agent_id", "issuer", "revision", "last_authenticated_at", "active"}
    )
    created: Final = managed_write_fields({"identity": configuration}, None, "admin")
    assert not isinstance(created, AgentIdentityFailure)
    assert created["retired_identities"]["create"]["client_id"] == CLIENT
    replacement: Final = managed_write_fields(
        {"identity": {**configuration, "client_id": HUMAN}}, managed_agent(), "admin"
    )
    assert not isinstance(replacement, AgentIdentityFailure)
    assert replacement["retired_identities"]["create"]["client_id"] == HUMAN


def test_unchanged_binding_preserves_revision_and_authentication_evidence() -> None:
    configuration: Final = BINDING.model_dump(
        exclude={"agent_id", "issuer", "revision", "last_authenticated_at", "active"}
    )
    assert managed_write_fields({"identity": configuration}, managed_agent(), "admin") == {}


@pytest.mark.parametrize("identity", [None, BINDING.model_copy(update={"active": False})])
def test_enabling_unbound_or_inactive_identity_requires_rebinding(identity: AgentIdentityBinding | None) -> None:
    agent: Final = managed_agent().model_copy(update={"identity": identity, "enabled": False})
    result: Final = managed_write_fields({"enabled": True}, agent, "admin")
    assert isinstance(result, AgentIdentityFailure)
    assert "Bind an identity" in result.message


@pytest.mark.parametrize("mode", ["delegated", "both"])
def test_explicit_empty_scope_requirements_can_be_registered_and_preserved(mode: str) -> None:
    from litellm.types.proxy.agent_identity import EntraIdentityConfig

    configuration: Final = EntraIdentityConfig(
        provider="microsoft_entra",
        tenant_id=TENANT,
        client_id=CLIENT,
        service_principal_id=PRINCIPAL,
        required_scopes=(),
    )
    created: Final = managed_write_fields(
        {"identity": configuration.model_dump(), "execution_mode": mode}, None, "admin"
    )
    assert not isinstance(created, AgentIdentityFailure)
    assert created["identity"]["create"]["required_scopes"] == ()
    agent: Final = managed_agent().model_copy(update={"identity": BINDING.model_copy(update={"required_scopes": ()})})
    updated: Final = managed_write_fields({"execution_mode": mode}, agent, "admin")
    assert not isinstance(updated, AgentIdentityFailure)
    assert updated["execution_mode"] == mode


@pytest.mark.parametrize(
    "incoming",
    [
        {"identity": {"provider": "microsoft_entra", "tenant_id": "invalid", "client_id": CLIENT}},
        {"execution_mode": "unknown"},
    ],
)
def test_invalid_identity_configuration_returns_a_public_validation_failure(incoming: dict[str, object]) -> None:
    result: Final = managed_write_fields(incoming, None, "admin")
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "identity_denied"
    assert result.message.startswith("Invalid agent identity configuration:")


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


BLUEPRINT: Final = "55555555-5555-4555-8555-555555555555"
BLUEPRINT_BINDING: Final = BINDING.model_copy(update={"blueprint_id": BLUEPRINT})


def blueprint_claims(**overrides: object) -> dict[str, object]:
    return claims(**{"xms_par_app_azp": BLUEPRINT, "xms_act_fct": "3 9 11", **overrides})


def test_blueprint_binding_accepts_matching_agent_identity_token() -> None:
    result: Final = classify_agent_subject(BLUEPRINT_BINDING, blueprint_claims(), "autonomous")
    assert result == AgentSubject(kind="application", oid=PRINCIPAL, mode="autonomous")


def test_blueprint_binding_accepts_uppercase_blueprint_claim() -> None:
    result: Final = classify_agent_subject(
        BLUEPRINT_BINDING, blueprint_claims(xms_par_app_azp=BLUEPRINT.upper()), "autonomous"
    )
    assert result == AgentSubject(kind="application", oid=PRINCIPAL, mode="autonomous")


def test_blueprint_binding_accepts_matching_delegated_token() -> None:
    result: Final = classify_agent_subject(
        BLUEPRINT_BINDING,
        blueprint_claims(oid=HUMAN, scp="user_impersonation"),
        "delegated",
    )
    assert result == AgentSubject(kind="delegated_subject", oid=HUMAN, mode="delegated")


@pytest.mark.parametrize(
    "overrides",
    [
        {"xms_par_app_azp": None},
        {"xms_par_app_azp": "66666666-6666-4666-8666-666666666666"},
        {"xms_par_app_azp": 11},
        {"xms_act_fct": None},
        {"xms_act_fct": "3 9"},
        {"xms_act_fct": "111"},
        {"xms_act_fct": [11]},
    ],
)
@pytest.mark.parametrize("shape", ["autonomous", "delegated"])
def test_blueprint_binding_rejects_tokens_outside_the_pinned_blueprint(
    overrides: dict[str, object], shape: str
) -> None:
    extra: Final = {} if shape == "autonomous" else {"oid": HUMAN, "scp": "user_impersonation"}
    blue_claims: Final = blueprint_claims(**extra)
    for key, value in overrides.items():
        if value is None:
            blue_claims.pop(key)
        else:
            blue_claims[key] = value
    result: Final = classify_agent_subject(BLUEPRINT_BINDING, blue_claims, "both")
    assert isinstance(result, AgentIdentityFailure)
    assert result.message == "Token was not issued to an agent identity of the configured blueprint"


def test_unbound_blueprint_binding_accepts_foreign_parent_claim() -> None:
    result: Final = classify_agent_subject(
        BINDING,
        claims(xms_par_app_azp="66666666-6666-4666-8666-666666666666"),
        "autonomous",
    )
    assert result == AgentSubject(kind="application", oid=PRINCIPAL, mode="autonomous")


def test_new_binding_carries_normalized_blueprint_id() -> None:
    created: Final = managed_write_fields(
        {
            "identity": {
                "provider": "microsoft_entra",
                "tenant_id": TENANT,
                "client_id": CLIENT,
                "service_principal_id": PRINCIPAL,
                "blueprint_id": BLUEPRINT.upper(),
            }
        },
        None,
        "admin",
    )
    assert not isinstance(created, AgentIdentityFailure)
    assert created.get("identity", {}).get("create", {}).get("blueprint_id") == BLUEPRINT
    replacement: Final = managed_write_fields(
        {
            "identity": {
                "provider": "microsoft_entra",
                "tenant_id": TENANT,
                "client_id": CLIENT,
                "service_principal_id": PRINCIPAL,
                "blueprint_id": BLUEPRINT.upper(),
            }
        },
        managed_agent(),
        "admin",
    )
    assert not isinstance(replacement, AgentIdentityFailure)
    assert replacement.get("identity", {}).get("upsert", {}).get("update", {}).get("blueprint_id") == BLUEPRINT


def test_blueprint_only_change_rebinds_with_new_revision_and_cleared_evidence() -> None:
    agent: Final = managed_agent().model_copy(
        update={
            "identity": BLUEPRINT_BINDING.model_copy(
                update={"last_authenticated_at": datetime(2026, 9, 30, tzinfo=timezone.utc)}
            )
        }
    )
    result: Final = managed_write_fields(
        {
            "identity": {
                "provider": "microsoft_entra",
                "tenant_id": TENANT,
                "client_id": CLIENT,
                "service_principal_id": PRINCIPAL,
                "required_roles": ["Agent.Invoke"],
                "required_scopes": ["user_impersonation"],
                "blueprint_id": "66666666-6666-4666-8666-666666666666",
            }
        },
        agent,
        "admin",
    )
    assert not isinstance(result, AgentIdentityFailure)
    update: Final = result.get("identity", {}).get("upsert", {}).get("update", {})
    assert update
    assert update.get("revision") != BLUEPRINT_BINDING.revision
    assert update.get("last_authenticated_at") is None


def test_entra_config_normalizes_and_validates_blueprint_id() -> None:
    config: Final = EntraIdentityConfig(
        provider="microsoft_entra",
        tenant_id=TENANT,
        client_id=CLIENT,
        blueprint_id=BLUEPRINT.upper(),
    )
    assert config.blueprint_id == BLUEPRINT
    with pytest.raises(ValidationError):
        EntraIdentityConfig(
            provider="microsoft_entra",
            tenant_id=TENANT,
            client_id=CLIENT,
            blueprint_id="not-a-uuid",
        )
