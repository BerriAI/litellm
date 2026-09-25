from typing import Final

import pytest

from litellm.proxy.agent_endpoints.managed_identity import classify_agent_subject, managed_write_fields
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityBinding, AgentIdentityFailure, AgentSubject

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
    assert created["retired_identities"]["connectOrCreate"]["create"]["client_id"] == CLIENT
    replacement: Final = managed_write_fields(
        {"identity": {**configuration, "client_id": HUMAN}}, managed_agent(), "admin"
    )
    assert not isinstance(replacement, AgentIdentityFailure)
    assert replacement["retired_identities"]["connectOrCreate"]["create"]["client_id"] == HUMAN


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


def test_delegated_identity_requires_a_scope() -> None:
    agent: Final = managed_agent().model_copy(update={"identity": BINDING.model_copy(update={"required_scopes": ()})})
    result: Final = managed_write_fields({"execution_mode": "delegated"}, agent, "admin")
    assert isinstance(result, AgentIdentityFailure)
    assert "delegated scope" in result.message


@pytest.mark.parametrize("budget", [{"max_budget": -1}, {"max_budget": float("inf")}])
def test_invalid_budget_changes_are_rejected(budget: dict[str, object]) -> None:
    result: Final = managed_write_fields({"budget": budget}, managed_agent(), "admin")
    assert isinstance(result, AgentIdentityFailure)
    assert "Invalid agent identity or budget configuration" in result.message


def test_budget_updates_preserve_current_window_until_duration_changes() -> None:
    from datetime import datetime, timezone

    from litellm.types.proxy.agent_identity import AgentBudgetState

    reset: Final = datetime(2027, 1, 1, tzinfo=timezone.utc)
    agent: Final = managed_agent().model_copy(
        update={
            "budget_id": "budget",
            "litellm_budget_table": AgentBudgetState(
                budget_id="budget", max_budget=1, budget_duration="1d", budget_reset_at=reset
            ),
        }
    )
    same: Final = managed_write_fields({"budget": {"max_budget": 2, "budget_duration": "1d"}}, agent, "admin")
    assert not isinstance(same, AgentIdentityFailure)
    assert same["litellm_budget_table"]["update"]["budget_reset_at"] == reset
    assert same["litellm_budget_table"]["update"]["max_budget"] == 2
    changed: Final = managed_write_fields({"budget": {"max_budget": 2, "budget_duration": "1h"}}, agent, "admin")
    assert not isinstance(changed, AgentIdentityFailure)
    assert changed["litellm_budget_table"]["update"]["budget_reset_at"] != reset
    removed: Final = managed_write_fields({"budget": None}, agent, "admin")
    assert removed == {"litellm_budget_table": {"disconnect": True}}
    assert managed_write_fields({"budget": None}, managed_agent(), "admin") == {}


def test_new_agent_budget_is_created_with_administrator_attribution() -> None:
    result: Final = managed_write_fields({"budget": {"max_budget": 0}}, None, "admin")
    assert not isinstance(result, AgentIdentityFailure)
    assert result["litellm_budget_table"]["create"] == {
        "max_budget": 0,
        "budget_duration": None,
        "budget_reset_at": None,
        "created_by": "admin",
        "updated_by": "admin",
    }


@pytest.mark.parametrize("roles", ["Agent.Invoke", [42], None])
def test_malformed_application_roles_are_rejected(roles: object) -> None:
    result: Final = classify_agent_subject(BINDING, claims(roles=roles), "autonomous")
    assert isinstance(result, AgentIdentityFailure)
    assert "Invalid application roles" in result.message
