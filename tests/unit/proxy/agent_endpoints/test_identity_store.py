from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prisma.models import LiteLLM_VerifiedSubject

from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore, resolve_managed_agent
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.repositories.table_repositories import (
    AgentIdentityRepository,
    AgentsRepository,
    VerifiedSubjectRepository,
)
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import (
    AgentIdentityBinding,
    AgentIdentityFailure,
    ManagedAgentContext,
    MicrosoftInteractiveSubject,
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
    revision="revision-one",
)
CLAIMS: Final = {"iss": ISSUER, "tid": TENANT, "azp": CLIENT, "oid": PRINCIPAL, "roles": ["Agent.Invoke"]}


def stored_agent(**overrides: object) -> AgentResponse:
    return AgentResponse.model_validate(
        {
            "agent_id": "agent-one",
            "agent_name": "Research",
            "agent_card_params": {},
            "identity": BINDING,
            "identity_managed": True,
            "execution_mode": "both",
            **overrides,
        }
    )


def setup_store(
    agent: AgentResponse | None = stored_agent(),
    human: LiteLLM_VerifiedSubject | None = None,
    cache: UserApiKeyCache | None = None,
) -> tuple[AgentIdentityStore, AsyncMock, AsyncMock, AsyncMock]:
    agents: Final = AsyncMock()
    identities: Final = AsyncMock()
    humans: Final = AsyncMock()
    agents.find_unique.return_value = agent
    identities.find_unique.return_value = BINDING
    identities.update_many.return_value = 1
    humans.find_unique.return_value = human
    db: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_agentstable=agents,
            litellm_agentidentity=identities,
            litellm_verifiedsubject=humans,
        )
    )
    return (
        AgentIdentityStore(AgentsRepository(db), AgentIdentityRepository(db), VerifiedSubjectRepository(db), cache=cache),
        agents,
        identities,
        humans,
    )


@pytest.mark.asyncio
async def test_application_authentication_has_no_fabricated_human() -> None:
    store, _, _, humans = setup_store()
    result: Final = await store.resolve_verified_claims(CLAIMS)
    assert isinstance(result, ManagedAgentContext)
    assert result.agent_id == "agent-one"
    assert result.mode == "autonomous"
    assert result.user_id is None
    humans.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_binding_lookup_cache_keeps_policy_reads_authoritative() -> None:
    cache: Final = UserApiKeyCache()
    store, agents, identities, _ = setup_store(cache=cache)
    other: Final = AgentIdentityStore(store.agents, store.identities, store.humans, cache=cache)
    assert isinstance(await store.resolve_verified_claims(CLAIMS), ManagedAgentContext)
    assert isinstance(await other.resolve_verified_claims(CLAIMS), ManagedAgentContext)
    identities.find_unique.assert_awaited_once()
    assert agents.find_unique.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        None,
        stored_agent(enabled=False),
        stored_agent(identity=None),
        stored_agent(identity_managed=False),
        stored_agent(execution_mode="delegated"),
        stored_agent(identity=BINDING.model_copy(update={"active": False})),
        stored_agent(identity=BINDING.model_copy(update={"client_id": HUMAN, "revision": "new-binding"})),
        stored_agent(identity=BINDING.model_copy(update={"required_roles": ("New.Role",), "revision": "new-policy"})),
    ],
)
async def test_lifecycle_is_read_on_every_request_without_cached_allow(changed: AgentResponse | None) -> None:
    store, agents, identities, _ = setup_store(cache=UserApiKeyCache())
    agents.find_unique.side_effect = [stored_agent(), changed]
    assert isinstance(await store.resolve_verified_claims(CLAIMS), ManagedAgentContext)
    denial: Final = await store.resolve_verified_claims(CLAIMS)
    assert isinstance(denial, AgentIdentityFailure)
    assert denial.code == "identity_denied"
    identities.find_unique.assert_awaited_once()
    assert agents.find_unique.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable_table", ["agents", "identities", "humans"])
async def test_identity_store_failure_never_becomes_a_legacy_allow(unavailable_table: str) -> None:
    store, agents, identities, humans = setup_store()
    table: Final = {"agents": agents, "identities": identities, "humans": humans}[unavailable_table]
    table.find_unique.side_effect = RuntimeError("database unavailable")
    result: Final = await store.resolve_verified_claims({**CLAIMS, "oid": HUMAN, "scp": "user_impersonation"})
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "policy_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable_table", ["agents", "humans"])
async def test_cached_binding_cannot_hide_authoritative_storage_failure(unavailable_table: str) -> None:
    store, agents, identities, humans = setup_store(cache=UserApiKeyCache())
    assert isinstance(await store.resolve_verified_claims(CLAIMS), ManagedAgentContext)
    table: Final = {"agents": agents, "humans": humans}[unavailable_table]
    table.find_unique.side_effect = ConnectionError("writer unavailable")
    result: Final = await store.resolve_verified_claims({**CLAIMS, "oid": HUMAN, "scp": "user_impersonation"})
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "policy_unavailable"
    identities.find_unique.assert_awaited_once()


@pytest.mark.asyncio
async def test_unclassified_delegated_subject_cannot_authenticate_as_a_user() -> None:
    store, _, _, _ = setup_store()
    result: Final = await store.resolve_verified_claims(
        {**CLAIMS, "oid": HUMAN, "scp": "user_impersonation", "idtyp": "user"}
    )
    assert isinstance(result, AgentIdentityFailure)
    assert "first sign in" in result.message


@pytest.mark.asyncio
async def test_delegated_subject_uses_canonical_sso_user_not_email_claim() -> None:
    human: Final = LiteLLM_VerifiedSubject(
        kind="human",
        subject_id="subject-one",
        issuer=ISSUER,
        tenant_id=TENANT,
        oid=HUMAN,
        user_id="canonical-user",
        verified_via="sso_interactive",
        verified_at=datetime.now(timezone.utc),
    )
    store, _, identities, humans = setup_store(human=human, cache=UserApiKeyCache())
    result: Final = await store.resolve_verified_claims(
        {
            **CLAIMS,
            "oid": HUMAN,
            "scp": "user_impersonation",
            "email": "untrusted-alias@example.com",
        }
    )
    assert isinstance(result, ManagedAgentContext)
    assert result.mode == "delegated"
    assert result.user_id == "canonical-user"
    humans.find_unique.assert_awaited_once_with(
        where={"issuer_tenant_id_oid": {"issuer": ISSUER, "tenant_id": TENANT, "oid": HUMAN}}
    )
    humans.find_unique.return_value = None
    denied: Final = await store.resolve_verified_claims({**CLAIMS, "oid": HUMAN, "scp": "user_impersonation"})
    assert isinstance(denied, AgentIdentityFailure)
    assert denied.code == "identity_denied"
    identities.find_unique.assert_awaited_once()
    assert humans.find_unique.await_count == 2


@pytest.mark.asyncio
async def test_rebinding_during_authentication_does_not_mark_new_identity_verified() -> None:
    store, _, identities, _ = setup_store()
    identities.update_many.return_value = 0
    context: Final = ManagedAgentContext(agent_id="agent-one", binding_revision="old-revision", mode="autonomous")
    result: Final = await store.record_authentication(context)
    assert isinstance(result, AgentIdentityFailure)
    assert "changed" in result.message
    assert identities.update_many.call_args.kwargs["where"] == {
        "agent_id": "agent-one",
        "revision": "old-revision",
        "active": True,
        "agent": {"is": {"enabled": True, "identity_managed": True}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [None, stored_agent(identity=None), stored_agent(identity_managed=False)])
async def test_stale_binding_cannot_bypass_lifecycle(agent: AgentResponse | None) -> None:
    store, _, _, _ = setup_store(agent=agent)
    assert isinstance(await store.resolve_verified_claims(CLAIMS), AgentIdentityFailure)


@pytest.mark.asyncio
async def test_unrelated_non_entra_claims_do_not_query_identity_store() -> None:
    store, agents, identities, _ = setup_store()
    assert await store.resolve_verified_claims({"sub": "ordinary-user"}) is None
    identities.find_unique.assert_not_awaited()
    agents.find_unique.assert_not_awaited()


HUMAN_CLAIMS: Final = {"iss": ISSUER, "tid": TENANT, "azp": CLIENT, "oid": HUMAN, "scp": "user_impersonation"}


@pytest.mark.asyncio
async def test_bound_agents_and_policy_failures_are_never_served_from_the_miss_cache() -> None:
    store, _, identities, _ = setup_store()
    assert isinstance(await store.resolve_verified_claims(CLAIMS), ManagedAgentContext)
    assert isinstance(await store.resolve_verified_claims(CLAIMS), ManagedAgentContext)
    assert identities.find_unique.await_count == 2
    identities.find_unique.side_effect = ConnectionError("database down")
    assert isinstance(await store.resolve_verified_claims(CLAIMS), AgentIdentityFailure)
    assert isinstance(await store.resolve_verified_claims(CLAIMS), AgentIdentityFailure)
    assert identities.find_unique.await_count == 4


@pytest.mark.asyncio
async def test_retired_client_cannot_fall_back_to_ordinary_user_authentication() -> None:
    from prisma.models import LiteLLM_RetiredAgentIdentity

    from litellm.repositories.table_repositories import RetiredAgentIdentityRepository

    identities: Final = AsyncMock()
    identities.find_unique.return_value = None
    retired: Final = AsyncMock()
    retired.find_unique.return_value = LiteLLM_RetiredAgentIdentity(
        binding_id="retired",
        agent_id="agent-one",
        provider="microsoft_entra",
        issuer=ISSUER,
        tenant_id=TENANT,
        client_id=CLIENT,
    )
    db: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_agentidentity=identities,
            litellm_retiredagentidentity=retired,
            litellm_agentstable=AsyncMock(),
            litellm_verifiedsubject=AsyncMock(),
        )
    )
    store: Final = AgentIdentityStore(
        AgentsRepository(db),
        AgentIdentityRepository(db),
        VerifiedSubjectRepository(db),
        RetiredAgentIdentityRepository(db),
    )
    result: Final = await store.resolve_verified_claims({**CLAIMS, "oid": HUMAN, "scp": "user_impersonation"})
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "identity_denied"
    assert "retired" in result.message


@pytest.mark.asyncio
async def test_missing_revision_cannot_create_entra_authentication_evidence() -> None:
    store, _, identities, _ = setup_store()
    result: Final = await store.record_authentication(ManagedAgentContext(agent_id="agent-one", mode="autonomous"))
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "identity_denied"
    identities.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_authentication_evidence_write_failure_is_not_success() -> None:
    store, _, identities, _ = setup_store()
    identities.update_many.side_effect = RuntimeError("writer unavailable")
    result: Final = await store.record_authentication(
        ManagedAgentContext(agent_id="agent-one", binding_revision="revision-one", mode="autonomous")
    )
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "policy_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [True, False])
async def test_retired_binding_denies_and_history_outage_cannot_become_legacy_fallback(unavailable: bool) -> None:

    database: Final = MagicMock()
    database.writer_db.litellm_agentidentity.find_unique = AsyncMock(return_value=None)
    database.writer_db.litellm_verifiedsubject.find_unique = AsyncMock(return_value=None)
    database.writer_db.litellm_retiredagentidentity.find_unique = AsyncMock(
        return_value={"client_id": CLIENT}, side_effect=RuntimeError("unavailable") if unavailable else None
    )
    result: Final = await AgentIdentityStore.from_client(database).resolve_verified_claims(CLAIMS)
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == ("policy_unavailable" if unavailable else "identity_denied")
    assert result.message == (
        "Retired agent identity could not be checked" if unavailable else "This agent identity binding has been retired"
    )


@pytest.mark.asyncio
async def test_new_binding_is_enforced_after_another_worker_commits_it() -> None:
    _, agents, identities, humans = setup_store()
    identities.find_unique.return_value = None
    retired: Final = AsyncMock()
    retired.find_unique.return_value = None
    db: Final = SimpleNamespace(
        writer_db=SimpleNamespace(
            litellm_agentstable=agents,
            litellm_agentidentity=identities,
            litellm_verifiedsubject=humans,
            litellm_retiredagentidentity=retired,
            litellm_retiredagent=retired,
        )
    )
    worker: Final = AgentIdentityStore.from_client(db, cache=UserApiKeyCache())
    claims: Final = {**CLAIMS, "oid": "55555555-5555-4555-8555-555555555555"}
    assert await worker.resolve_verified_claims(claims) is None
    identities.find_unique.return_value = BINDING
    denied: Final = await worker.resolve_verified_claims(claims)
    assert isinstance(denied, AgentIdentityFailure)
    assert denied.code == "identity_denied"
    assert "Application token contradicts" in denied.message
    assert identities.find_unique.await_count == 2


@pytest.mark.asyncio
async def test_non_string_subject_does_not_query_directory_ownership() -> None:
    store, _, _, humans = setup_store()
    assert await store.subject(ISSUER, TENANT, None) is None
    humans.find_unique.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_missing_or_unavailable_retirement_history_fails_closed(configured: bool) -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_retiredagent.find_unique = AsyncMock(side_effect=RuntimeError("history unavailable"))
    store: Final = AgentIdentityStore.from_client(database) if configured else setup_store()[0]
    result: Final = await store.retired_agent("deleted-agent")
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "policy_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["canonical-user", "another-user"])
async def test_interactive_enrollment_preserves_existing_subject_ownership(owner: str) -> None:
    store, _, _, humans = setup_store()
    humans.upsert.return_value = LiteLLM_VerifiedSubject(
        subject_id="subject-one",
        issuer=ISSUER,
        tenant_id=TENANT,
        oid=HUMAN,
        user_id=owner,
        kind="human",
        verified_via="sso_interactive",
        verified_at=datetime.now(timezone.utc),
    )
    result: Final = await store.enroll_interactive_human(
        MicrosoftInteractiveSubject(issuer=ISSUER, tenant_id=TENANT, oid=HUMAN), "canonical-user"
    )
    if owner == "canonical-user":
        assert result is None
    else:
        assert isinstance(result, AgentIdentityFailure)
        assert result.code == "identity_denied"
    assert humans.upsert.call_args.kwargs["data"]["update"] == {}
    assert humans.upsert.call_args.kwargs["data"]["create"]["user_id"] == "canonical-user"


@pytest.mark.asyncio
async def test_interactive_enrollment_outage_fails_closed() -> None:
    store, _, _, humans = setup_store()
    humans.upsert.side_effect = ConnectionError("writer unavailable")
    result: Final = await store.enroll_interactive_human(
        MicrosoftInteractiveSubject(issuer=ISSUER, tenant_id=TENANT, oid=HUMAN), "canonical-user"
    )
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "policy_unavailable"


@pytest.mark.asyncio
async def test_matching_revision_records_successful_authentication() -> None:
    store, _, identities, _ = setup_store()
    assert (
        await store.record_authentication(
            ManagedAgentContext(agent_id="agent-one", binding_revision="revision-one", mode="autonomous")
        )
        is None
    )
    identities.update_many.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("outage", [False, True])
async def test_resolver_maps_denials_and_outages_to_public_errors(outage: bool) -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentidentity.find_unique = AsyncMock(
        return_value=BINDING, side_effect=ConnectionError("unavailable") if outage else None
    )
    database.writer_db.litellm_verifiedsubject.find_unique = AsyncMock(return_value=None)
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=stored_agent(enabled=False))
    with pytest.raises(HTTPException) as exc:
        await resolve_managed_agent(CLAIMS, database)
    assert exc.value.status_code == (503 if outage else 403)


@pytest.mark.asyncio
async def test_resolver_preserves_unconfigured_and_unrelated_authentication() -> None:
    assert await resolve_managed_agent(CLAIMS, None) is None
    assert await resolve_managed_agent({"sub": "ordinary-user"}, MagicMock()) is None
    store, _, identities, _ = setup_store()
    identities.find_unique.return_value = None
    assert await store.resolve_verified_claims(CLAIMS) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [True, False])
async def test_application_and_unregistered_clients_do_not_depend_on_human_subject_storage(registered: bool) -> None:
    store, _, identities, humans = setup_store()
    identities.find_unique.return_value = BINDING if registered else None
    humans.find_unique.side_effect = RuntimeError("subject database unavailable")
    result: Final = await store.resolve_verified_claims(CLAIMS)
    if registered:
        assert isinstance(result, ManagedAgentContext)
        assert result.mode == "autonomous"
        assert result.user_id is None
    else:
        assert result is None
    humans.find_unique.assert_not_awaited()
