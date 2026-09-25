from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from prisma.models import LiteLLM_VerifiedHumanSubject

from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
from litellm.repositories.table_repositories import (
    AgentIdentityRepository,
    AgentsRepository,
    VerifiedHumanSubjectRepository,
)
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityBinding, AgentIdentityFailure, ManagedAgentContext

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
    human: LiteLLM_VerifiedHumanSubject | None = None,
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
            litellm_verifiedhumansubject=humans,
        )
    )
    return (
        AgentIdentityStore(AgentsRepository(db), AgentIdentityRepository(db), VerifiedHumanSubjectRepository(db)),
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
    humans.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_lifecycle_is_read_on_every_request_without_cached_allow() -> None:
    store, agents, _, _ = setup_store()
    agents.find_unique.side_effect = [stored_agent(), stored_agent(enabled=False)]
    assert isinstance(await store.resolve_verified_claims(CLAIMS), ManagedAgentContext)
    denial: Final = await store.resolve_verified_claims(CLAIMS)
    assert isinstance(denial, AgentIdentityFailure)
    assert denial.code == "identity_denied"


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
async def test_unclassified_delegated_subject_cannot_authenticate_as_a_user() -> None:
    store, _, _, _ = setup_store()
    result: Final = await store.resolve_verified_claims(
        {**CLAIMS, "oid": HUMAN, "scp": "user_impersonation", "idtyp": "user"}
    )
    assert isinstance(result, AgentIdentityFailure)
    assert "first sign in" in result.message


@pytest.mark.asyncio
async def test_delegated_subject_uses_canonical_sso_user_not_email_claim() -> None:
    human: Final = LiteLLM_VerifiedHumanSubject(
        subject_id="subject-one",
        issuer=ISSUER,
        tenant_id=TENANT,
        oid=HUMAN,
        user_id="canonical-user",
        verified_via="sso_interactive",
        verified_at=datetime.now(timezone.utc),
    )
    store, _, _, humans = setup_store(human=human)
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


@pytest.mark.asyncio
async def test_rebinding_during_authentication_does_not_mark_new_identity_verified() -> None:
    store, _, identities, _ = setup_store()
    identities.update_many.return_value = 0
    context: Final = ManagedAgentContext(agent_id="agent-one", binding_revision="old-revision", mode="autonomous")
    result: Final = await store.record_authentication(context)
    assert isinstance(result, AgentIdentityFailure)
    assert "changed" in result.message
    assert identities.update_many.call_args.kwargs["where"] == {"agent_id": "agent-one", "revision": "old-revision"}


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
            litellm_verifiedhumansubject=AsyncMock(),
        )
    )
    store: Final = AgentIdentityStore(
        AgentsRepository(db),
        AgentIdentityRepository(db),
        VerifiedHumanSubjectRepository(db),
        RetiredAgentIdentityRepository(db),
    )
    result: Final = await store.resolve_verified_claims({**CLAIMS, "oid": HUMAN, "scp": "user_impersonation"})
    assert isinstance(result, AgentIdentityFailure)
    assert result.code == "identity_denied"
    assert "retired" in result.message
