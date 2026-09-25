from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.managed_authorization import (
    actor_admission_failure,
    admit_managed_actor,
    invocation_target,
)
from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityBinding, AgentIdentityFailure, ManagedAgentContext

BINDING: Final = AgentIdentityBinding(
    agent_id="agent",
    provider="microsoft_entra",
    tenant_id="tenant",
    client_id="client",
    service_principal_id="principal",
    issuer="issuer",
    revision="current",
)


def agent(**overrides: object) -> AgentResponse:
    return AgentResponse.model_validate(
        {
            "agent_id": "agent",
            "agent_name": "Agent",
            "agent_card_params": {},
            "identity": BINDING,
            "identity_managed": True,
            "execution_mode": "both",
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "state",
    [
        {"enabled": False},
        {"identity": None},
        {"identity": BINDING.model_copy(update={"active": False})},
        {"execution_mode": "delegated"},
    ],
)
def test_keys_cannot_bypass_lifecycle_or_delegated_only_mode(state: dict[str, object]) -> None:
    assert isinstance(actor_admission_failure(agent(**state), None), AgentIdentityFailure)


def test_keys_remain_valid_for_enabled_autonomous_agents() -> None:
    assert actor_admission_failure(agent(execution_mode="autonomous"), None) is None


@pytest.mark.parametrize(
    "context",
    [
        ManagedAgentContext(agent_id="agent", binding_revision="previous", mode="autonomous"),
        ManagedAgentContext(agent_id="another", binding_revision="current", mode="autonomous"),
        ManagedAgentContext(agent_id="agent", binding_revision="current", mode="delegated"),
    ],
)
def test_stale_binding_and_unverified_delegation_cannot_pass_admission(context: ManagedAgentContext) -> None:
    assert isinstance(actor_admission_failure(agent(), context), AgentIdentityFailure)


def test_caller_cannot_construct_trusted_subject_or_policy() -> None:
    context: Final = ManagedAgentContext(
        agent_id="agent", binding_revision="current", mode="delegated", user_id="human"
    )
    auth: Final = UserAPIKeyAuth.model_validate(
        {
            "managed_agent_context": context,
            "managed_agent_policy": agent(),
            "billing_agent_policy": agent(),
            "invoked_agent_id": "forged-target",
            "agent_invocation_cost": 0.0,
        }
    )
    assert auth.managed_agent_context is None
    assert auth.managed_agent_policy is None
    assert auth.billing_agent_policy is None
    assert auth.invoked_agent_id is None
    assert auth.agent_invocation_cost is None


@pytest.mark.asyncio
async def test_agent_budget_accumulates_across_credentials_and_denies_the_next_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy import proxy_server
    from litellm.proxy.agent_endpoints.auth.managed_authorization import check_agent_budget

    counters: Final = DualCache()
    counters.set_cache("spend:agent:agent", 0.0)
    counters.set_cache("spend:key:first", 0.0)
    counters.set_cache("spend:key:second", 0.0)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", counters)
    policy: Final = agent(spend=0, litellm_budget_table={"budget_id": "budget", "max_budget": 0.5})
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    auth.billing_agent_policy = policy
    await check_agent_budget(auth)
    await proxy_server.increment_spend_counters(
        token="first", team_id=None, user_id=None, response_cost=0.3, billing_agent_id="agent"
    )
    await check_agent_budget(auth)
    await proxy_server.increment_spend_counters(
        token="second", team_id=None, user_id=None, response_cost=0.3, billing_agent_id="agent"
    )
    with pytest.raises(litellm.BudgetExceededError):
        await check_agent_budget(auth)
    assert await counters.async_get_cache("spend:agent:agent") == pytest.approx(0.6)
    assert await counters.async_get_cache("spend:key:first") == pytest.approx(0.3)
    assert await counters.async_get_cache("spend:key:second") == pytest.approx(0.3)


@pytest.mark.asyncio
@pytest.mark.parametrize("autonomous", (True, False))
async def test_invocation_prepares_target_fee_for_the_correct_agent(
    monkeypatch: pytest.MonkeyPatch,
    autonomous: bool,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.auth.managed_authorization import prepare_agent_invocation
    from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore

    target: Final = agent(litellm_params={"cost_per_query": 0.25})
    registry: Final = agent_registry.AgentRegistry()
    registry.register_agent(target)
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    database: Final = MagicMock()
    database.db.litellm_agentstable.find_unique = AsyncMock(return_value=target)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    permission: Final = LiteLLM_ObjectPermissionTable(object_permission_id="invoke-grant", agents=["agent"])
    auth: Final = UserAPIKeyAuth(
        agent_id="caller" if autonomous else None,
        user_id=None if autonomous else "human",
        object_permission=permission,
    )
    if autonomous:
        caller: Final = agent(agent_id="caller", object_permission=permission.model_dump())
        auth.managed_agent_policy = caller
        auth.billing_agent_policy = caller
    await prepare_agent_invocation(auth, "agent", AgentIdentityStore.from_client(database))
    assert auth.agent_invocation_cost == pytest.approx(0.25)
    assert auth.invoked_agent_id == "agent"
    assert auth.billing_agent_policy is not None
    assert auth.billing_agent_policy.agent_id == ("caller" if autonomous else "agent")


@pytest.mark.asyncio
async def test_deleted_agent_key_cannot_fall_back_to_unmanaged_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import HTTPException

    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.auth.managed_authorization import admit_managed_actor
    from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore

    registry: Final = agent_registry.AgentRegistry()
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    database: Final = MagicMock()
    database.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
    with pytest.raises(HTTPException, match="Agent no longer exists"):
        await admit_managed_actor(UserAPIKeyAuth(agent_id="deleted"), AgentIdentityStore.from_client(database))
    registry.register_agent(AgentResponse(agent_id="configured", agent_name="Configured", agent_card_params={}))
    auth: Final = UserAPIKeyAuth(agent_id="configured")
    await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert auth.managed_agent_policy is None


@pytest.mark.parametrize(
    "route,body,expected",
    [
        ("/a2a/agent", {}, "agent"),
        ("/v1/a2a/agent/", {}, "agent"),
        ("/v1/chat/completions", {"model": "a2a/Readable name"}, "Readable name"),
        ("/v1/chat/completions", {"model": "a2a/"}, None),
        ("/v1/chat/completions", {"model": "ordinary-model"}, None),
        ("/a2a", {}, None),
    ],
)
def test_invocation_routes_resolve_the_same_target(route: str, body: dict[str, object], expected: str | None) -> None:
    assert invocation_target(route, body) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [True, False])
async def test_aggregate_budget_applies_to_both_entra_and_legacy_agent_keys(managed: bool) -> None:
    policy: Final = agent(identity_managed=managed, litellm_budget_table={"budget_id": "budget", "max_budget": 0.5})
    database: Final = MagicMock()
    database.db.litellm_agentstable.find_unique = AsyncMock(return_value=policy)
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert auth.billing_agent_policy == policy
    assert auth.managed_agent_policy == (policy if managed else None)


@pytest.mark.asyncio
async def test_agent_admission_database_outage_fails_closed() -> None:
    database: Final = MagicMock()
    database.db.litellm_agentstable.find_unique = AsyncMock(side_effect=RuntimeError("DB unavailable"))
    with pytest.raises(HTTPException) as failure:
        await admit_managed_actor(UserAPIKeyAuth(agent_id="agent"), AgentIdentityStore.from_client(database))
    assert failure.value.status_code == 503


@pytest.mark.asyncio
async def test_human_authentication_does_not_load_an_agent() -> None:
    database: Final = MagicMock()
    database.db.litellm_agentstable.find_unique = AsyncMock()
    await admit_managed_actor(UserAPIKeyAuth(user_id="human"), AgentIdentityStore.from_client(database))
    database.db.litellm_agentstable.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_agent_key_is_rejected_at_admission() -> None:
    database: Final = MagicMock()
    database.db.litellm_agentstable.find_unique = AsyncMock(return_value=agent(enabled=False))
    with pytest.raises(HTTPException) as failure:
        await admit_managed_actor(UserAPIKeyAuth(agent_id="agent"), AgentIdentityStore.from_client(database))
    assert failure.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("permitted", [True, False])
async def test_verified_human_still_needs_an_explicit_agent_invocation_grant(
    monkeypatch: pytest.MonkeyPatch,
    permitted: bool,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_UserTable
    from litellm.proxy.auth import auth_checks

    policy: Final = agent()
    database: Final = MagicMock()
    database.db.litellm_agentstable.find_unique = AsyncMock(return_value=policy)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="human-grants",
        agents=["agent"] if permitted else [],
    )
    human: Final = LiteLLM_UserTable(user_id="human", teams=[], object_permission=permission)
    monkeypatch.setattr(auth_checks, "get_user_object", AsyncMock(return_value=human))
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    auth.managed_agent_context = ManagedAgentContext(
        agent_id="agent",
        binding_revision="current",
        mode="delegated",
        user_id="human",
    )
    if permitted:
        await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
        assert auth.managed_agent_policy == policy
        assert auth.billing_agent_policy == policy
    else:
        with pytest.raises(HTTPException) as failure:
            await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
        assert failure.value.status_code == 403


def test_execution_mode_must_match_verified_token_mode() -> None:
    context: Final = ManagedAgentContext(agent_id="agent", binding_revision="current", mode="autonomous")
    failure: Final = actor_admission_failure(agent(execution_mode="delegated"), context)
    assert isinstance(failure, AgentIdentityFailure)
    assert "execution mode" in failure.message
