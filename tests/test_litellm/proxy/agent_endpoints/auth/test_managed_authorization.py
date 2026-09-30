from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.managed_authorization import (
    actor_admission_failure,
    admit_managed_actor,
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


@pytest.mark.parametrize("mode", ["autonomous", "both", "delegated"])
def test_keys_cannot_impersonate_an_entra_bound_agent(mode: str) -> None:
    assert isinstance(actor_admission_failure(agent(execution_mode=mode), None), AgentIdentityFailure)


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


@pytest.mark.asyncio
async def test_deleted_agent_key_cannot_fall_back_to_unmanaged_authentication() -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
    database.writer_db.litellm_retiredagent.find_unique = AsyncMock(return_value={"original_agent_id": "deleted"})
    with pytest.raises(HTTPException, match="Agent no longer exists"):
        await admit_managed_actor(UserAPIKeyAuth(agent_id="deleted"), AgentIdentityStore.from_client(database))
    database.writer_db.litellm_retiredagent.find_unique.return_value = None
    auth: Final = UserAPIKeyAuth(agent_id="legacy-attribution-label")
    await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert auth.managed_agent_policy is None
    database.db.litellm_agentstable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_agent_history_outage_does_not_permit_legacy_fallback() -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
    database.writer_db.litellm_retiredagent.find_unique = AsyncMock(side_effect=RuntimeError("unavailable"))
    with pytest.raises(HTTPException) as failure:
        await admit_managed_actor(UserAPIKeyAuth(agent_id="deleted"), AgentIdentityStore.from_client(database))
    assert failure.value.status_code == 503


@pytest.mark.asyncio
async def test_agent_admission_database_outage_fails_closed() -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(side_effect=RuntimeError("DB unavailable"))
    with pytest.raises(HTTPException) as failure:
        await admit_managed_actor(UserAPIKeyAuth(agent_id="agent"), AgentIdentityStore.from_client(database))
    assert failure.value.status_code == 503


@pytest.mark.asyncio
async def test_human_authentication_does_not_load_an_agent() -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock()
    await admit_managed_actor(UserAPIKeyAuth(user_id="human"), AgentIdentityStore.from_client(database))
    database.writer_db.litellm_agentstable.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_agent_key_is_rejected_at_admission() -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=agent(enabled=False))
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
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=policy)
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


@pytest.mark.asyncio
async def test_legacy_jwt_cannot_adopt_an_agent_bound_on_another_worker() -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=agent(execution_mode="autonomous"))
    auth: Final = UserAPIKeyAuth(agent_id="agent", jwt_claims={"agent": "agent", "sub": "unrelated-subject"})
    with pytest.raises(HTTPException) as denied:
        await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert denied.value.status_code == 403
    assert auth.managed_agent_policy is None


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [False, True])
async def test_managed_context_or_binding_requires_database(monkeypatch: pytest.MonkeyPatch, bound: bool) -> None:
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry

    registry: Final = AgentRegistry()
    registry.register_agent(agent(identity_managed=bound, identity=BINDING if bound else None))
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    if not bound:
        auth.managed_agent_context = ManagedAgentContext(
            agent_id="agent", binding_revision="current", mode="autonomous"
        )
    with pytest.raises(HTTPException) as denied:
        await admit_managed_actor(auth, None)
    assert denied.value.status_code == 503


@pytest.mark.asyncio
async def test_autonomous_app_rejects_persisted_virtual_key_impersonation() -> None:
    policy: Final = agent(execution_mode="autonomous")
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=policy)
    auth: Final = UserAPIKeyAuth(agent_id="agent", api_key="persisted-key")
    with pytest.raises(HTTPException, match="bound identity provider token") as denied:
        await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert denied.value.status_code == 403
    assert auth.managed_agent_policy is None
    assert auth.billing_agent_policy is None


@pytest.mark.parametrize("mode,user", [("autonomous", None), ("delegated", "verified-human")])
def test_matching_identity_revision_and_execution_mode_pass_admission(mode: str, user: str | None) -> None:
    context: Final = ManagedAgentContext.model_validate(
        {"agent_id": "agent", "binding_revision": "current", "mode": mode, "user_id": user}
    )
    assert actor_admission_failure(agent(), context) is None


@pytest.mark.asyncio
async def test_bound_autonomous_actor_is_admitted_without_a_human() -> None:
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=agent())
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    auth.managed_agent_context = ManagedAgentContext(agent_id="agent", binding_revision="current", mode="autonomous")
    await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert auth.managed_agent_policy == agent()
    assert auth.billing_agent_policy == agent()
    assert auth.user_id is None


@pytest.mark.asyncio
async def test_admitted_managed_actor_requires_fresh_policy_so_revocations_bind_next_request() -> None:
    """Managed MCP grants (toolsets, access groups) are read through the shared resolvers, which only
    bypass the warm cache and the replica when the subject carries requires_fresh_policy"""
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=agent())
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    auth.managed_agent_context = ManagedAgentContext(agent_id="agent", binding_revision="current", mode="autonomous")
    assert auth.requires_fresh_policy is False
    await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert auth.requires_fresh_policy is True


async def test_jwt_delegation_verification_is_consumed_once_and_cannot_be_supplied_by_a_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.agent_endpoints.auth import agent_permission_handler

    policy: Final = agent()
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=policy)
    store: Final = AgentIdentityStore.from_client(database)
    grants: Final = AsyncMock(return_value=frozenset())
    monkeypatch.setattr(agent_permission_handler, "verified_human_agent_grants", grants)
    auth: Final = UserAPIKeyAuth.model_validate({"agent_id": "agent", "_managed_delegation_verified": True})
    assert auth._managed_delegation_verified is False
    auth.managed_agent_context = ManagedAgentContext(
        agent_id="agent", binding_revision="current", mode="delegated", user_id="human"
    )
    auth._managed_delegation_verified = True
    assert "_managed_delegation_verified" not in auth.model_dump()
    await admit_managed_actor(auth, store)
    grants.assert_not_awaited()
    assert auth._managed_delegation_verified is False
    with pytest.raises(HTTPException) as failure:
        await admit_managed_actor(auth, store)
    assert failure.value.status_code == 403
    grants.assert_awaited_once_with("human", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("database_available", (False, True))
async def test_ordinary_agent_admission_preserves_legacy_authentication(
    monkeypatch: pytest.MonkeyPatch, database_available: bool
) -> None:
    from litellm.proxy.agent_endpoints import agent_registry

    registry: Final = agent_registry.AgentRegistry()
    ordinary: Final = agent(identity_managed=False, identity=None)
    registry.register_agent(ordinary)
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=ordinary)
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    await admit_managed_actor(auth, AgentIdentityStore.from_client(database) if database_available else None)
    assert auth.agent_id == "agent"
    assert auth.managed_agent_policy is None
    assert auth.requires_fresh_policy is False
