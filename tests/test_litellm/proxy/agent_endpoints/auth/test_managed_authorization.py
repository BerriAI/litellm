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


def test_caller_cannot_construct_trusted_subject_or_policy() -> None:
    context: Final = ManagedAgentContext(
        agent_id="agent", binding_revision="current", mode="delegated", user_id="human"
    )
    auth: Final = UserAPIKeyAuth.model_validate(
        {
            "managed_agent_context": context,
            "requires_fresh_policy": True,
            "mcp_explicit_grants_only": True,
            "managed_agent_policy": agent(),
            "billing_agent_policy": agent(),
            "invoked_agent_id": "forged-target",
            "agent_invocation_cost": 0.0,
        }
    )
    assert auth.requires_fresh_policy is False
    assert auth.mcp_explicit_grants_only is False
    assert "mcp_explicit_grants_only" not in auth.model_dump()
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
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=target)
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
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=policy)
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    await admit_managed_actor(auth, AgentIdentityStore.from_client(database))
    assert auth.billing_agent_policy == policy
    assert auth.managed_agent_policy == (policy if managed else None)


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
@pytest.mark.parametrize("state,status", [("missing", 403), ("outage", 503), ("denied", 403), ("invalid-fee", 503)])
async def test_invocation_cannot_bypass_missing_policy_permission_or_invalid_price(
    monkeypatch: pytest.MonkeyPatch, state: str, status: int
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.auth.managed_authorization import prepare_agent_invocation

    registered: Final = agent(litellm_params={"cost_per_query": -1 if state == "invalid-fee" else 0.25})
    registry: Final = agent_registry.AgentRegistry()
    registry.register_agent(registered)
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(
        return_value=None if state == "missing" else registered,
        side_effect=RuntimeError("unavailable") if state == "outage" else None,
    )
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="grant", agents=[] if state == "denied" else ["agent"]
    )
    auth: Final = UserAPIKeyAuth(user_id="human", object_permission=permission)
    with pytest.raises(HTTPException) as failure:
        await prepare_agent_invocation(auth, "agent", AgentIdentityStore.from_client(database))
    assert failure.value.status_code == status
    assert auth.agent_invocation_cost is None


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
@pytest.mark.parametrize("managed_flag", [False, True])
async def test_managed_invocation_requires_database(monkeypatch: pytest.MonkeyPatch, managed_flag: bool) -> None:
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
    from litellm.proxy.agent_endpoints.auth.managed_authorization import prepare_agent_invocation

    registry: Final = AgentRegistry()
    registry.register_agent(agent(identity_managed=managed_flag))
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    with pytest.raises(HTTPException) as denied:
        await prepare_agent_invocation(UserAPIKeyAuth(user_id="human"), "agent", None)
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


@pytest.mark.asyncio
async def test_unknown_invocation_target_leaves_billing_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.auth.managed_authorization import prepare_agent_invocation

    monkeypatch.setattr(agent_registry, "global_agent_registry", agent_registry.AgentRegistry())
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    auth: Final = UserAPIKeyAuth(user_id="human")
    await prepare_agent_invocation(auth, "missing", None)
    assert auth.invoked_agent_id is None
    assert auth.billing_agent_policy is None


@pytest.mark.parametrize(
    "route,method,allowed",
    [
        ("/v1/agents", "GET", True),
        ("/v1/agents", "POST", False),
        ("/v1/chat/completions", "POST", True),
        ("/v1/chat/completions", "DELETE", False),
        ("/openai/deployments/model/chat/completions", "POST", True),
        ("/engines/openai/model/chat/completions", "POST", True),
        ("/openai/deployments/openai/model/images/generations", "POST", True),
        ("/openai/deployments/openai/model/images/edits", "POST", True),
        ("/v1beta/models/gemini-model:generateContent", "POST", True),
        ("/v1/realtime", "GET", True),
        ("/v1/realtime", "POST", False),
        ("/v1/realtime/client_secrets", "POST", False),
        ("/mcp/tools/call", "POST", True),
        ("/a2a/target/message/send", "POST", True),
        ("/v1/agents/target", "PATCH", False),
        ("/v1/responses/other-response", "GET", False),
        ("/v1/files", "GET", False),
        ("/v1/files", "POST", False),
        ("/openai/v1/files", "GET", False),
        ("/anthropic/v1/files", "GET", False),
    ],
)
def test_managed_route_scope_excludes_provider_resources(route: str, method: str, allowed: bool) -> None:
    from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_agent_route_allowed

    assert managed_agent_route_allowed(route, method) is allowed


@pytest.mark.parametrize(
    "route,body,settings,cli_model,path_model,expected",
    [
        ("/v1/chat/completions", {"model": "body"}, {"completion_model": "default"}, "cli", "path", "default"),
        ("/v1/moderations", {"model": "body"}, {"moderation_model": "default"}, "cli", None, "cli"),
        ("/v1/audio/speech", {"model": "body"}, {"completion_model": "ignored"}, None, None, "body"),
        ("/openai/deployments/path/embeddings", {"model": "body"}, {}, None, "path", "path"),
        ("/v1/messages/count_tokens", {"model": "body"}, {"completion_model": "ignored"}, "cli", None, "body"),
        ("/mcp/tools/call", {}, {"completion_model": "ignored"}, "cli", None, None),
        ("/v1/images/generations", {"model": "image"}, {"completion_model": "text"}, None, None, "image"),
        ("/v1/images/generations", {}, {"image_generation_model": "image"}, None, None, "image"),
        ("/v1/images/edits", {}, {"image_generation_model": "image"}, None, None, "image"),
        ("/v1/rerank", {"model": "reranker"}, {"completion_model": "text"}, "cli", None, "reranker"),
        ("/v1beta/models/path:countTokens", {"model": "body"}, {"completion_model": "text"}, "cli", "path", "path"),
    ],
)
def test_managed_inference_resolves_dispatch_precedence(route, body, settings, cli_model, path_model, expected):
    from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_inference_request

    assert managed_inference_request(route, body, settings, cli_model, path_model).get("model") == expected


def test_managed_inference_without_any_model_cannot_skip_model_grants():
    from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_inference_request

    with pytest.raises(HTTPException, match="explicit or configured model"):
        managed_inference_request("/v1/moderations", {}, {}, None)


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/images/generations", "/v1/images/edits"])
def test_managed_inference_query_model_takes_precedence_over_body(route: str):
    from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_inference_request

    assert managed_inference_request(route, {"model": "body"}, {}, None, query_model="query")["model"] == "query"


def test_managed_inference_ignores_unsupported_query_model():
    from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_inference_request

    assert (
        managed_inference_request("/v1/messages", {"model": "body"}, {}, None, query_model="query")["model"] == "body"
    )


@pytest.mark.parametrize("route", ["/realtime", "/v1/realtime", "/openai/v1/realtime"])
def test_managed_realtime_requires_a_model_and_ignores_completion_defaults(route: str) -> None:
    from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_inference_request

    with pytest.raises(HTTPException, match="explicit or configured model"):
        managed_inference_request(route, {}, {"completion_model": "allowed-default"}, "cli")
    assert managed_inference_request(
        route, {"model": "requested"}, {"completion_model": "allowed-default"}, "cli"
    )["model"] == "requested"
