from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth

from litellm.proxy.agent_endpoints.auth.managed_authorization import (
    actor_admission_failure,
    invocation_target,
)
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


def test_execution_mode_must_match_verified_token_mode() -> None:
    context: Final = ManagedAgentContext(agent_id="agent", binding_revision="current", mode="autonomous")
    failure: Final = actor_admission_failure(agent(execution_mode="delegated"), context)
    assert isinstance(failure, AgentIdentityFailure)
    assert "execution mode" in failure.message


@pytest.mark.parametrize("mode,user", [("autonomous", None), ("delegated", "verified-human")])
def test_matching_identity_revision_and_execution_mode_pass_admission(mode: str, user: str | None) -> None:
    context: Final = ManagedAgentContext.model_validate(
        {"agent_id": "agent", "binding_revision": "current", "mode": mode, "user_id": user}
    )
    assert actor_admission_failure(agent(), context) is None


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
    assert (
        managed_inference_request(route, {"model": "requested"}, {"completion_model": "allowed-default"}, "cli")[
            "model"
        ]
        == "requested"
    )


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
