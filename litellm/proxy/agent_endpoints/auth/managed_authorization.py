from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from types import MappingProxyType
from typing import Annotated, Final, Literal

from pydantic import Field, TypeAdapter, ValidationError

from litellm._auth import managed

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
from litellm.proxy.agent_endpoints.managed_identity import raise_identity_failure
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityFailure, ManagedAgentContext

_MANAGED_REALTIME_ROUTES: Final = frozenset(
    (
        "/realtime",
        "/v1/realtime",
        "/openai/v1/realtime",
        "/realtime/translations",
        "/v1/realtime/translations",
    )
)
_MANAGED_LIVE_ROUTES: Final = frozenset(("/live/sessions", "/v1/live/sessions", "/openai/v1/live/sessions"))
_MANAGED_MODEL_ROUTES: Final = frozenset(
    f"{prefix}/{operation}"
    for prefix, operation in product(
        ("", "/v1"),
        (
            "chat/completions",
            "completions",
            "embeddings",
            "responses",
            "messages",
            "messages/count_tokens",
            "images/generations",
            "images/edits",
            "audio/transcriptions",
            "audio/speech",
            "moderations",
            "rerank",
            "decisions",
            "systemone",
            "extraction",
            "ocr",
        ),
    )
) | frozenset(
    (
        "/openai/v1/responses",
        "/v2/rerank",
        "/claude_code_gateway/v1/messages",
        "/claude_code_gateway/v1/messages/count_tokens",
        "/cursor/chat/completions",
    )
)
_MANAGED_MODEL_PATHS: Final = (
    "/engines/{model:path}/chat/completions",
    "/engines/{model:path}/completions",
    "/engines/{model:path}/embeddings",
    "/openai/deployments/{model:path}/chat/completions",
    "/openai/deployments/{model:path}/completions",
    "/openai/deployments/{model:path}/embeddings",
    "/openai/deployments/{model:path}/images/generations",
    "/openai/deployments/{model:path}/images/edits",
    "/v1beta/models/{model_name:path}:countTokens",
    "/v1beta/models/{model_name:path}:generateContent",
    "/v1beta/models/{model_name:path}:streamGenerateContent",
    "/models/{model_name:path}:countTokens",
    "/models/{model_name:path}:generateContent",
    "/models/{model_name:path}:streamGenerateContent",
)
_MANAGED_MCP_ROUTES: Final = tuple(
    route for route in LiteLLMRoutes.mcp_inference_routes.value if route not in ("/token", "/introspect")
)


_MODEL_ROUTE_KINDS: Final[
    Mapping[str, Literal["image_generation", "image_edit", "moderation", "speech", "body", "path"]]
] = MappingProxyType(
    {
        "/images/generations": "image_generation",
        "/images/edits": "image_edit",
        "/moderations": "moderation",
        "/audio/transcriptions": "moderation",
        "/audio/speech": "speech",
        "/rerank": "body",
        "/decisions": "body",
        "/systemone": "body",
        "/extraction": "body",
        "/messages/count_tokens": "body",
        ":countTokens": "path",
    }
)


def managed_agent_route_allowed(route: str, method: str | None) -> bool:
    from litellm.proxy.auth.route_checks import RouteChecks

    if route in ("/agents", "/v1/agents"):
        return method in (None, "GET", "HEAD")
    if route in _MANAGED_REALTIME_ROUTES or route in _MANAGED_LIVE_ROUTES:
        return method in (None, "GET")
    if route in _MANAGED_MODEL_ROUTES or RouteChecks.check_route_access(route, _MANAGED_MODEL_PATHS):
        return method in (None, "POST")
    return RouteChecks.check_route_access(route, _MANAGED_MCP_ROUTES) or RouteChecks.check_route_access(
        route, LiteLLMRoutes.agent_inference_routes.value
    )


def managed_inference_request(
    route: str,
    body: Mapping[str, object],
    settings: Mapping[str, object],
    cli_model: str | None,
    path_model: object = None,
    query_model: object = None,
) -> dict[str, object]:
    from litellm.proxy.auth.route_checks import RouteChecks

    if route in _MANAGED_LIVE_ROUTES:
        return dict(body)
    if route in _MANAGED_REALTIME_ROUTES:
        from litellm.proxy.common_utils.http_parsing_utils import resolve_realtime_route_model

        model: Final = resolve_realtime_route_model(
            query_model if isinstance(query_model, str) else None, None, route.endswith("/realtime/translations")
        )
        if not isinstance(model, str) or not model:
            raise_identity_failure(
                AgentIdentityFailure(message="Managed inference requires an explicit or configured model")
            )
        return {**body, "model": model}
    if route not in _MANAGED_MODEL_ROUTES and not RouteChecks.check_route_access(route, _MANAGED_MODEL_PATHS):
        return dict(body)
    from litellm.proxy.common_utils.http_parsing_utils import resolve_inference_model

    kind: Final = next((kind for suffix, kind in _MODEL_ROUTE_KINDS.items() if route.endswith(suffix)), "completion")
    endpoint_model: Final = path_model or (
        query_model if route.endswith(("/completions", "/embeddings", "/images/generations", "/images/edits")) else None
    )
    effective: Final = resolve_inference_model(body.get("model"), settings, cli_model, endpoint_model, kind=kind)
    if not isinstance(effective, str) or not effective:
        raise_identity_failure(
            AgentIdentityFailure(message="Managed inference requires an explicit or configured model")
        )
    return {**body, "model": effective}


def managed_agent_policy(auth: "UserAPIKeyAuth | None") -> AgentResponse | None:
    """The admitted managed policy, or ``None`` when the subject was never admitted as a managed agent.

    ``admit_managed_actor`` only assigns ``managed_agent_policy`` after ``actor_admission_failure``
    has verified the bound context, so an ``AgentResponse`` here means admission succeeded.
    """
    policy: Final = auth.managed_agent_policy if auth is not None else None
    return policy if isinstance(policy, AgentResponse) else None


async def admit_managed_actor(auth: UserAPIKeyAuth, store: AgentIdentityStore | None) -> None:
    delegation_verified: Final = auth._managed_delegation_verified  # pyright: ignore[reportPrivateUsage]  # the one-shot delegation marker is a PrivateAttr by design
    auth._managed_delegation_verified = False  # pyright: ignore[reportPrivateUsage]  # consumed here so a replayed token cannot reuse it
    facts: Final = managed.ActorLookup(
        agent_id=auth.agent_id,
        has_database=store is not None,
        context=_managed_context(auth.managed_agent_context),
        jwt_claims=bool(auth.jwt_claims),
        delegation_verified=delegation_verified,
    )
    await _drive_actor(auth, store, facts)


async def _drive_actor(
    auth: UserAPIKeyAuth,
    store: AgentIdentityStore | None,
    facts: managed.ActorLookup,
    agent: AgentResponse | None = None,
) -> None:
    decision: Final = managed.decide(facts)
    if decision.denial is not None:
        raise_identity_failure(_managed_failure(decision.denial))
    if decision.action == "Allow":
        return
    if decision.action == "ReadRegistry":
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        assert auth.agent_id is not None
        registered: Final = global_agent_registry.get_agent_by_id(auth.agent_id)
        await _drive_actor(
            auth,
            store,
            replace(
                facts,
                registry_loaded=True,
                registry_managed=registered is not None
                and (registered.identity_managed or registered.identity is not None),
            ),
        )
        return
    if decision.action == "ReadAgent":
        assert store is not None and auth.agent_id is not None
        loaded: Final = await store.agent(auth.agent_id)
        if isinstance(loaded, AgentIdentityFailure):
            raise_identity_failure(loaded)
        await _drive_actor(
            auth,
            store,
            replace(facts, agent_loaded=True, agent=_managed_agent(loaded) if loaded is not None else None),
            loaded,
        )
        return
    if decision.action == "ReadRetired":
        assert store is not None and auth.agent_id is not None
        retired: Final = await store.retired_agent(auth.agent_id)
        if isinstance(retired, AgentIdentityFailure):
            raise_identity_failure(retired)
        await _drive_actor(auth, store, replace(facts, retired=retired), agent)
        return
    if decision.action == "AssignPolicy":
        assert agent is not None
        auth.managed_agent_policy = agent
        auth.billing_agent_policy = agent
        auth.requires_fresh_policy = True
        await _drive_actor(auth, store, replace(facts, policy_assigned=True), agent)
        return
    if decision.action == "ReadHumanGrants":
        from litellm.proxy.agent_endpoints.auth.agent_permission_handler import verified_human_agent_grants

        assert auth.managed_agent_context is not None
        grants: Final = await verified_human_agent_grants(auth.managed_agent_context.user_id, auth.team_id)
        await _drive_actor(auth, store, replace(facts, human_grants=tuple(grants)), agent)
        return
    raise RuntimeError("Invalid managed actor decision")


def _managed_agent(agent: AgentResponse) -> managed.Agent:
    return managed.Agent(
        agent_id=agent.agent_id,
        enabled=agent.enabled,
        identity_managed=agent.identity_managed,
        identity=managed.BoundIdentity(active=agent.identity.active, revision=agent.identity.revision)
        if agent.identity is not None
        else None,
        execution_mode=agent.execution_mode,
    )


def _managed_context(context: ManagedAgentContext | None) -> managed.Context | None:
    return (
        managed.Context(
            agent_id=context.agent_id,
            binding_revision=context.binding_revision,
            mode=context.mode,
            user_id=context.user_id,
        )
        if context is not None
        else None
    )


def _managed_failure(denial: managed.ManagedDenial) -> AgentIdentityFailure:
    if denial == "DatabaseRequired":
        return AgentIdentityFailure(code="policy_unavailable", message="Managed agent policy requires a database")
    messages: Final[Mapping[managed.ManagedDenial, str]] = MappingProxyType(
        {
            "Disabled": "Agent execution is disabled",
            "MissingContext": "This agent requires its bound identity provider token",
            "ChangedIdentity": "Agent identity changed during authentication; retry",
            "ExecutionMode": "Agent is not enabled for this execution mode",
            "MissingHuman": "A verified human subject is required",
            "Removed": "Agent no longer exists",
            "VerifiedIdentityRequired": "A managed agent requires a matching verified identity",
            "HumanNotPermitted": "The delegated user is not permitted to invoke this agent",
        }
    )
    return AgentIdentityFailure(message=messages[denial])


def actor_admission_failure(
    agent: AgentResponse,
    context: ManagedAgentContext | None,
) -> AgentIdentityFailure | None:
    decision: Final = managed.decide(managed.Actor(_managed_agent(agent), _managed_context(context)))
    return _managed_failure(decision.denial) if decision.denial is not None else None


_INVOCATION_COST: Final = TypeAdapter(Annotated[float, Field(ge=0, allow_inf_nan=False)])


def invocation_target(route: str, body: Mapping[str, object]) -> str | None:
    components: Final = tuple(route.strip("/").split("/"))
    path: Final = components[1:] if components and components[0] == "v1" else components
    if len(path) >= 2 and path[0] == "a2a":
        return path[1] or None
    model: Final = body.get("model")
    return model.removeprefix("a2a/") or None if isinstance(model, str) and model.startswith("a2a/") else None


async def prepare_agent_invocation(
    auth: UserAPIKeyAuth, target_name: str, store: AgentIdentityStore | None, *, billable: bool = True
) -> None:
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import AgentRequestHandler
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    registered: Final = await get_agent_with_read_through(target_name)
    if registered is None:
        return
    registered_managed: Final = registered.identity_managed or registered.identity is not None
    if store is None and registered_managed:
        raise_identity_failure(
            AgentIdentityFailure(code="policy_unavailable", message="Managed agent policy requires a database")
        )
    target: Final = await store.agent(registered.agent_id) if store is not None else None
    if isinstance(target, AgentIdentityFailure):
        raise_identity_failure(target)
    if target is None and registered_managed:
        raise_identity_failure(AgentIdentityFailure(message="Invoked agent no longer exists"))
    effective: Final = target if target is not None else registered
    if not effective.identity_managed and auth.managed_agent_policy is None:
        return
    if not await AgentRequestHandler.is_agent_allowed(effective.agent_id, auth):
        raise_identity_failure(AgentIdentityFailure(message="The caller is not permitted to invoke this agent"))
    auth.invoked_agent_id = effective.agent_id
    auth.invoked_agent_policy = effective
    if auth.agent_id is None and effective.identity_managed:
        auth.billing_agent_policy = effective
    raw_fee: Final = (effective.litellm_params or MappingProxyType({})).get("cost_per_query", 0.0) if billable else 0.0
    try:
        fee: Final = _INVOCATION_COST.validate_python(raw_fee)
    except ValidationError:
        raise_identity_failure(
            AgentIdentityFailure(code="policy_unavailable", message="Agent invocation price is invalid")
        )
    auth.agent_invocation_cost = fee
