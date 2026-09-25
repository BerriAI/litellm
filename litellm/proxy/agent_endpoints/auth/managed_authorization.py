from collections.abc import Mapping
from itertools import product
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import Field, TypeAdapter, ValidationError

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
from litellm.proxy.agent_endpoints.managed_identity import raise_identity_failure
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityFailure, ManagedAgentContext

_MANAGED_REALTIME_ROUTES: Final = frozenset(("/realtime", "/v1/realtime", "/openai/v1/realtime"))
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


def managed_agent_route_allowed(route: str, method: str | None) -> bool:
    from litellm.proxy.auth.route_checks import RouteChecks

    if route in ("/agents", "/v1/agents"):
        return method in (None, "GET", "HEAD")
    if route in _MANAGED_REALTIME_ROUTES:
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

    if route in _MANAGED_REALTIME_ROUTES:
        model: Final = query_model or body.get("model")
        if not isinstance(model, str) or not model:
            raise_identity_failure(
                AgentIdentityFailure(message="Managed inference requires an explicit or configured model")
            )
        return {**body, "model": model}
    if route not in _MANAGED_MODEL_ROUTES and not RouteChecks.check_route_access(route, _MANAGED_MODEL_PATHS):
        return dict(body)
    from litellm.proxy.common_utils.http_parsing_utils import resolve_inference_model

    kind: Final = (
        "image_generation"
        if route.endswith("/images/generations")
        else "image_edit"
        if route.endswith("/images/edits")
        else "moderation"
        if route.endswith(("/moderations", "/audio/transcriptions"))
        else "speech"
        if route.endswith("/audio/speech")
        else "body"
        if route.endswith(("/rerank", "/messages/count_tokens"))
        else "path"
        if route.endswith(":countTokens")
        else "completion"
    )
    endpoint_model: Final = path_model or (
        query_model if route.endswith(("/completions", "/embeddings", "/images/generations", "/images/edits")) else None
    )
    effective: Final = resolve_inference_model(body.get("model"), settings, cli_model, endpoint_model, kind=kind)
    if not isinstance(effective, str) or not effective:
        raise_identity_failure(
            AgentIdentityFailure(message="Managed inference requires an explicit or configured model")
        )
    return {**body, "model": effective}


async def admit_managed_actor(auth: UserAPIKeyAuth, store: AgentIdentityStore | None) -> None:
    if auth.agent_id is None:
        return
    if store is None:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        registered: Final = global_agent_registry.get_agent_by_id(auth.agent_id)
        if auth.managed_agent_context is not None or (
            registered is not None and (registered.identity_managed or registered.identity is not None)
        ):
            raise_identity_failure(
                AgentIdentityFailure(code="policy_unavailable", message="Managed agent policy requires a database")
            )
        return
    agent: Final = await store.agent(auth.agent_id)
    if isinstance(agent, AgentIdentityFailure):
        raise_identity_failure(agent)
    if agent is None:
        retired: Final = await store.retired_agent(auth.agent_id)
        if isinstance(retired, AgentIdentityFailure):
            raise_identity_failure(retired)
        if auth.managed_agent_context is not None or retired:
            raise_identity_failure(AgentIdentityFailure(message="Agent no longer exists"))
        return
    if not agent.identity_managed:
        return
    if auth.jwt_claims and auth.managed_agent_context is None:
        raise_identity_failure(AgentIdentityFailure(message="A managed agent requires a matching verified identity"))
    failure: Final = actor_admission_failure(agent, auth.managed_agent_context)
    if failure is not None:
        raise_identity_failure(failure)
    auth.managed_agent_policy = agent
    auth.billing_agent_policy = agent
    if auth.managed_agent_context is not None and auth.managed_agent_context.mode == "delegated":
        from litellm.proxy.agent_endpoints.auth.agent_permission_handler import verified_human_agent_grants

        grants: Final = await verified_human_agent_grants(auth.managed_agent_context.user_id)
        if agent.agent_id not in grants:
            raise_identity_failure(
                AgentIdentityFailure(message="The delegated user is not permitted to invoke this agent")
            )


def actor_admission_failure(
    agent: AgentResponse,
    context: ManagedAgentContext | None,
) -> AgentIdentityFailure | None:
    if (
        not agent.enabled
        or not agent.directory_active
        or agent.directory_access_group_ids == ()
        or agent.identity is None
        or not agent.identity.active
    ):
        return AgentIdentityFailure(message="Agent execution is disabled")
    if context is None:
        return AgentIdentityFailure(message="This agent requires its bound identity provider token")
    if context.agent_id != agent.agent_id or context.binding_revision != agent.identity.revision:
        return AgentIdentityFailure(message="Agent identity changed during authentication; retry")
    if agent.execution_mode not in (context.mode, "both"):
        return AgentIdentityFailure(message="Agent is not enabled for this execution mode")
    if context.mode == "delegated" and not context.user_id:
        return AgentIdentityFailure(message="A verified human subject is required")
    return None


_INVOCATION_COST: Final = TypeAdapter(Annotated[float, Field(ge=0, allow_inf_nan=False)])


def invocation_target(route: str, body: Mapping[str, object]) -> str | None:
    model: Final = body.get("model")
    if isinstance(model, str) and model.startswith("a2a/"):
        return model.removeprefix("a2a/") or None
    components: Final = tuple(route.strip("/").split("/"))
    path: Final = components[1:] if components and components[0] == "v1" else components
    return path[1] if len(path) >= 2 and path[0] == "a2a" else None


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
