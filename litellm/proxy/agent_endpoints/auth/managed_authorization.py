from collections.abc import Mapping
from itertools import product
from typing import Final

from litellm.proxy._types import LiteLLMRoutes
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


def actor_admission_failure(
    agent: AgentResponse,
    context: ManagedAgentContext | None,
) -> AgentIdentityFailure | None:
    if not agent.enabled or agent.identity is None or not agent.identity.active:
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


def invocation_target(route: str, body: Mapping[str, object]) -> str | None:
    model: Final = body.get("model")
    if isinstance(model, str) and model.startswith("a2a/"):
        return model.removeprefix("a2a/") or None
    components: Final = tuple(route.strip("/").split("/"))
    path: Final = components[1:] if components and components[0] == "v1" else components
    return path[1] if len(path) >= 2 and path[0] == "a2a" else None
