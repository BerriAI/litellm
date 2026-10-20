"""
A2A Agent Routing

Handles routing for A2A agents (models with "a2a/<agent-name>" prefix).
Looks up agents in the registry and injects their API base URL.
"""

from collections.abc import Mapping
from typing import Any, Final

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    agent_completion_kwargs,
    bridge_model_name,
)
from litellm.constants import A2A_CHAT_COMPLETION_BRIDGE_PROVIDERS
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.databricks_oauth import (
    resolve_databricks_backend_auth,
)
from litellm.proxy.agent_endpoints.utils import merge_agent_headers
from litellm.types.agents import AgentResponse

_HEADERS: Final = TypeAdapter(Mapping[str, str])
_OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
_AGENT_PINNED_KWARGS: Final = ("api_base", "api_key")
_AGENT_LAYERED_KWARGS: Final = ("custom_inputs", "extra_body")
_CLIENT_TARGET_KWARGS: Final = ("api_base", "base_url")
_PROXY_HEADER_PREFIX: Final = "x-litellm-"


def _agent_extra_headers(agent_kwargs: Mapping[str, object]) -> Mapping[str, str] | None:
    extra_headers: Final = agent_kwargs.get("extra_headers")
    return None if extra_headers is None else _HEADERS.validate_python(extra_headers)


def _drop_params_enabled(data: Mapping[str, object], litellm_params: Mapping[str, object]) -> bool:
    return litellm.drop_params is True or data.get("drop_params") is True or litellm_params.get("drop_params") is True


def _client_extra_headers(data: Mapping[str, object], litellm_params: Mapping[str, object]) -> Mapping[str, str] | None:
    extra_headers: Final = data.get("extra_headers")
    if not extra_headers:
        return None
    try:
        headers: Final = _HEADERS.validate_python(extra_headers)
    except ValidationError as e:
        if _drop_params_enabled(data, litellm_params):
            return None
        raise HTTPException(
            status_code=400,
            detail=(
                "extra_headers must be an object mapping header names to string values, "
                "or set drop_params: true to drop it"
            ),
        ) from e
    return {name: value for name, value in headers.items() if not name.lower().startswith(_PROXY_HEADER_PREFIX)}


def _mappings(client_value: object, agent_value: object) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    try:
        return _OBJECT_MAPPING.validate_python(client_value), _OBJECT_MAPPING.validate_python(agent_value)
    except ValidationError:
        return None


def _agent_entries_over_client(client_value: object, agent_value: object) -> object:
    mappings: Final = _mappings(client_value, agent_value)
    return agent_value if mappings is None else {**mappings[0], **mappings[1]}


def _agent_keys_over_client(client_value: object, agent_value: object) -> object:
    mappings: Final = _mappings(client_value, agent_value)
    if mappings is None:
        return agent_value
    client, agent = mappings
    return {**client, **{key: _agent_entries_over_client(client.get(key), value) for key, value in agent.items()}}


def _with_headers(data: Mapping[str, object], extra_headers: Mapping[str, str] | None) -> Mapping[str, object]:
    request: Final = {key: value for key, value in data.items() if key != "extra_headers"}
    return {**request, **({"extra_headers": extra_headers} if extra_headers else {})}


def _bridge_request_data(
    data: Mapping[str, object],
    litellm_params: Mapping[str, object],
    static_headers: Mapping[str, str] | None,
) -> Mapping[str, object]:
    agent_kwargs: Final = agent_completion_kwargs(litellm_params)
    pinned: Final = {key: agent_kwargs[key] for key in _AGENT_PINNED_KWARGS if key in agent_kwargs}
    agent_headers: Final = merge_agent_headers(
        dynamic_headers=static_headers,
        static_headers=_agent_extra_headers(agent_kwargs),
    )
    extra_headers: Final = merge_agent_headers(
        dynamic_headers=_client_extra_headers(data, litellm_params),
        static_headers=agent_headers,
    )
    layered: Final = {
        key: _agent_keys_over_client(data[key], agent_kwargs[key])
        for key in _AGENT_LAYERED_KWARGS
        if key in data and key in agent_kwargs
    }
    return _with_headers(
        {**agent_kwargs, **data, **layered, **pinned, "model": bridge_model_name(litellm_params)},
        extra_headers,
    )


async def route_a2a_agent_request(
    data: dict[str, object],  # mutable-ok: the URL agent path writes api_base into the caller's request
    route_type: str,
    user_api_key_dict: UserAPIKeyAuth | None = None,
) -> Any | None:
    """
    Route A2A agent requests directly to litellm with injected API base.

    Returns None if not an A2A request (allows normal routing to continue).
    """
    # Import here to avoid circular imports
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
        AgentRequestHandler,
    )
    from litellm.proxy.common_utils.registry_read_through import (
        get_agent_with_read_through,
    )
    from litellm.proxy.route_llm_request import (
        ROUTE_ENDPOINT_MAPPING,
        ProxyModelNotFoundError,
    )

    model_name: Final = data.get("model", "")

    # Check if this is an A2A agent request
    if not isinstance(model_name, str) or not model_name.startswith("a2a/"):
        return None

    # Extract agent name (e.g., "a2a/my-agent" -> "my-agent")
    agent_name: Final = model_name[4:]

    # Look up agent in registry
    agent: Final = await get_agent_with_read_through(agent_name)
    if agent is None:
        verbose_proxy_logger.error("[A2A] Agent '%s' not found in registry", agent_name)
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name, retryable_with_model_read_through=False)

    # Verify the caller is permitted to use this agent (admins bypass the check)
    is_admin: Final = user_api_key_dict is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )
    if not is_admin or agent.identity_managed:
        is_allowed: Final = await AgentRequestHandler.is_agent_allowed(
            agent_id=agent.agent_id,
            user_api_key_auth=user_api_key_dict,
        )
        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_name}' is not allowed for your key/team. Contact proxy admin for access.",
            )

    litellm_params: Final = agent.litellm_params or {}

    if litellm_params.get("custom_llm_provider") in A2A_CHAT_COMPLETION_BRIDGE_PROVIDERS:
        bridge_auth: Final = await resolve_databricks_backend_auth(litellm_params)
        return _call_route(route_type, _bridge_agent_request(data, agent, bridge_auth.litellm_params))

    # Get API base URL from agent config
    if not agent.agent_card_params or "url" not in agent.agent_card_params:
        verbose_proxy_logger.error("[A2A] Agent '%s' has no URL configured", agent_name)
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name, retryable_with_model_read_through=False)

    # Inject API base and route to litellm
    data["api_base"] = agent.agent_card_params["url"]
    verbose_proxy_logger.debug("[A2A] Routing %s to %s", model_name, data["api_base"])

    return _call_route(route_type, data)


def _bridge_agent_request(
    data: Mapping[str, object],
    agent: AgentResponse,
    litellm_params: Mapping[str, object],
) -> Mapping[str, object]:
    card_url: Final[object] = (agent.agent_card_params or {}).get("url")
    api_base: Final = {"api_base": card_url} if card_url else {}
    request: Final = {key: value for key, value in data.items() if key not in _CLIENT_TARGET_KWARGS}
    return _bridge_request_data({**request, **api_base}, litellm_params, agent.static_headers)


def _call_route(route_type: str, request_data: Mapping[str, object]) -> object:
    return getattr(litellm, route_type)(**request_data)
