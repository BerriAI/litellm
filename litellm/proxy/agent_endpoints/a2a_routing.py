"""
A2A Agent Routing

Handles routing for A2A agents (models with "a2a/<agent-name>" prefix).
Looks up agents in the registry and injects their API base URL.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from fastapi import HTTPException
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.llms.a2a.common_utils import A2AError, convert_messages_to_prompt, extract_text_from_a2a_response
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_OBJECT_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_HEADERS_ADAPTER: Final = TypeAdapter(dict[str, str])
_MESSAGES_ADAPTER: Final = TypeAdapter(list[AllMessageValues])


class _A2ATextPart(TypedDict):
    kind: ReadOnly[str]
    text: ReadOnly[str]


class _A2AMessage(TypedDict):
    role: ReadOnly[str]
    parts: ReadOnly[tuple[_A2ATextPart, ...]]
    messageId: ReadOnly[str]


class _A2AParams(TypedDict):
    message: ReadOnly[_A2AMessage]


async def _route_registered_provider(
    data: Mapping[str, object],
    model_name: str,
    api_base: str,
    litellm_params: Mapping[str, object],
) -> ModelResponse | CustomStreamWrapper:
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator

    raw_messages: Final = data.get("messages")
    messages: Final = _MESSAGES_ADAPTER.validate_python(raw_messages)
    stream: Final = data.get("stream") is True
    request_id: Final = str(uuid4())
    params: Final[_A2AParams] = {
        "message": {
            "role": "user",
            "parts": ({"kind": "text", "text": convert_messages_to_prompt(messages)},),
            "messageId": str(uuid4()),
        }
    }
    provider_params: Final = _OBJECT_DICT_ADAPTER.validate_python(litellm_params)
    bridge_params: Final = _OBJECT_DICT_ADAPTER.validate_python(params)
    configured_headers: Final = litellm_params.get("extra_headers") or litellm_params.get("headers")
    agent_extra_headers: Final = (
        _HEADERS_ADAPTER.validate_python(configured_headers) if isinstance(configured_headers, dict) else None
    )

    if stream:
        streaming_response: Final = A2ACompletionBridgeHandler.handle_streaming(
            request_id=request_id,
            params=bridge_params,
            litellm_params=provider_params,
            api_base=api_base,
            agent_extra_headers=agent_extra_headers,
        )
        completion_stream: Final = A2AModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=False,
            model=model_name,
        )
        logging_obj: Final = data.get("litellm_logging_obj")
        if not isinstance(logging_obj, Logging):
            raise TypeError("litellm_logging_obj is required for streaming A2A requests")
        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model_name,
            custom_llm_provider="a2a",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options"),
        )

    response: Final = await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id=request_id,
        params=bridge_params,
        litellm_params=provider_params,
        api_base=api_base,
        agent_extra_headers=agent_extra_headers,
    )
    error_value: Final = response.get("error")
    if isinstance(error_value, dict):
        error: Final = _OBJECT_DICT_ADAPTER.validate_python(error_value)
        error_message: Final = error.get("message")
        raise A2AError(
            status_code=500,
            message=f"A2A error: {error_message if isinstance(error_message, str) else 'Unknown error'}",
        )

    text: Final = extract_text_from_a2a_response(response)
    model_response: Final = ModelResponse(
        id=str(response.get("id") or request_id),
        model=model_name,
        choices=[  # mutable-ok: ModelResponse requires a choices list
            Choices(finish_reason="stop", index=0, message=Message(content=text, role="assistant"))
        ],
    )
    return model_response


async def route_a2a_agent_request(
    data: Mapping[str, object],
    route_type: str,
    user_api_key_dict: UserAPIKeyAuth | None = None,
) -> Awaitable[object] | None:
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
    if not is_admin:
        is_allowed: Final = await AgentRequestHandler.is_agent_allowed(
            agent_id=agent.agent_id,
            user_api_key_auth=user_api_key_dict,
        )
        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_name}' is not allowed for your key/team. Contact proxy admin for access.",
            )

    # Get API base URL from agent config
    agent_card_params: Final = agent.agent_card_params
    agent_url: Final = agent_card_params.get("url") if agent_card_params else None
    if not isinstance(agent_url, str) or not agent_url:
        verbose_proxy_logger.error("[A2A] Agent '%s' has no URL configured", agent_name)
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name, retryable_with_model_read_through=False)

    registered_params_value: Final = agent.litellm_params
    registered_provider_value: Final = (
        registered_params_value.get("custom_llm_provider") if registered_params_value else None
    )
    registered_provider: Final = registered_provider_value if isinstance(registered_provider_value, str) else None
    configured_api_base: Final = registered_params_value.get("api_base") if registered_params_value else None
    api_base: Final = configured_api_base if isinstance(configured_api_base, str) else agent_url
    if (
        registered_provider
        and registered_provider != "a2a"
        and route_type == "acompletion"
        and registered_params_value is not None
    ):
        verbose_proxy_logger.debug("[A2A] Routing %s through %s", model_name, registered_provider)
        return _route_registered_provider(
            data=data,
            model_name=model_name,
            api_base=api_base,
            litellm_params=registered_params_value,
        )

    completion_data: Final = MappingProxyType({**data, "api_base": api_base})
    verbose_proxy_logger.debug("[A2A] Routing %s to %s", model_name, api_base)
    return getattr(litellm, f"{route_type}")(**completion_data)  # pyright: ignore[reportAny]  # dynamic SDK route
