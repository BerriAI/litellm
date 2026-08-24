"""
A2A Agent Routing

Handles routing for A2A agents (models with "a2a/<agent-name>" prefix).
Looks up agents in the registry and injects their API base URL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from fastapi import HTTPException
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.interactions.agents.utils import merge_agent_headers
from litellm.llms.a2a.common_utils import A2AError, convert_messages_to_prompt, extract_text_from_a2a_response
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_OBJECT_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_HEADERS_ADAPTER: Final = TypeAdapter(dict[str, str])
_MESSAGES_ADAPTER: Final = TypeAdapter(list[AllMessageValues])
_FORWARDED_REQUEST_PARAMS: Final = frozenset(
    {
        "audio",
        "frequency_penalty",
        "functions",
        "function_call",
        "include_server_side_tool_invocations",
        "logit_bias",
        "logprobs",
        "guardrails",
        "max_completion_tokens",
        "max_tokens",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "seed",
        "service_tier",
        "stop",
        "store",
        "temperature",
        "thinking",
        "timeout",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
    }
)


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
    api_base: str | None,
    litellm_params: Mapping[str, object],
    static_headers: Mapping[str, str] | None,
    dynamic_headers: Mapping[str, str] | None = None,
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
    provider_params: Final = {
        **_OBJECT_DICT_ADAPTER.validate_python(litellm_params),
        **{
            key: data[key]
            for key in _FORWARDED_REQUEST_PARAMS
            if key in data and data[key] is not None
        },
    }
    bridge_params: Final = _OBJECT_DICT_ADAPTER.validate_python(params)
    configured_headers: Final = litellm_params.get("extra_headers") or litellm_params.get("headers")
    configured_headers_dict: Final = (
        _HEADERS_ADAPTER.validate_python(configured_headers) if isinstance(configured_headers, dict) else None
    )
    agent_extra_headers: Final = merge_agent_headers(
        dynamic_headers=merge_agent_headers(
            dynamic_headers=dynamic_headers,
            static_headers=configured_headers_dict,
        ),
        static_headers=static_headers,
    )
    if agent_extra_headers:
        provider_params["extra_headers"] = agent_extra_headers

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
    usage: Final = response.get("usage")
    if usage is not None:
        setattr(model_response, "usage", usage)

    logging_obj: Final = data.get("litellm_logging_obj")
    if isinstance(logging_obj, Logging):

        def _enqueue_logging() -> None:
            asyncio.create_task(
                logging_obj.dispatch_success_handlers(
                    model_response,
                    cache_hit=False,
                    prefer_async_handlers=True,
                )
            )

        logging_obj._enqueue_deferred_logging = _enqueue_logging

    return model_response


def _merge_agent_guardrails(
    data: Mapping[str, object],
    agent_guardrails: object,
) -> Mapping[str, object]:
    if not agent_guardrails:
        return data

    configured_guardrails: list[object] = (
        agent_guardrails if isinstance(agent_guardrails, list) else [agent_guardrails]
    )
    metadata = data.get("metadata")
    metadata_guardrails = metadata.get("guardrails") if isinstance(metadata, dict) else None
    root_guardrails = data.get("guardrails")
    existing_guardrails: list[object] = []
    for value in (metadata_guardrails, root_guardrails):
        if isinstance(value, list):
            existing_guardrails.extend(value)
        elif value:
            existing_guardrails.append(value)

    merged_guardrails = existing_guardrails + [
        guardrail for guardrail in configured_guardrails if guardrail not in existing_guardrails
    ]
    if isinstance(data, dict):
        data["guardrails"] = merged_guardrails
        if isinstance(metadata, dict):
            metadata["guardrails"] = merged_guardrails
        return data

    merged_data = dict(data)
    merged_data["guardrails"] = merged_guardrails
    if isinstance(metadata, dict):
        merged_data["metadata"] = {**metadata, "guardrails": merged_guardrails}
    return merged_data


def _get_agent_dynamic_headers(
    data: Mapping[str, object],
    agent_id: str,
    agent_name: str,
    extra_headers: list[str] | None,
) -> dict[str, str]:
    proxy_request: Final = data.get("proxy_server_request")
    raw_headers: object = proxy_request.get("headers") if isinstance(proxy_request, Mapping) else None
    if not isinstance(raw_headers, Mapping):
        metadata: Final = data.get("metadata")
        raw_headers = metadata.get("headers") if isinstance(metadata, Mapping) else None
    normalized_headers: Final = (
        {str(key).lower(): str(value) for key, value in raw_headers.items()}
        if isinstance(raw_headers, Mapping)
        else {}
    )

    dynamic_headers: dict[str, str] = {}
    for header_name in extra_headers or []:
        header_name_str: Final = str(header_name)
        value: Final = normalized_headers.get(header_name_str.lower())
        if value is not None:
            dynamic_headers[header_name_str] = value

    for alias in (agent_id.lower(), agent_name.lower()):
        prefix: Final = f"x-a2a-{alias}-"
        for key, value in normalized_headers.items():
            if key.startswith(prefix):
                header_name: Final = key[len(prefix) :]
                if header_name:
                    dynamic_headers[header_name] = value
    return dynamic_headers


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
    registered_params_value: Final = agent.litellm_params
    registered_provider_value: Final = (
        registered_params_value.get("custom_llm_provider") if registered_params_value else None
    )
    registered_provider: Final = registered_provider_value if isinstance(registered_provider_value, str) else None
    configured_api_base: Final = registered_params_value.get("api_base") if registered_params_value else None
    api_base: Final = configured_api_base if isinstance(configured_api_base, str) else agent_url
    registered_model: Final = registered_params_value.get("model") if registered_params_value else None
    cardless_provider: Final = (
        registered_provider == "bedrock" and isinstance(registered_model, str) and "agentcore" in registered_model
    )
    if (not isinstance(agent_url, str) or not agent_url) and not cardless_provider:
        verbose_proxy_logger.error("[A2A] Agent '%s' has no URL configured", agent_name)
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name, retryable_with_model_read_through=False)

    routed_data: Final = _merge_agent_guardrails(
        data=data,
        agent_guardrails=registered_params_value.get("guardrails") if registered_params_value else None,
    )
    registered_dynamic_headers: Final = _get_agent_dynamic_headers(
        data=routed_data,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        extra_headers=agent.extra_headers,
    )
    if (
        registered_provider
        and registered_provider != "a2a"
        and route_type == "acompletion"
        and registered_params_value is not None
    ):
        verbose_proxy_logger.debug("[A2A] Routing %s through %s", model_name, registered_provider)
        return _route_registered_provider(
            data=routed_data,
            model_name=model_name,
            api_base=api_base,
            litellm_params=registered_params_value,
            static_headers=agent.static_headers,
            dynamic_headers=registered_dynamic_headers,
        )

    completion_data: Final = MappingProxyType({**routed_data, "api_base": api_base})
    verbose_proxy_logger.debug("[A2A] Routing %s to %s", model_name, api_base)
    return getattr(litellm, f"{route_type}")(**completion_data)  # pyright: ignore[reportAny]  # dynamic SDK route
