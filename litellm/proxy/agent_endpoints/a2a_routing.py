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
from litellm.types.utils import Choices, CustomPricingLiteLLMParams, Message, ModelResponse

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
        "guided_json",
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
        "safety_identifier",
        "stop",
        "store",
        "stream_options",
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
        "output_config",
        "prompt_cache_key",
    }
)
_A2A_PRICING_PARAMS: Final = frozenset({"cost_per_query", "response_cost"}) | frozenset(
    CustomPricingLiteLLMParams.model_fields
)


def _get_agent_request_headers(data: Mapping[str, object]) -> dict[str, str]:
    proxy_request: Final = data.get("proxy_server_request")
    raw_headers: object = proxy_request.get("headers") if isinstance(proxy_request, Mapping) else None
    if not isinstance(raw_headers, Mapping):
        metadata: Final = data.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = data.get("litellm_metadata")
        raw_headers = metadata.get("headers") if isinstance(metadata, Mapping) else None
    return (
        {str(key).lower(): str(value) for key, value in raw_headers.items()} if isinstance(raw_headers, Mapping) else {}
    )


class _A2ATextPart(TypedDict):
    kind: ReadOnly[str]
    text: ReadOnly[str]


class _A2AMessage(TypedDict):
    role: ReadOnly[str]
    parts: ReadOnly[tuple[_A2ATextPart, ...]]
    messageId: ReadOnly[str]
    contextId: ReadOnly[str | None]


class _A2AParams(TypedDict):
    message: ReadOnly[_A2AMessage]
    messages: ReadOnly[list[AllMessageValues]]


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
    raw_session_id: Final = data.get("litellm_session_id")
    metadata: Final = data.get("metadata")
    session_id: Final = (
        raw_session_id
        if isinstance(raw_session_id, str)
        else metadata.get("session_id")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("session_id"), str)
        else None
    )
    params: Final[_A2AParams] = {
        "message": {
            "role": "user",
            "parts": ({"kind": "text", "text": convert_messages_to_prompt(messages)},),
            "messageId": str(uuid4()),
            "contextId": session_id,
        },
        "messages": messages,
    }
    provider_params: Final = {
        **_OBJECT_DICT_ADAPTER.validate_python(litellm_params),
        **{key: data[key] for key in _FORWARDED_REQUEST_PARAMS if key in data and data[key] is not None},
    }
    registered_provider: Final = litellm_params.get("custom_llm_provider")
    registered_model: Final = litellm_params.get("model")
    native_provider: Final = registered_provider == "pydantic_ai_agents" or (
        registered_provider == "bedrock" and isinstance(registered_model, str) and "agentcore" in registered_model
    )
    bridge_params: Final = _OBJECT_DICT_ADAPTER.validate_python(
        {"message": params["message"]} if native_provider else params
    )
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

    logging_obj: Final = data.get("litellm_logging_obj")
    if isinstance(logging_obj, Logging):
        provider_params["no-log"] = True
        provider_model: Final = litellm_params.get("model")
        if isinstance(provider_model, str):
            logging_obj.model_call_details["model"] = provider_model
            logging_obj.model_call_details.setdefault("litellm_params", {})["model"] = provider_model
        provider_name: Final = litellm_params.get("custom_llm_provider")
        if isinstance(provider_name, str):
            logging_obj.model_call_details["custom_llm_provider"] = provider_name
        pricing_params = {
            key: litellm_params[key]
            for key in _A2A_PRICING_PARAMS
            if key in litellm_params and litellm_params[key] is not None
        }
        if pricing_params:
            logging_obj.litellm_params.update(pricing_params)
            logging_obj.model_call_details["litellm_params"].update(pricing_params)
            logging_obj.custom_pricing = True

    if stream:
        streaming_response: Final = A2ACompletionBridgeHandler.handle_streaming(
            request_id=request_id,
            params=bridge_params,
            litellm_params=provider_params,
            api_base=api_base,
            agent_extra_headers=agent_extra_headers,
            agent_static_headers=static_headers,
        )
        completion_stream: Final = A2AModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=False,
            model=model_name,
        )
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
        agent_static_headers=static_headers,
    )
    error_value: Final = response.get("error")
    if isinstance(error_value, dict):
        error: Final = _OBJECT_DICT_ADAPTER.validate_python(error_value)
        error_message: Final = error.get("message")
        raise A2AError(
            status_code=500,
            message=f"A2A error: {error_message if isinstance(error_message, str) else 'Unknown error'}",
        )

    result: Final = response.get("result")
    result_dict: Final = result if isinstance(result, Mapping) else {}
    nested_message: Final = result_dict.get("message")
    response_message: Final = nested_message if isinstance(nested_message, Mapping) else result_dict
    response_choices: Final = response.get("choices")
    choice_payloads: Final = response_choices if isinstance(response_choices, list) else result_dict.get("choices")

    def _serialize_value(value: object) -> object:
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if hasattr(value, "dict"):
            return value.dict(exclude_none=True)
        return value

    def _build_message(message_payload: Mapping[str, object], content: str) -> Message:
        message_kwargs: dict[str, object] = {
            "content": content,
            "role": "assistant",
        }
        raw_tool_calls = message_payload.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            message_kwargs["tool_calls"] = raw_tool_calls
        for field in (
            "audio",
            "annotations",
            "function_call",
            "images",
            "provider_specific_fields",
            "reasoning_content",
            "reasoning_items",
            "thinking_blocks",
        ):
            value = message_payload.get(field)
            if value is not None:
                message_kwargs[field] = _serialize_value(value)
        return Message(**message_kwargs)

    if isinstance(choice_payloads, list) and choice_payloads:
        model_choices = []
        for choice_index, choice in enumerate(choice_payloads):
            choice_mapping: Mapping[str, object] = choice if isinstance(choice, Mapping) else {}
            raw_message = choice_mapping.get("message")
            message_payload: Mapping[str, object] = raw_message if isinstance(raw_message, Mapping) else choice_mapping
            choice_kwargs: dict[str, object] = {
                "finish_reason": (
                    choice_mapping.get("finish_reason")
                    if isinstance(choice_mapping.get("finish_reason"), str)
                    else message_payload.get("finish_reason")
                    if isinstance(message_payload.get("finish_reason"), str)
                    else "stop"
                ),
                "index": choice_mapping.get("index", choice_index)
                if isinstance(choice_mapping.get("index", choice_index), int)
                else choice_index,
                "message": _build_message(
                    message_payload,
                    extract_text_from_a2a_response({"result": message_payload}),
                ),
            }
            raw_logprobs = choice_mapping.get("logprobs", message_payload.get("logprobs"))
            if raw_logprobs is not None:
                choice_kwargs["logprobs"] = _serialize_value(raw_logprobs)
            model_choices.append(Choices(**choice_kwargs))
    else:
        tool_calls: Final = response_message.get("tool_calls")
        normalized_tool_calls: Final = tool_calls if isinstance(tool_calls, list) else None
        finish_reason: Final = response_message.get("finish_reason")
        text: Final = extract_text_from_a2a_response(response)
        choice_kwargs = {
            "finish_reason": (
                finish_reason if isinstance(finish_reason, str) else "tool_calls" if normalized_tool_calls else "stop"
            ),
            "index": 0,
            "message": _build_message(response_message, text),
        }
        raw_logprobs = response_message.get("logprobs")
        if raw_logprobs is not None:
            choice_kwargs["logprobs"] = _serialize_value(raw_logprobs)
        model_choices = [Choices(**choice_kwargs)]
    model_response: Final = ModelResponse(
        id=str(response.get("id") or request_id),
        model=model_name,
        choices=model_choices,
        system_fingerprint=response.get("system_fingerprint")
        if isinstance(response.get("system_fingerprint"), str)
        else None,
        service_tier=response.get("service_tier") if isinstance(response.get("service_tier"), str) else None,
    )
    raw_usage: Final = response.get("usage")
    usage: Final = litellm.Usage(**raw_usage) if isinstance(raw_usage, Mapping) else raw_usage
    if usage is not None:
        model_response.usage = usage
        if isinstance(logging_obj, Logging):
            logging_obj.model_call_details["usage"] = usage

    if isinstance(logging_obj, Logging):

        def _enqueue_logging(final_response: ModelResponse | None = None) -> None:
            asyncio.create_task(
                logging_obj.dispatch_success_handlers(
                    final_response if final_response is not None else model_response,
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

    configured_guardrails: list[object] = agent_guardrails if isinstance(agent_guardrails, list) else [agent_guardrails]
    metadata_key: Final = "litellm_metadata" if "litellm_metadata" in data else "metadata"
    metadata = data.get(metadata_key)
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
            data[metadata_key] = {**metadata, "guardrails": merged_guardrails}
        return data

    merged_data = dict(data)
    merged_data["guardrails"] = merged_guardrails
    if isinstance(metadata, dict):
        merged_data[metadata_key] = {**metadata, "guardrails": merged_guardrails}
    return merged_data


async def merge_a2a_agent_guardrails_before_hooks(data: Mapping[str, object]) -> Mapping[str, object]:
    model_name: Final = data.get("model")
    if not isinstance(model_name, str) or not model_name.startswith("a2a/"):
        return data

    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    agent = await get_agent_with_read_through(model_name[4:])
    if agent is None or not agent.litellm_params:
        return data
    return _merge_agent_guardrails(data, agent.litellm_params.get("guardrails"))


async def authorize_a2a_agent_before_hooks(
    data: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth | None,
) -> Mapping[str, object]:
    model_name: Final = data.get("model")
    if not isinstance(model_name, str) or not model_name.startswith("a2a/"):
        return data

    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import AgentRequestHandler
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    agent = await get_agent_with_read_through(model_name[4:])
    if agent is None:
        return data

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
                detail=f"Agent '{agent.agent_name}' is not allowed for your key/team. Contact proxy admin for access.",
            )

    if (agent.litellm_params or {}).get("require_trace_id_on_calls_to_agent"):
        _enforce_inbound_trace_id(data, agent.agent_id)

    if isinstance(data, dict):
        data["agent_id"] = agent.agent_id
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            data["metadata"] = metadata
        metadata["agent_id"] = agent.agent_id
    return data


def _get_agent_dynamic_headers(
    data: Mapping[str, object],
    agent_id: str,
    agent_name: str,
    extra_headers: list[str] | None,
) -> dict[str, str]:
    normalized_headers: Final = _get_agent_request_headers(data)

    dynamic_headers: dict[str, str] = {}
    for header_name in extra_headers or []:
        header_name_str: Final = str(header_name)
        if header_name_str.lower().startswith("x-litellm-"):
            continue
        value: Final = normalized_headers.get(header_name_str.lower())
        if value is not None:
            dynamic_headers[header_name_str] = value

    for alias in (agent_id.lower(), agent_name.lower()):
        prefix: Final = f"x-a2a-{alias}-"
        for key, value in normalized_headers.items():
            if key.startswith(prefix):
                header_name: Final = key[len(prefix) :]
                if header_name and not header_name.lower().startswith("x-litellm-"):
                    dynamic_headers[header_name] = value
    return dynamic_headers


def _get_agent_identity_headers(
    user_api_key_dict: UserAPIKeyAuth | None,
    trace_id: object | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if user_api_key_dict is not None and user_api_key_dict.user_id:
        headers["X-LiteLLM-User-Id"] = user_api_key_dict.user_id
    if user_api_key_dict is not None and user_api_key_dict.team_id:
        headers["X-LiteLLM-Team-Id"] = user_api_key_dict.team_id
    if trace_id:
        headers["X-LiteLLM-Trace-Id"] = str(trace_id)
    return headers


def _enforce_inbound_trace_id(data: Mapping[str, object], agent_id: str) -> None:
    from litellm.proxy.litellm_pre_call_utils import get_chain_id_from_headers

    if not get_chain_id_from_headers(_get_agent_request_headers(data)):
        raise HTTPException(
            status_code=400,
            detail=f"Agent '{agent_id}' requires x-litellm-trace-id header on all inbound requests.",
        )


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

    if (agent.litellm_params or {}).get("require_trace_id_on_calls_to_agent"):
        _enforce_inbound_trace_id(data, agent.agent_id)

    # Get API base URL from agent config
    agent_card_params: Final = agent.agent_card_params
    agent_url: Final = agent_card_params.get("url") if agent_card_params else None
    registered_params_value: Final = agent.litellm_params
    registered_provider_value: Final = (
        registered_params_value.get("custom_llm_provider") if registered_params_value else None
    )
    registered_provider: Final = registered_provider_value if isinstance(registered_provider_value, str) else None
    from litellm.a2a_protocol.litellm_completion_bridge.handler import A2A_USER_API_KEY_HASH_PARAM

    registered_params_for_route: Final[Mapping[str, object]] = (
        {
            **registered_params_value,
            A2A_USER_API_KEY_HASH_PARAM: user_api_key_dict.api_key,
        }
        if registered_params_value and user_api_key_dict is not None and user_api_key_dict.api_key
        else registered_params_value or {}
    )
    configured_api_base: Final = registered_params_value.get("api_base") if registered_params_value else None
    api_base: Final = configured_api_base if isinstance(configured_api_base, str) and configured_api_base else agent_url
    cardless_provider: Final = registered_provider is not None and registered_provider != "a2a"
    has_configured_api_base: Final = isinstance(configured_api_base, str) and bool(configured_api_base)
    if (not isinstance(agent_url, str) or not agent_url) and not has_configured_api_base and not cardless_provider:
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
    registered_static_headers: Mapping[str, str] | None = agent.static_headers
    if registered_params_value and registered_params_value.get("databricks_oauth"):
        from litellm.proxy.agent_endpoints.databricks_oauth import resolve_databricks_app_auth_header

        databricks_headers = await resolve_databricks_app_auth_header(dict(registered_params_value))
        registered_static_headers = merge_agent_headers(
            dynamic_headers=registered_static_headers,
            static_headers=databricks_headers,
        )
    registered_static_headers = merge_agent_headers(
        dynamic_headers=registered_static_headers,
        static_headers=_get_agent_identity_headers(user_api_key_dict, data.get("litellm_trace_id")),
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
            litellm_params=registered_params_for_route,
            static_headers=registered_static_headers,
            dynamic_headers=registered_dynamic_headers,
        )

    completion_data: Final = MappingProxyType({**routed_data, "api_base": api_base})
    verbose_proxy_logger.debug("[A2A] Routing %s to %s", model_name, api_base)
    return getattr(litellm, f"{route_type}")(**completion_data)  # pyright: ignore[reportAny]  # dynamic SDK route
