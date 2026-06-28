"""
A2A Protocol endpoints for LiteLLM Proxy.

Allows clients to invoke agents through LiteLLM using the A2A protocol.
The A2A SDK can point to LiteLLM's URL and invoke agents registered with LiteLLM.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.databricks_oauth import (
    DATABRICKS_OAUTH_PARAM,
    resolve_databricks_app_auth_header,
)
from litellm.proxy.agent_endpoints.utils import merge_agent_headers
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import all_litellm_params

router = APIRouter()

_PASCAL_TO_WIRE: Dict[str, str] = {
    "GetTask": "tasks/get",
    "ListTasks": "tasks/list",
    "CancelTask": "tasks/cancel",
    "SubscribeToTask": "tasks/resubscribe",
    "CreateTaskPushNotificationConfig": "tasks/pushNotificationConfig/set",
    "GetTaskPushNotificationConfig": "tasks/pushNotificationConfig/get",
    "ListTaskPushNotificationConfigs": "tasks/pushNotificationConfig/list",
    "DeleteTaskPushNotificationConfig": "tasks/pushNotificationConfig/delete",
    "GetExtendedAgentCard": "agent/getAuthenticatedExtendedCard",
}


def _validate_push_notification_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise HTTPException(
            status_code=400,
            detail="Push notification URL must use HTTPS",
        )
    try:
        validate_url(url)
    except (SSRFError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _caller_identity_headers(user_api_key_dict: UserAPIKeyAuth) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if user_api_key_dict.user_id:
        headers["X-LiteLLM-User-Id"] = user_api_key_dict.user_id
    if user_api_key_dict.team_id:
        headers["X-LiteLLM-Team-Id"] = user_api_key_dict.team_id
    return headers


def _forwarding_headers(
    user_api_key_dict: UserAPIKeyAuth,
    request_data: dict,
    agent_extra_headers: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    sanitized = (
        {k: v for k, v in agent_extra_headers.items() if not k.lower().startswith("x-litellm-")}
        if agent_extra_headers
        else None
    )
    merged = merge_agent_headers(dynamic_headers=sanitized, static_headers=None) or {}
    identity = _caller_identity_headers(user_api_key_dict)
    trace_id = request_data.get("litellm_trace_id")
    if trace_id:
        identity["X-LiteLLM-Trace-Id"] = str(trace_id)
    merged.update(identity)
    return merged or None


def _jsonrpc_error(
    request_id: Optional[Any],
    code: int,
    message: str,
    status_code: int = 400,
) -> JSONResponse:
    """Create a JSON-RPC 2.0 error response."""
    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        },
        status_code=status_code,
    )


def _get_agent(agent_id: str):
    """Look up an agent by ID or name. Returns None if not found."""
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    agent = global_agent_registry.get_agent_by_id(agent_id=agent_id)
    if agent is None:
        agent = global_agent_registry.get_agent_by_name(agent_name=agent_id)
    return agent


def _enforce_inbound_trace_id(agent: Any, request: Request) -> None:
    """Raise 400 if agent requires x-litellm-trace-id on inbound calls and it is missing."""
    agent_litellm_params = agent.litellm_params or {}
    if not agent_litellm_params.get("require_trace_id_on_calls_to_agent"):
        return

    from litellm.proxy.litellm_pre_call_utils import get_chain_id_from_headers

    headers_dict = dict(request.headers)
    trace_id = get_chain_id_from_headers(headers_dict)
    if not trace_id:
        raise HTTPException(
            status_code=400,
            detail=(f"Agent '{agent.agent_id}' requires x-litellm-trace-id header on all inbound requests."),
        )


async def _forward_jsonrpc(
    agent_url: str,
    body: dict,
    extra_headers: Optional[Dict[str, str]] = None,
) -> dict:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    headers = {"Content-Type": "application/json", **(extra_headers or {})}
    handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2A,
        params={"timeout": 60.0},
    )
    resp = await handler.post(agent_url, json=body, headers=headers)
    try:
        result = resp.json()
    except Exception:
        resp.raise_for_status()
        raise
    if not resp.is_success and "error" not in result:
        resp.raise_for_status()
    return result


async def _a2a_sse_event_source(
    agent_url: str,
    body: dict,
    request_id: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[dict, None]:
    """Stream an upstream A2A SSE response as parsed JSON-RPC event dicts.

    Upstream HTTP/JSON-RPC errors are surfaced as a single JSON-RPC error event
    so the caller can relay them instead of breaking the stream.
    """
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.agents import _normalize_a2a_jsonrpc_response
    from litellm.types.llms.custom_http import httpxSpecialProvider

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **(extra_headers or {}),
    }
    handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2A,
        params={"timeout": None},
    )
    async_client = handler.client
    req = async_client.build_request("POST", agent_url, json=body, headers=headers)
    resp = await async_client.send(req, stream=True)
    try:
        if not resp.is_success:
            error_body = await resp.aread()
            error_event: Optional[dict] = None
            try:
                parsed = json.loads(error_body)
                if isinstance(parsed, dict) and "error" in parsed:
                    error_event = _normalize_a2a_jsonrpc_response(parsed, request_id=request_id)
            except Exception:
                error_event = None
            yield error_event or {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": resp.reason_phrase},
            }
            return
        async for line in resp.aiter_lines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            payload = stripped[len("data:") :].strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except Exception:
                continue
    finally:
        await resp.aclose()


async def _forward_jsonrpc_sse(
    agent_url: str,
    body: dict,
    request_id: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    proxy_logging_obj: Optional[Any] = None,
    user_api_key_dict: Optional[Any] = None,
    request_data: Optional[dict] = None,
) -> StreamingResponse:
    event_source = _a2a_sse_event_source(agent_url, body, request_id=request_id, extra_headers=extra_headers)

    def _serialize_chunk(chunk: Any) -> str:
        return f"data: {json.dumps(chunk)}\n\n"

    def _serialize_error(proxy_exc: Any) -> str:
        return (
            "data: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": getattr(proxy_exc, "message", str(proxy_exc)),
                    },
                }
            )
            + "\n\n"
        )

    if proxy_logging_obj is not None and user_api_key_dict is not None and request_data is not None:
        # Route streamed events through the shared streaming generator so the
        # post-call streaming hook (and therefore agent guardrails) inspects
        # tasks/resubscribe output the same way message/stream does.
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        generator: AsyncGenerator[str, None] = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=event_source,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
            proxy_logging_obj=proxy_logging_obj,
            serialize_chunk=_serialize_chunk,
            serialize_error=_serialize_error,
        )
    else:

        async def _passthrough() -> AsyncGenerator[str, None]:
            async for chunk in event_source:
                yield _serialize_chunk(chunk)

        generator = _passthrough()

    return StreamingResponse(generator, media_type="text/event-stream")


async def _handle_stream_message(
    api_base: Optional[str],
    request_id: Any,
    params: dict,
    litellm_params: Optional[dict] = None,
    agent_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    proxy_server_request: Optional[dict] = None,
    *,
    agent_extra_headers: Optional[Dict[str, str]] = None,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    request_data: Optional[dict] = None,
    proxy_logging_obj: Optional[Any] = None,
) -> StreamingResponse:
    """Handle message/stream method via SDK functions.

    When user_api_key_dict, request_data, and proxy_logging_obj are provided,
    uses common_request_processing.async_streaming_data_generator with NDJSON
    serializers so proxy hooks and cost injection apply.
    """
    from litellm.a2a_protocol import asend_message_streaming
    from litellm.a2a_protocol.main import A2A_SDK_AVAILABLE

    if not A2A_SDK_AVAILABLE:

        async def _error_stream():
            yield (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": "Server error: 'a2a' package not installed",
                        },
                    }
                )
                + "\n"
            )

        return StreamingResponse(_error_stream(), media_type="application/x-ndjson")

    from a2a.types import MessageSendParams, SendStreamingMessageRequest

    use_proxy_hooks = user_api_key_dict is not None and request_data is not None and proxy_logging_obj is not None

    async def stream_response():
        try:
            a2a_request = SendStreamingMessageRequest(
                id=request_id,
                params=MessageSendParams(**params),
            )
            a2a_stream = asend_message_streaming(
                request=a2a_request,
                api_base=api_base,
                litellm_params=litellm_params,
                agent_id=agent_id,
                metadata=metadata,
                proxy_server_request=proxy_server_request,
                agent_extra_headers=agent_extra_headers,
            )

            if (
                use_proxy_hooks
                and user_api_key_dict is not None
                and request_data is not None
                and proxy_logging_obj is not None
            ):
                from litellm.proxy.common_request_processing import (
                    ProxyBaseLLMRequestProcessing,
                )

                def _ndjson_chunk(chunk: Any) -> str:
                    if hasattr(chunk, "model_dump"):
                        obj = chunk.model_dump(mode="json", exclude_none=True)
                    else:
                        obj = chunk
                    return json.dumps(obj) + "\n"

                def _ndjson_error(proxy_exc: Any) -> str:
                    return (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32603,
                                    "message": getattr(
                                        proxy_exc,
                                        "message",
                                        f"Streaming error: {proxy_exc!s}",
                                    ),
                                },
                            }
                        )
                        + "\n"
                    )

                async for line in ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
                    response=a2a_stream,
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                    proxy_logging_obj=proxy_logging_obj,
                    serialize_chunk=_ndjson_chunk,
                    serialize_error=_ndjson_error,
                ):
                    yield line
            else:
                async for chunk in a2a_stream:
                    if hasattr(chunk, "model_dump"):
                        yield (json.dumps(chunk.model_dump(mode="json", exclude_none=True)) + "\n")
                    else:
                        yield json.dumps(chunk) + "\n"
        except Exception as e:
            verbose_proxy_logger.exception(f"Error streaming A2A response: {e}")
            if (
                use_proxy_hooks
                and proxy_logging_obj is not None
                and user_api_key_dict is not None
                and request_data is not None
            ):
                transformed_exception = await proxy_logging_obj.post_call_failure_hook(
                    user_api_key_dict=user_api_key_dict,
                    original_exception=e,
                    request_data=request_data,
                )
                if transformed_exception is not None:
                    e = transformed_exception
            if isinstance(e, HTTPException):
                raise
            yield (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Streaming error: {str(e)}",
                        },
                    }
                )
                + "\n"
            )

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@router.get(
    "/a2a/{agent_id}/.well-known/agent-card.json",
    tags=["[beta] A2A Agents"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/a2a/{agent_id}/.well-known/agent.json",
    tags=["[beta] A2A Agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_agent_card(
    agent_id: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get the agent card for an agent (A2A discovery endpoint).

    Supports both standard paths:
    - /.well-known/agent-card.json
    - /.well-known/agent.json

    The URL in the agent card is rewritten to point to the LiteLLM proxy,
    so all subsequent A2A calls go through LiteLLM for logging and cost tracking.
    """
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
        AgentRequestHandler,
    )

    try:
        agent = _get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        # Check agent permission (skip for admin users)
        is_allowed = await AgentRequestHandler.is_agent_allowed(
            agent_id=agent.agent_id,
            user_api_key_auth=user_api_key_dict,
        )
        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_id}' is not allowed for your key/team. Contact proxy admin for access.",
            )

        if not agent.agent_card_params:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_id}' has no agent card configured",
            )

        # Copy and rewrite URL to point to LiteLLM proxy
        agent_card = {
            **agent.agent_card_params,
            "url": f"{str(request.base_url).rstrip('/')}/a2a/{agent_id}",
        }

        verbose_proxy_logger.debug(f"Returning agent card for '{agent_id}' with proxy URL: {agent_card['url']}")
        return JSONResponse(content=agent_card)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting agent card: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/a2a/{agent_id}",
    tags=["[beta] A2A Agents"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/a2a/{agent_id}/message/send",
    tags=["[beta] A2A Agents"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/v1/a2a/{agent_id}/message/send",
    tags=["[beta] A2A Agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def invoke_agent_a2a(
    agent_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Invoke an agent using the A2A protocol (JSON-RPC 2.0).

    Supported methods:
    - message/send: Send a message and get a response
    - message/stream: Send a message and stream the response
    """
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
        AgentRequestHandler,
    )
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    body: Dict[str, Any] = {}
    request_data: Dict[str, Any] = body
    try:
        body = await request.json()
        request_data = body

        verbose_proxy_logger.debug(f"A2A request for agent '{agent_id}': {body}")

        # Validate JSON-RPC format
        if body.get("jsonrpc") != "2.0":
            return _jsonrpc_error(body.get("id"), -32600, "Invalid Request: jsonrpc must be '2.0'")

        request_id: Optional[Any] = body.get("id")
        method: Optional[str] = body.get("method")
        params = body.get("params", {})

        if method:
            method = _PASCAL_TO_WIRE.get(method, method)

        if isinstance(params, dict):
            # extract any litellm params from the params - eg. 'guardrails'
            # ``metadata`` is intentionally excluded: it's a first-class A2A
            # ``MessageSendParams`` field that the completion bridge forwards
            # downstream via ``get_forward_metadata``. Stripping it here would
            # collide with litellm's spend-tracking ``metadata`` kwarg and
            # silently drop the caller's A2A request-level metadata.
            params_to_remove = []
            for key, value in params.items():
                if key in all_litellm_params and key not in {"id", "metadata"}:
                    params_to_remove.append(key)
                    body[key] = value
            for key in params_to_remove:
                params.pop(key)

        # Find the agent
        agent = _get_agent(agent_id)
        if agent is None:
            return _jsonrpc_error(request_id, -32000, f"Agent '{agent_id}' not found", 404)

        is_allowed = await AgentRequestHandler.is_agent_allowed(
            agent_id=agent.agent_id,
            user_api_key_auth=user_api_key_dict,
        )
        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_id}' is not allowed for your key/team. Contact proxy admin for access.",
            )

        _enforce_inbound_trace_id(agent, request)

        # Get backend URL and agent name
        agent_card_params = agent.agent_card_params or {}
        agent_url = agent_card_params.get("url")
        agent_name = agent_card_params.get("name", agent_id)

        # Get litellm_params (may include custom_llm_provider for completion bridge)
        litellm_params = agent.litellm_params or {}
        custom_llm_provider = litellm_params.get("custom_llm_provider")

        # Hand the authenticated key hash to the completion bridge so provider
        # configs can scope provider-side session state per key (e.g. LangFlow
        # session memory) instead of trusting the client-supplied A2A contextId.
        if custom_llm_provider and user_api_key_dict.api_key:
            from litellm.a2a_protocol.litellm_completion_bridge.handler import (
                A2A_USER_API_KEY_HASH_PARAM,
            )

            litellm_params = {
                **litellm_params,
                A2A_USER_API_KEY_HASH_PARAM: user_api_key_dict.api_key,
            }

        # URL is required unless using completion bridge with a provider that derives endpoint from model
        # (e.g., bedrock/agentcore derives endpoint from ARN in model string)
        if not agent_url and not custom_llm_provider:
            return _jsonrpc_error(request_id, -32000, f"Agent '{agent_id}' has no URL configured", 500)

        verbose_proxy_logger.info(f"Proxying A2A request to agent '{agent_id}' at {agent_url or 'completion-bridge'}")

        # Set up data dict for litellm processing
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["agent_id"] = agent.agent_id
        body["agent_id"] = agent.agent_id

        body.update(
            {
                "model": f"a2a_agent/{agent_name}",
                "custom_llm_provider": "a2a_agent",
            }
        )

        # Add litellm data (user_api_key, user_id, team_id, etc.)
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        processor = ProxyBaseLLMRequestProcessing(data=body)
        data, logging_obj = await processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type="asend_message",
            version=version,
        )
        request_data = data

        # Build merged headers for the backend agent
        static_headers: Dict[str, str] = dict(agent.static_headers or {})

        raw_headers = dict(request.headers)
        normalized = {k.lower(): v for k, v in raw_headers.items()}

        dynamic_headers: Dict[str, str] = {}

        # 1. Admin-configured extra_headers: forward named headers from client request
        if agent.extra_headers:
            for header_name in agent.extra_headers:
                header_name_str = str(header_name)
                val = normalized.get(header_name_str.lower())
                if val is not None:
                    dynamic_headers[header_name_str] = val

        # 2. Convention-based forwarding: x-a2a-{agent_id_or_name}-{header_name}
        #    Matches both agent_id (UUID) and agent_name (alias), case-insensitive.
        for alias in (agent.agent_id.lower(), agent.agent_name.lower()):
            prefix = f"x-a2a-{alias}-"
            for key, val in normalized.items():
                if key.startswith(prefix):
                    header_name = key[len(prefix) :]
                    if header_name:
                        dynamic_headers[header_name] = val

        agent_extra_headers = merge_agent_headers(
            dynamic_headers=dynamic_headers or None,
            static_headers=static_headers or None,
        )

        # Databricks App endpoints require a short-lived OAuth M2M token rather
        # than a static bearer. Only agents explicitly configured with a
        # ``databricks_oauth`` block get one; every other agent is left untouched.
        if litellm_params.get(DATABRICKS_OAUTH_PARAM):
            databricks_auth = await resolve_databricks_app_auth_header(litellm_params)
            if databricks_auth:
                agent_extra_headers = {
                    **(agent_extra_headers or {}),
                    **databricks_auth,
                }

        # Merge agent-level guardrails into data so post_call_success_hook and
        # _handle_stream_message both pick them up.  A2A agents use model
        # a2a_agent/*, which is not an llm_router deployment, so
        # _check_and_merge_model_level_guardrails() skips them.
        _agent_guardrails = litellm_params.get("guardrails")
        if _agent_guardrails:
            if not isinstance(_agent_guardrails, list):
                _agent_guardrails = [_agent_guardrails]
            _existing_guardrails: List = data.get("guardrails") or []
            if not isinstance(_existing_guardrails, list):
                _existing_guardrails = [_existing_guardrails]
            data["guardrails"] = _existing_guardrails + [g for g in _agent_guardrails if g not in _existing_guardrails]

        # Route through SDK functions
        if method == "message/send":
            from litellm.a2a_protocol import asend_message
            from litellm.a2a_protocol.main import A2A_SDK_AVAILABLE

            if not A2A_SDK_AVAILABLE:
                return _jsonrpc_error(
                    request_id,
                    -32603,
                    "Server error: 'a2a' package not installed. Please install 'a2a-sdk'.",
                    500,
                )
            from a2a.types import MessageSendParams, SendMessageRequest

            a2a_request = SendMessageRequest(
                id=request_id if request_id is not None else "",
                params=MessageSendParams(**params),
            )
            # Defer spend-log until after post_call_success_hook so guardrail
            # results written by the unified_guardrail hook are captured.
            logging_obj._defer_async_logging = True  # type: ignore[union-attr]
            response = await asend_message(
                request=a2a_request,
                api_base=agent_url,
                litellm_params=litellm_params,
                agent_id=agent.agent_id,
                metadata=data.get("metadata", {}),
                proxy_server_request=data.get("proxy_server_request"),
                litellm_logging_obj=logging_obj,
                agent_extra_headers=agent_extra_headers,
            )

            try:
                response = await proxy_logging_obj.post_call_success_hook(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    response=response,
                )
            finally:
                _enqueue_fn = getattr(logging_obj, "_enqueue_deferred_logging", None)
                if _enqueue_fn is not None:
                    logging_obj._enqueue_deferred_logging = None  # type: ignore[union-attr]
                    _enqueue_fn()

            return JSONResponse(
                content=(
                    response.model_dump(mode="json", exclude_none=True)  # type: ignore
                    if hasattr(response, "model_dump")
                    else response
                )
            )

        elif method == "message/stream":
            return await _handle_stream_message(
                api_base=agent_url,
                request_id=request_id if request_id is not None else "",
                params=params,
                litellm_params=litellm_params,
                agent_id=agent.agent_id,
                metadata=data.get("metadata", {}),
                proxy_server_request=data.get("proxy_server_request"),
                agent_extra_headers=agent_extra_headers,
                user_api_key_dict=user_api_key_dict,
                request_data=data,
                proxy_logging_obj=proxy_logging_obj,
            )
        elif method in {
            "tasks/get",
            "tasks/list",
            "tasks/cancel",
            "tasks/pushNotificationConfig/set",
            "tasks/pushNotificationConfig/get",
            "tasks/pushNotificationConfig/list",
            "tasks/pushNotificationConfig/delete",
            "agent/getAuthenticatedExtendedCard",
        }:
            if not agent_url:
                return _jsonrpc_error(request_id, -32000, f"Agent '{agent_id}' has no URL configured", 500)
            if method == "tasks/pushNotificationConfig/set":
                if not isinstance(params, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="params must be an object",
                    )
                push_config = params.get("pushNotificationConfig", {})
                if "pushNotificationConfig" in params and not isinstance(push_config, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="pushNotificationConfig must be an object",
                    )
                for callback_url in (params.get("url"), push_config.get("url")):
                    if not callback_url:
                        continue
                    if not isinstance(callback_url, str):
                        raise HTTPException(
                            status_code=400,
                            detail="Push notification URL must be a string",
                        )
                    _validate_push_notification_url(callback_url)
            forward_body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            caller_headers = _forwarding_headers(
                user_api_key_dict=user_api_key_dict,
                request_data=data,
                agent_extra_headers=agent_extra_headers,
            )
            result = await _forward_jsonrpc(agent_url, forward_body, extra_headers=caller_headers)
            if method == "agent/getAuthenticatedExtendedCard":
                if isinstance(result.get("result"), dict) and "url" in result["result"]:
                    result["result"]["url"] = f"{str(request.base_url).rstrip('/')}/a2a/{agent_id}"
            from litellm.types.agents import LiteLLMSendMessageResponse

            response = LiteLLMSendMessageResponse.from_dict(result, request_id=request_id)
            response = await proxy_logging_obj.post_call_success_hook(
                user_api_key_dict=user_api_key_dict,
                data=data,
                response=response,
            )
            return JSONResponse(
                content=(
                    response.model_dump(mode="json", exclude_none=True) if hasattr(response, "model_dump") else response
                )
            )

        elif method == "tasks/resubscribe":
            if not agent_url:
                return _jsonrpc_error(request_id, -32000, f"Agent '{agent_id}' has no URL configured", 500)
            forward_body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            sse_caller_headers = _forwarding_headers(
                user_api_key_dict=user_api_key_dict,
                request_data=data,
                agent_extra_headers=agent_extra_headers,
            )
            return await _forward_jsonrpc_sse(
                agent_url,
                forward_body,
                request_id=request_id,
                extra_headers=sse_caller_headers,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
                request_data=data,
            )

        else:
            return _jsonrpc_error(request_id, -32601, f"Method '{method}' not found")

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error invoking agent: {e}")
        try:
            await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data=request_data,
            )
        except Exception:
            pass
        return _jsonrpc_error(body.get("id"), -32603, f"Internal error: {str(e)}", 500)
