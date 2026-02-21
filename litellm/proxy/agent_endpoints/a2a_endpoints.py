"""
A2A Protocol endpoints for LiteLLM Proxy.

Allows clients to invoke agents through LiteLLM using the A2A protocol.
The A2A SDK can point to LiteLLM's URL and invoke agents registered with LiteLLM.
"""

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import all_litellm_params

router = APIRouter()


def _jsonrpc_error(
    request_id: Optional[str],
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


async def _handle_stream_message(
    api_base: Optional[str],
    request_id: str,
    params: dict,
    litellm_params: Optional[dict] = None,
    agent_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    proxy_server_request: Optional[dict] = None,
    *,
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
            yield json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Server error: 'a2a' package not installed",
                    },
                }
            ) + "\n"

        return StreamingResponse(_error_stream(), media_type="application/x-ndjson")

    from a2a.types import MessageSendParams, SendStreamingMessageRequest

    use_proxy_hooks = (
        user_api_key_dict is not None
        and request_data is not None
        and proxy_logging_obj is not None
    )

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
            )

            if use_proxy_hooks and user_api_key_dict is not None and request_data is not None and proxy_logging_obj is not None:
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
                    return json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": getattr(
                                    proxy_exc, "message", f"Streaming error: {proxy_exc!s}"
                                ),
                            },
                        }
                    ) + "\n"

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
                        yield json.dumps(
                            chunk.model_dump(mode="json", exclude_none=True)
                        ) + "\n"
                    else:
                        yield json.dumps(chunk) + "\n"
        except Exception as e:
            verbose_proxy_logger.exception(f"Error streaming A2A response: {e}")
            if use_proxy_hooks and proxy_logging_obj is not None and user_api_key_dict is not None and request_data is not None:
                transformed_exception = await proxy_logging_obj.post_call_failure_hook(
                    user_api_key_dict=user_api_key_dict,
                    original_exception=e,
                    request_data=request_data,
                )
                if transformed_exception is not None:
                    e = transformed_exception
            if isinstance(e, HTTPException):
                raise
            yield json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": f"Streaming error: {str(e)}"},
                }
            ) + "\n"

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

        # Copy and rewrite URL to point to LiteLLM proxy
        agent_card = dict(agent.agent_card_params)
        agent_card["url"] = f"{str(request.base_url).rstrip('/')}/a2a/{agent_id}"

        verbose_proxy_logger.debug(
            f"Returning agent card for '{agent_id}' with proxy URL: {agent_card['url']}"
        )
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
    from litellm.a2a_protocol import asend_message
    from litellm.a2a_protocol.main import A2A_SDK_AVAILABLE
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
        AgentRequestHandler,
    )
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    body = {}
    try:
        body = await request.json()

        verbose_proxy_logger.debug(f"A2A request for agent '{agent_id}': {body}")

        # Validate JSON-RPC format
        if body.get("jsonrpc") != "2.0":
            return _jsonrpc_error(
                body.get("id"), -32600, "Invalid Request: jsonrpc must be '2.0'"
            )

        request_id = body.get("id")
        method = body.get("method")
        params = body.get("params", {})

        if params:
            # extract any litellm params from the params - eg. 'guardrails'
            params_to_remove = []
            for key, value in params.items():
                if key in all_litellm_params:
                    params_to_remove.append(key)
                    body[key] = value
            for key in params_to_remove:
                params.pop(key)

        if not A2A_SDK_AVAILABLE:
            return _jsonrpc_error(
                request_id,
                -32603,
                "Server error: 'a2a' package not installed. Please install 'a2a-sdk'.",
                500,
            )

        # Find the agent
        agent = _get_agent(agent_id)
        if agent is None:
            return _jsonrpc_error(
                request_id, -32000, f"Agent '{agent_id}' not found", 404
            )

        is_allowed = await AgentRequestHandler.is_agent_allowed(
            agent_id=agent.agent_id,
            user_api_key_auth=user_api_key_dict,
        )
        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_id}' is not allowed for your key/team. Contact proxy admin for access.",
            )

        # Get backend URL and agent name
        agent_url = agent.agent_card_params.get("url")
        agent_name = agent.agent_card_params.get("name", agent_id)

        # Get litellm_params (may include custom_llm_provider for completion bridge)
        litellm_params = agent.litellm_params or {}
        custom_llm_provider = litellm_params.get("custom_llm_provider")

        # URL is required unless using completion bridge with a provider that derives endpoint from model
        # (e.g., bedrock/agentcore derives endpoint from ARN in model string)
        if not agent_url and not custom_llm_provider:
            return _jsonrpc_error(
                request_id, -32000, f"Agent '{agent_id}' has no URL configured", 500
            )

        verbose_proxy_logger.info(
            f"Proxying A2A request to agent '{agent_id}' at {agent_url or 'completion-bridge'}"
        )

        # Set up data dict for litellm processing
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

        # Route through SDK functions
        if method == "message/send":
            from a2a.types import MessageSendParams, SendMessageRequest

            a2a_request = SendMessageRequest(
                id=request_id,
                params=MessageSendParams(**params),
            )
            response = await asend_message(
                request=a2a_request,
                api_base=agent_url,
                litellm_params=litellm_params,
                agent_id=agent.agent_id,
                metadata=data.get("metadata", {}),
                proxy_server_request=data.get("proxy_server_request"),
            )

            response = await proxy_logging_obj.post_call_success_hook(
                user_api_key_dict=user_api_key_dict,
                data=data,
                response=response,
            )
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
                request_id=request_id,
                params=params,
                litellm_params=litellm_params,
                agent_id=agent.agent_id,
                metadata=data.get("metadata", {}),
                proxy_server_request=data.get("proxy_server_request"),
                user_api_key_dict=user_api_key_dict,
                request_data=data,
                proxy_logging_obj=proxy_logging_obj,
            )
        else:
            return _jsonrpc_error(request_id, -32601, f"Method '{method}' not found")

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error invoking agent: {e}")
        return _jsonrpc_error(body.get("id"), -32603, f"Internal error: {str(e)}", 500)
