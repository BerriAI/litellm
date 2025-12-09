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
    a2a_client: Any,
    request_id: str,
    params: dict,
) -> StreamingResponse:
    """Handle message/stream method."""
    from a2a.types import MessageSendParams, SendStreamingMessageRequest

    a2a_request = SendStreamingMessageRequest(
        id=request_id,
        params=MessageSendParams(**params),
    )

    async def stream_response():
        try:
            async for chunk in a2a_client.send_message_streaming(a2a_request):
                yield json.dumps(chunk.model_dump(mode="json", exclude_none=True)) + "\n"
        except Exception as e:
            verbose_proxy_logger.exception(f"Error streaming A2A response: {e}")
            yield json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Streaming error: {str(e)}"},
            }) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@router.get(
    "/a2a/{agent_id}/.well-known/agent-card.json",
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
    from litellm.a2a_protocol import asend_message, create_a2a_client
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
        AgentRequestHandler,
    )
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_config,
        version,
    )

    body = {}
    try:
        body = await request.json()
        verbose_proxy_logger.debug(f"A2A request for agent '{agent_id}': {body}")

        # Validate JSON-RPC format
        if body.get("jsonrpc") != "2.0":
            return _jsonrpc_error(body.get("id"), -32600, "Invalid Request: jsonrpc must be '2.0'")

        request_id = body.get("id")
        method = body.get("method")
        params = body.get("params", {})

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

        # Get backend URL and agent name
        agent_url = agent.agent_card_params.get("url")
        agent_name = agent.agent_card_params.get("name", agent_id)
        if not agent_url:
            return _jsonrpc_error(request_id, -32000, f"Agent '{agent_id}' has no URL configured", 500)

        verbose_proxy_logger.info(f"Proxying A2A request to agent '{agent_id}' at {agent_url}")

        # Set up data dict for litellm processing
        body.update({
            "model": f"a2a_agent/{agent_name}",
            "custom_llm_provider": "a2a_agent",
        })

        # Add litellm data (user_api_key, user_id, team_id, etc.)
        data = await add_litellm_data_to_request(
            data=body,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
            general_settings=general_settings,
            version=version,
        )

        # Create A2A client
        a2a_client = await create_a2a_client(base_url=agent_url)

        if method == "message/send":
            from a2a.types import MessageSendParams, SendMessageRequest

            a2a_request = SendMessageRequest(
                id=request_id,
                params=MessageSendParams(**params),
            )

            # Pass litellm data through kwargs for proper logging
            response = await asend_message(
                a2a_client=a2a_client,
                request=a2a_request,
                metadata=data.get("metadata", {}),
                proxy_server_request=data.get("proxy_server_request"),
            )
            return JSONResponse(content=response.model_dump(mode="json", exclude_none=True))

        elif method == "message/stream":
            return await _handle_stream_message(a2a_client, request_id, params)
        else:
            return _jsonrpc_error(request_id, -32601, f"Method '{method}' not found")

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error invoking agent: {e}")
        return _jsonrpc_error(body.get("id"), -32603, f"Internal error: {str(e)}", 500)
