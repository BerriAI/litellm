"""
Agent endpoints for registering + discovering agents via LiteLLM.

Follows the A2A Spec.

1. Register an agent via POST `/v1/agents`
2. Discover agents via GET `/v1/agents`
3. Get specific agent via GET `/v1/agents/{agent_id}`
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_streaming_response,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.types.agents import AgentConfig
from litellm.types.utils import TokenCountResponse

router = APIRouter()


@router.get(
    "/v1/agents",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[AgentConfig],
)
async def get_agents(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # Used for auth
):
    """
    Example usage:
    ```
    curl -X GET "http://localhost:4000/v1/agents" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer your-key" \
    ```

    Returns: List[AgentConfig]
    """
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    try:
        return global_agent_registry.get_agent_list()
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise HTTPException(
            status_code=500, detail={"error": f"Internal server error: {str(e)}"}
        )
