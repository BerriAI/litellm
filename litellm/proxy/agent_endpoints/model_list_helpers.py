"""
Helper functions for appending A2A agents to model lists.

Used by proxy model endpoints to make agents appear in UI alongside models.
"""
from typing import List

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)


async def append_agents_to_model_group(
    model_groups: List[ModelGroupInfoProxy],
    user_api_key_dict: UserAPIKeyAuth,
) -> List[ModelGroupInfoProxy]:
    """
    Append A2A agents to model groups list for UI display.
    
    Converts agents to model format with "a2a/<agent-name>" naming
    so they appear in playground and work with LiteLLM routing.
    """
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
            AgentRequestHandler,
        )

        allowed_agent_ids = await AgentRequestHandler.get_allowed_agents(
            user_api_key_auth=user_api_key_dict
        )
        
        for agent_id in allowed_agent_ids:
            agent = global_agent_registry.get_agent_by_id(agent_id)
            if agent is not None:
                model_groups.append(
                    ModelGroupInfoProxy(
                        model_group=f"a2a/{agent.agent_name}",
                        mode="chat",
                        providers=["a2a"],
                    )
                )
    except Exception as e:
        verbose_proxy_logger.debug(
            f"Error appending agents to model_group/info: {e}"
        )
    
    return model_groups


async def append_agents_to_model_info(
    models: List[dict],
    user_api_key_dict: UserAPIKeyAuth,
) -> List[dict]:
    """
    Append A2A agents to model info list for UI display.
    
    Converts agents to model format with "a2a/<agent-name>" naming
    so they appear in models page and work with LiteLLM routing.
    """
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
            AgentRequestHandler,
        )

        allowed_agent_ids = await AgentRequestHandler.get_allowed_agents(
            user_api_key_auth=user_api_key_dict
        )
        
        for agent_id in allowed_agent_ids:
            agent = global_agent_registry.get_agent_by_id(agent_id)
            if agent is not None:
                models.append({
                    "model_name": f"a2a/{agent.agent_name}",
                    "litellm_params": {
                        "model": f"a2a/{agent.agent_name}",
                        "custom_llm_provider": "a2a",
                    },
                    "model_info": {
                        "id": agent.agent_id,
                        "mode": "chat",
                        "db_model": True,
                        "created_by": agent.created_by,
                        "created_at": agent.created_at,
                        "updated_at": agent.updated_at,
                    },
                })
    except Exception as e:
        verbose_proxy_logger.debug(
            f"Error appending agents to v2/model/info: {e}"
        )
    
    return models
