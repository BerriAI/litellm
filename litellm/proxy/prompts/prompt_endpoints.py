"""
CRUD ENDPOINTS FOR PROMPTS
"""

import inspect
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.prompts.init_prompts import ListPromptsResponse, PromptSpec

router = APIRouter()


def _get_prompts_list_response(
    prompts_config: List[PromptSpec],
) -> ListPromptsResponse:
    """
    Helper function to get the guardrails list response
    """
    prompt_configs: List[PromptSpec] = []
    for prompt in prompts_config:
        # validate prompt
        if not isinstance(prompt, dict):
            verbose_proxy_logger.info(
                f"Prompt must be a dictionary, got {type(prompt)}, skipping..."
            )
            continue

        # validate prompt_id
        if not prompt.get("prompt_id"):
            verbose_proxy_logger.info(f"Prompt ID is required, skipping... {prompt}")
            continue

        # validate litellm_params
        if not prompt.get("litellm_params"):
            verbose_proxy_logger.info(
                f"Litellm params are required, skipping... {prompt}"
            )
            continue

        # validate prompt_info
        if not prompt.get("prompt_info"):
            verbose_proxy_logger.info(f"Prompt info is required, skipping... {prompt}")
            continue

        prompt_configs.append(
            PromptSpec(
                prompt_id=prompt.get("prompt_id"),
                litellm_params=prompt.get("litellm_params"),
                prompt_info=prompt.get("prompt_info"),
            )
        )
    return ListPromptsResponse(prompts=prompt_configs)


@router.get(
    "/prompt/list",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_prompts():
    """
    List of available prompts for a given key.
    """
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    return list(IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.values())
