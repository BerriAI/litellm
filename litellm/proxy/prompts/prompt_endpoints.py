"""
CRUD ENDPOINTS FOR PROMPTS
"""

from typing import List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.prompts.init_prompts import ListPromptsResponse, PromptSpec

router = APIRouter()


@router.get(
    "/prompt/list",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListPromptsResponse,
)
async def list_prompts(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List of available prompts for a given key.
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    # check key metadata for prompts
    key_metadata = user_api_key_dict.metadata
    if key_metadata is not None:
        prompts = cast(Optional[List[str]], key_metadata.get("prompts", None))
        if prompts is not None:
            return ListPromptsResponse(
                prompts=[
                    IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS[prompt]
                    for prompt in prompts
                    if prompt in IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS
                ]
            )
    # check if user is proxy admin - show all prompts
    if user_api_key_dict.user_role is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return ListPromptsResponse(
            prompts=list(IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.values())
        )
    else:
        return ListPromptsResponse(prompts=[])


@router.get(
    "/prompt/info",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptSpec,
)
async def get_prompt(
    prompt_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get info about a prompt
    """
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    ## CHECK IF USER HAS ACCESS TO PROMPT
    prompts: Optional[List[str]] = None
    if user_api_key_dict.metadata is not None:
        prompts = cast(
            Optional[List[str]], user_api_key_dict.metadata.get("prompts", None)
        )
        if prompts is not None and prompt_id not in prompts:
            raise HTTPException(status_code=400, detail=f"Prompt {prompt_id} not found")
    if user_api_key_dict.user_role is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        pass
    else:
        raise HTTPException(
            status_code=403,
            detail=f"You are not authorized to access this prompt. Your role - {user_api_key_dict.user_role}, Your key's prompts - {prompts}",
        )

    prompt_spec = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(prompt_id)
    if prompt_spec is None:
        raise HTTPException(status_code=400, detail=f"Prompt {prompt_id} not found")
    return prompt_spec
