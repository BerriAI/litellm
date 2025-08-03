"""
CRUD ENDPOINTS FOR PROMPTS
"""

from typing import List, Optional, cast

from fastapi import APIRouter, Depends

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.prompts.init_prompts import ListPromptsResponse

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
