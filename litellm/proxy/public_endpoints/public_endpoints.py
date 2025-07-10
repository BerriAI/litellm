from typing import List

from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.router import ModelGroupInfo

router = APIRouter()


@router.get(
    "/public/model_hub",
    tags=["public", "model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[ModelGroupInfo],
)
async def public_model_hub():
    import litellm
    from litellm.proxy.proxy_server import _get_model_group_info, llm_router

    if llm_router is None:
        raise HTTPException(
            status_code=400, detail=CommonProxyErrors.no_llm_router.value
        )

    model_groups: List[ModelGroupInfo] = []
    if litellm.public_model_groups is not None:
        model_groups = _get_model_group_info(
            llm_router=llm_router,
            all_models_str=litellm.public_model_groups,
            model_group=None,
        )

    return model_groups
