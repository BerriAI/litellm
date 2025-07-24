from typing import List

from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)
from litellm.types.proxy.public_endpoints.public_endpoints import PublicModelHubInfo

router = APIRouter()


@router.get(
    "/public/model_hub",
    tags=["public", "model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[ModelGroupInfoProxy],
)
async def public_model_hub():
    import litellm
    from litellm.proxy.proxy_server import _get_model_group_info, llm_router

    if llm_router is None:
        raise HTTPException(
            status_code=400, detail=CommonProxyErrors.no_llm_router.value
        )

    model_groups: List[ModelGroupInfoProxy] = []
    if litellm.public_model_groups is not None:
        model_groups = _get_model_group_info(
            llm_router=llm_router,
            all_models_str=litellm.public_model_groups,
            model_group=None,
        )

    return model_groups


@router.get(
    "/public/model_hub/info",
    tags=["public", "model management"],
    response_model=PublicModelHubInfo,
)
async def public_model_hub_info():
    import litellm
    from litellm.proxy.proxy_server import _title, version

    try:
        from litellm_enterprise.proxy.proxy_server import EnterpriseProxyConfig

        custom_docs_description = EnterpriseProxyConfig.get_custom_docs_description()
    except Exception:
        custom_docs_description = None

    return PublicModelHubInfo(
        docs_title=_title,
        custom_docs_description=custom_docs_description,
        litellm_version=version,
        useful_links=litellm.public_model_groups_links,
    )
