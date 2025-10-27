"""
What is this? 

CRUD endpoints for managing pass-through endpoints
"""

from fastapi import APIRouter, Depends, Request, Response

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get(
    "/config/pass_through_endpoints/settings",
    dependencies=[Depends(user_api_key_auth)],
    tags=["pass-through-endpoints"],
    summary="Create pass-through endpoints for provider specific endpoints - https://docs.litellm.ai/docs/proxy/pass_through",
)
async def create_fine_tuning_job(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    pass
