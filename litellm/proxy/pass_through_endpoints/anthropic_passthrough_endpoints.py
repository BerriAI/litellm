"""/anthropic must be matched ahead of the native /{provider}/v1/files and
/{provider}/v1/batches routes, so it is registered at startup and defers to the
lazily loaded handler per call."""

from typing import Final

from fastapi import APIRouter, Depends, Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter()


@router.api_route(
    "/anthropic/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Anthropic Pass-through", "pass-through"],
)
async def anthropic_passthrough_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Response:
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/anthropic_completion)
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import anthropic_proxy_route

    return await anthropic_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
    )
