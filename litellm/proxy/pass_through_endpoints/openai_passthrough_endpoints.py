"""/openai_passthrough must be matched ahead of the native /{provider}/v1/files and
/{provider}/v1/batches routes, so unlike the other provider passthrough routes it is
registered at startup and defers to the lazily loaded handler per call."""

from typing import Final

from fastapi import APIRouter, Depends, Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter()


@router.api_route(
    "/openai_passthrough/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["OpenAI Pass-through", "pass-through"],
)
async def openai_passthrough_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Response:
    """
    Dedicated pass-through to the OpenAI API with no overlap with LiteLLM's native
    implementations (e.g. the Responses API at /v1/responses).

    Examples:
        - /openai_passthrough/v1/responses
        - /openai_passthrough/v1/responses/{response_id}
        - /openai_passthrough/v1/responses/{response_id}/input_items

    [Docs](https://docs.litellm.ai/docs/pass_through/openai_passthrough)
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import openai_proxy_route

    return await openai_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
    )
