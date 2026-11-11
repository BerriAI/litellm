import hmac
from types import MappingProxyType
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request

router: Final = APIRouter()
_HEADERS: Final = frozenset(
    {
        "authorization",
        "chatgpt-account-id",
        "content-type",
        "accept",
        "user-agent",
        "originator",
        "openai-beta",
        "session-id",
        "thread-id",
        "session_id",
        "conversation_id",
        "x-codex-turn-metadata",
        "x-codex-turn-state",
    }
)


@router.get("/backend-api/codex/{endpoint}", tags=["ChatGPT Pass-through", "pass-through"])
@router.post(
    "/backend-api/codex/{endpoint}",
    tags=["ChatGPT Pass-through", "pass-through"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"type": "object", "additionalProperties": True}}},
        }
    },
    responses={
        200: {
            "description": "Unmodified Codex response or server-sent events",
            "content": {
                "application/json": {"schema": {"type": "object", "additionalProperties": True}},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        }
    },
)
async def codex_passthrough_route(
    endpoint: Literal["models", "responses"],
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Forward a client's ChatGPT subscription with separate LiteLLM authentication."""
    expected_method: Final = "GET" if endpoint == "models" else "POST"
    if request.method != expected_method:
        raise HTTPException(status_code=405, detail="Method not allowed", headers={"Allow": expected_method})

    authorization: Final = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    gateway_key: Final = request.headers.get("x-litellm-api-key", "")
    gateway_parts: Final = gateway_key.split()
    gateway_token: Final = (
        gateway_parts[-1]
        if len(gateway_parts) == 1 or (len(gateway_parts) == 2 and gateway_parts[0].lower() == "bearer")
        else ""
    )
    if scheme.lower() != "bearer" or not token.strip() or not gateway_token:
        raise HTTPException(
            status_code=401,
            detail="Send the ChatGPT bearer in Authorization and the LiteLLM key in x-litellm-api-key",
        )
    if hmac.compare_digest(token.strip().encode(), gateway_token.encode()):
        raise HTTPException(status_code=401, detail="ChatGPT and LiteLLM credentials must be separate")

    headers: Final = MappingProxyType({name: value for name, value in request.headers.items() if name in _HEADERS})
    return await pass_through_request(
        request=request,
        target=f"{CHATGPT_API_BASE}/{endpoint}",
        custom_headers=dict(headers),
        user_api_key_dict=user_api_key_dict,
        custom_llm_provider="chatgpt",
        upstream_headers_filter=lambda _: headers,
    )
