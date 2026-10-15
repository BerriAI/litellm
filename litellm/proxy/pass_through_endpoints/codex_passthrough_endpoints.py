from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import TypeAdapter, ValidationError
from starlette.responses import JSONResponse

from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.proxy._types import SpecialHeaders, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import caller_headers_without_litellm_secrets
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request

router: Final = APIRouter()
_NEVER_FORWARDED: Final = frozenset(
    {"host", "content-length", "content-encoding", "accept-encoding", "cookie", "connection", "transfer-encoding"}
) | (SpecialHeaders.litellm_credential_header_names() - {"authorization"})
_BODY: Final = TypeAdapter(dict[str, object])


def require_codex_credentials(request: Request) -> None:
    from litellm.proxy.proxy_server import general_settings_view

    custom_header: Final = general_settings_view().get("litellm_key_header_name")
    gateway_key: Final = request.headers.get("x-litellm-api-key") or (
        request.headers.get(custom_header) if isinstance(custom_header, str) else None
    )
    if not gateway_key:
        raise HTTPException(status_code=401, detail="A separate LiteLLM gateway credential is required")
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="A ChatGPT subscription bearer token is required")


async def codex_gateway_auth(
    request: Request,
    _: Annotated[None, Depends(require_codex_credentials)],
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    return user


async def codex_response_lifetime() -> AsyncIterator[AsyncExitStack]:
    async with AsyncExitStack() as owner:
        yield owner


async def _relay_codex(
    request: Request, user: UserAPIKeyAuth, endpoint: Literal["models", "responses"], owner: AsyncExitStack
) -> Response:
    headers: Final = caller_headers_without_litellm_secrets(request, user, _NEVER_FORWARDED)
    if "authorization" not in headers:
        raise HTTPException(status_code=401, detail="The ChatGPT credential must differ from the gateway credential")
    raw: Final = await request.body()
    try:
        body: Final = _BODY.validate_json(raw) if raw else {}
    except ValidationError:
        return JSONResponse({"error": "A JSON object is required"}, status_code=400)
    return await pass_through_request(
        request=request,
        target=f"{CHATGPT_API_BASE}/{endpoint}",
        custom_headers=dict(headers),
        user_api_key_dict=user,
        custom_body=body,
        stream=body.get("stream") is True,
        custom_llm_provider="openai",
        follow_redirects=False,
        use_custom_headers_only=True,
        response_owner=owner,
    )


@router.get(
    "/backend-api/codex/models",
    tags=["ChatGPT subscription Pass-through", "pass-through"],
    description="Forward the signed-in user's native Codex catalog using a separate LiteLLM gateway credential.",
)
async def codex_models(
    request: Request,
    user: Annotated[UserAPIKeyAuth, Depends(codex_gateway_auth)],
    owner: Annotated[AsyncExitStack, Depends(codex_response_lifetime, scope="request")],
) -> Response:
    return await _relay_codex(request, user, "models", owner)


@router.post(
    "/backend-api/codex/responses",
    tags=["ChatGPT subscription Pass-through", "pass-through"],
    description="Relay ChatGPT subscription Responses over HTTP/SSE. The client owns login and token refresh.",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"type": "object", "additionalProperties": True}}},
        }
    },
    responses={
        200: {
            "description": "The upstream Responses payload or unchanged server-sent events",
            "content": {
                "application/json": {"schema": {"type": "object", "additionalProperties": True}},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        }
    },
)
async def codex_responses(
    request: Request,
    user: Annotated[UserAPIKeyAuth, Depends(codex_gateway_auth)],
    owner: Annotated[AsyncExitStack, Depends(codex_response_lifetime, scope="request")],
) -> Response:
    return await _relay_codex(request, user, "responses", owner)
