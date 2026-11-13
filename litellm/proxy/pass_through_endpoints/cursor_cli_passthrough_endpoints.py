from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import ConfigDict, TypeAdapter

from litellm.constants import CURSOR_CLI_PASS_THROUGH_ROUTE_PREFIX
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import check_opaque_inference_controls
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import caller_headers_without_litellm_secrets
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,  # pyright: ignore[reportUnknownVariableType]  # legacy relay returns an untyped response
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY
from litellm.types.proxy.auth.special_headers import SpecialHeaders

router: Final = APIRouter()
_METADATA: Final = TypeAdapter(Mapping[str, object])
_RESPONSE: Final = TypeAdapter(Response, config=ConfigDict(arbitrary_types_allowed=True))
_NEVER_FORWARDED: Final = frozenset(
    {
        "host",
        "content-length",
        "accept-encoding",
        "cookie",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
) | (SpecialHeaders.litellm_credential_header_names() - {"authorization"})


async def cursor_response_lifetime() -> AsyncIterator[AsyncExitStack]:
    async with AsyncExitStack() as owner:
        yield owner


@router.post(
    f"{CURSOR_CLI_PASS_THROUGH_ROUTE_PREFIX}/{{endpoint:path}}",
    tags=["Cursor CLI Pass-through", "pass-through"],
    description=(
        "Forward the signed-in Cursor CLI's agent session using a separate x-litellm-api-key. "
        "Set network.useHttp1ForAgent to true in the CLI's cli-config.json, then run "
        '`agent --agent-endpoint https://YOUR_PROXY/cursor-cli --header "x-litellm-api-key: Bearer YOUR_KEY"`. '
        "Leave --endpoint unchanged. Login, bootstrap and model discovery stay direct to Cursor. "
        "Use an unrestricted virtual key or master key. JWT and custom-auth credentials are unsupported. "
        "Keys bound to LiteLLM agents are unsupported. "
        "Model restrictions, token accounting and content guardrails are unsupported. "
        "Configured moderation, content guardrails or policy-engine rules disable this relay, "
        "even when not selected by this request. "
        "Request limits count RPCs, not agent sessions. Allow overlapping stream and append calls. "
        "Logs describe HTTP transport status, not the agent's outcome. The client owns login and token refresh."
    ),
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/connect+proto": {"schema": {"type": "string", "format": "binary"}},
                "application/proto": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
    responses={
        200: {
            "description": "Unchanged native Cursor response bytes, including streamed agent events",
            "content": {
                "application/proto": {"schema": {"type": "string", "format": "binary"}},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        }
    },
)
async def cursor_cli_passthrough(
    endpoint: Literal["agent.v1.AgentService/RunSSE", "aiserver.v1.BidiService/BidiAppend"],
    request: Request,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    owner: Annotated[AsyncExitStack, Depends(cursor_response_lifetime, scope="request")],
) -> Response:
    if not user.via_virtual_key or user.jwt_claims is not None:
        raise HTTPException(status_code=403, detail="Cursor passthrough requires a LiteLLM virtual key or master key")
    if not request.headers.get("x-litellm-api-key"):
        raise HTTPException(status_code=401, detail="A separate LiteLLM gateway credential is required")
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="A Cursor bearer token is required")
    check_opaque_inference_controls(user.model_dump())
    connection_headers: Final = frozenset(
        name.strip().lower() for name in request.headers.get("connection", "").split(",")
    )
    filtered: Final = caller_headers_without_litellm_secrets(request, user, _NEVER_FORWARDED | connection_headers)
    headers: Final = {name: value for name, value in filtered.items() if not name.startswith("x-pass-")}
    if "authorization" not in headers:
        raise HTTPException(status_code=401, detail="Cursor and LiteLLM credentials must be separate")
    state: Final = _METADATA.validate_python(request.scope.get("state", {}))
    relay_request: Final = Request(
        {
            **request.scope,
            "headers": [
                (name.encode("latin-1"), value.encode("latin-1"))
                for name, value in headers.items()
                if name != "authorization"
            ],
            "state": {
                **{name: value for name, value in state.items() if name != "_cached_headers"},
                LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY: await request.body(),
            },
        },
        request.receive,
    )
    return _RESPONSE.validate_python(
        await pass_through_request(
            request=relay_request,
            target=f"https://api2.cursor.sh/{endpoint}",
            custom_headers=headers,
            user_api_key_dict=user,
            custom_llm_provider="cursor_cli",
            upstream_headers_filter=lambda _: headers,
            raw_response=True,
            response_owner=owner,
        )
    )
