"""LiteAsk uses the signed-in administrator's existing gateway permissions."""

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pydantic import InstanceOf, JsonValue, TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth

from .agent import chat
from .approval import ApprovalError, NonceStore, claim_approval, open_approval, seal_approval
from .auth import assert_fresh_admin, fresh_admin
from .catalog import ArgumentsError, CatalogError, Tool, build_catalog, tool_request
from .dispatch import DispatchResult, credential_fingerprint, credential_secrets, dispatch
from .models import LiteAskApprovalRequest, LiteAskChatRequest, LiteAskConfig, LiteAskProposal, LiteAskResponse
from .redaction import sanitize

router: Final = APIRouter(
    prefix="/management/v1/liteask",
    tags=["LiteAsk"],  # mutable-ok: FastAPI's tags contract requires a list
)
Admin: TypeAlias = Annotated[UserAPIKeyAuth, Depends(fresh_admin)]
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_APP: Final[TypeAdapter[FastAPI]] = TypeAdapter(InstanceOf[FastAPI])


def _model() -> str | None:
    return os.getenv("LITELLM_LITEASK_MODEL", "").strip() or None


def _store() -> NonceStore | None:
    from litellm.proxy.proxy_server import redis_usage_cache, user_api_key_cache

    candidate: Final = redis_usage_cache or user_api_key_cache.redis_cache
    return candidate if isinstance(candidate, NonceStore) else None


def _catalog(request: Request) -> tuple[Tool, ...]:
    try:
        app: Final = _APP.validate_python(request.scope.get("app"))
        schema: Final = _JSON.validate_python(app.openapi())
    except ValidationError as error:
        raise HTTPException(503, "LiteAsk could not load the gateway's operation catalog.") from error
    catalog: Final = build_catalog(schema)
    if isinstance(catalog, CatalogError):
        raise HTTPException(503, "LiteAsk could not load the gateway's operation catalog.")
    return catalog


def _required_model() -> str:
    model: Final = _model()
    if model is None:
        raise HTTPException(404, "LiteAsk is not enabled on this gateway.")
    return model


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"  # rebind-ok: FastAPI response header boundary


async def _send(
    request: Request,
    caller: UserAPIKeyAuth,
    method: str,
    path: str,
    body: JsonValue,
    query: tuple[tuple[str, str], ...],
) -> DispatchResult:
    # Only the fixed catalog and the fixed inference path call this owner.
    await assert_fresh_admin(caller)
    return await dispatch(request, method, path, body, query, allowed_routes=frozenset({(method, path)}))


@router.get("/config", response_model=LiteAskConfig)
async def liteask_config(response: Response, caller: Admin) -> LiteAskConfig:
    _no_store(response)
    model: Final = _model()
    return LiteAskConfig(enabled=model is not None, model=model, can_execute_mutations=_store() is not None)


@router.post("/chat", response_model=LiteAskResponse)
async def liteask_chat(
    body: LiteAskChatRequest, request: Request, response: Response, caller: Admin
) -> LiteAskResponse:
    _no_store(response)
    model: Final = _required_model()
    if body.messages[-1].role != "user" or sum(len(item.content) for item in body.messages) > 48000:
        raise HTTPException(422, "End with a user message and keep the conversation under 48,000 characters.")
    fingerprint: Final = credential_fingerprint(request)
    if not fingerprint or not caller.user_id:
        raise HTTPException(403, "Sign in with your own gateway administrator account.")
    tools: Final = _catalog(request)

    async def send(method: str, path: str, payload: JsonValue, query: tuple[tuple[str, str], ...]) -> DispatchResult:
        return await _send(request, caller, method, path, payload, query)

    async def propose(tool: Tool, arguments: Mapping[str, JsonValue]) -> LiteAskResponse:
        if _store() is None:
            return LiteAskResponse(
                message="This gateway needs a shared Redis connection before LiteAsk can make changes."
            )
        sealed: Final = await seal_approval(
            user_id=caller.user_id or "",
            credential=fingerprint,
            conversation_id=str(body.conversation_id),
            tool=tool.operation.name,
            arguments=arguments,
            now=int(time.time()),
            store=_store(),
        )
        if isinstance(sealed, ApprovalError):
            raise HTTPException(sealed.status_code, sealed.message)
        token, expires_at = sealed
        return LiteAskResponse(
            message="Review this change before approving it. Nothing has changed yet.",
            proposal=LiteAskProposal(
                token=token,
                tool=tool.operation.name,
                title=tool.operation.title,
                arguments=arguments,
                expires_at=expires_at,
            ),
        )

    try:
        return await asyncio.wait_for(
            chat(
                model=model,
                messages=body.messages,
                tools=tools,
                send=send,
                propose=propose,
                secrets=credential_secrets(request),
            ),
            timeout=90,
        )
    except asyncio.TimeoutError as error:
        raise HTTPException(504, "LiteAsk took too long. Try a narrower request. No changes were made.") from error


@router.post("/approve", response_model=LiteAskResponse)
async def liteask_approve(
    body: LiteAskApprovalRequest, request: Request, response: Response, caller: Admin
) -> LiteAskResponse:
    _no_store(response)
    _required_model()
    fingerprint: Final = credential_fingerprint(request)
    if not fingerprint or not caller.user_id:
        raise HTTPException(403, "Sign in with your own gateway administrator account.")
    approval: Final = open_approval(
        body.token,
        user_id=caller.user_id,
        credential=fingerprint,
        conversation_id=str(body.conversation_id),
        now=int(time.time()),
    )
    if isinstance(approval, ApprovalError):
        raise HTTPException(approval.status_code, approval.message)
    tool: Final = next((item for item in _catalog(request) if item.operation.name == approval.tool), None)
    if tool is None or not tool.operation.mutation:
        raise HTTPException(400, "This operation is no longer available. Prepare the change again.")
    prepared: Final = tool_request(tool, _JSON.validate_python(approval.arguments))
    if isinstance(prepared, ArgumentsError):
        raise HTTPException(400, "The operation changed. Prepare the change again.")
    await assert_fresh_admin(caller)
    claimed: Final = await claim_approval(approval, store=_store(), now=int(time.time()))
    if isinstance(claimed, ApprovalError):
        raise HTTPException(claimed.status_code, claimed.message)
    try:
        result: Final = await asyncio.wait_for(
            _send(request, caller, tool.operation.method, prepared.path, prepared.body, prepared.query),
            timeout=60,
        )
    except asyncio.TimeoutError as error:
        raise HTTPException(
            504, "The result is uncertain. Check the gateway before preparing another change."
        ) from error
    key: Final = (
        result.data.get("key")
        if tool.operation.name in ("key_create", "user_create") and isinstance(result.data, dict)
        else None
    )
    generated_key: Final = key if isinstance(key, str) and result.status_code < 400 else None
    safe_result: Final = sanitize(
        result.data,
        (*credential_secrets(request), *((generated_key,) if generated_key else ())),
        response_operation=tool.operation.name if result.status_code < 400 else "",
    )
    if result.status_code >= 400:
        return LiteAskResponse(
            message=f"The gateway returned HTTP {result.status_code}. Check the current state before trying again.",
            result=safe_result,
        )
    return LiteAskResponse(
        message=f"Completed: {tool.operation.title}.",
        result=safe_result,
        generated_key=generated_key,
    )
