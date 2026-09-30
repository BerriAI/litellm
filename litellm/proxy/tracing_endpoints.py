"""
Agent tracing endpoints. Thin wrappers over `TraceReceiver`: auth -> tenant/scope -> one call.

POST /v1/traces                              OTLP/HTTP trace export (protobuf or JSON)
GET  /v1/traces                              TracePage
GET  /v1/traces/{trace_id}                   Trace
GET  /v1/traces/{trace_id}/spans/{span_id}   SpanDetail
"""

import time
from typing import Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from litellm.constants import OTLP_RETRY_AFTER_SECONDS
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.tracing import (
    Tenant,
    TraceReceiver,
    TracingBackpressureError,
    TracingPayloadTooLargeError,
)
from litellm.tracing.decode import encode_otlp_response
from litellm.tracing.types import SpanDetail, Trace, TracePage, TraceScope

router = APIRouter(tags=["agent tracing"])

MS_PER_DAY: Final = 24 * 60 * 60 * 1000
_ADMIN_ROLES: Final = (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

receiver: TraceReceiver | None = None  # set at proxy startup when agent tracing is enabled


def get_receiver() -> TraceReceiver:
    if receiver is None:
        raise HTTPException(
            status_code=501,
            detail="Agent tracing is not enabled. Set `tracing:` in general_settings and CLICKHOUSE_URL.",
        )
    return receiver


def tenant_for(user_api_key_dict: UserAPIKeyAuth) -> Tenant:
    return Tenant(
        team_id=user_api_key_dict.team_id or "",
        api_key_hash=user_api_key_dict.token or "",
        org_id=user_api_key_dict.org_id or "",
    )


def scope_for(user_api_key_dict: UserAPIKeyAuth) -> TraceScope:
    """Admins see everything; team members see their team; team-less keys see their own traces."""
    if user_api_key_dict.user_role in _ADMIN_ROLES:
        return TraceScope(team_ids=[], api_key_hash="")
    if user_api_key_dict.team_id:
        return TraceScope(team_ids=[user_api_key_dict.team_id], api_key_hash="")
    if not user_api_key_dict.token:
        raise HTTPException(status_code=403, detail="Not allowed to view agent traces")
    return TraceScope(team_ids=[""], api_key_hash=user_api_key_dict.token)


@router.post("/v1/traces", include_in_schema=False)
async def ingest_otlp_traces(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Response:
    tracing: Final = get_receiver()
    content_type: Final = request.headers.get("content-type")
    try:
        await tracing.ingest(
            body=await request.body(),
            content_type=content_type,
            content_encoding=request.headers.get("content-encoding"),
            tenant=tenant_for(user_api_key_dict),
        )
    except TracingBackpressureError:
        raise HTTPException(status_code=429, headers={"Retry-After": str(OTLP_RETRY_AFTER_SECONDS)})
    except TracingPayloadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    body, media_type = encode_otlp_response(content_type)
    return Response(content=body, media_type=media_type)


@router.get("/v1/traces", response_model=None)
async def list_agent_traces(
    start_ms: int | None = Query(None, description="Window start, unix ms. Default: 24h ago"),
    end_ms: int | None = Query(None, description="Window end, unix ms. Default: now"),
    cursor: str | None = Query(None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TracePage:
    now_ms: Final = int(time.time() * 1000)
    return await get_receiver().list_traces(
        scope=scope_for(user_api_key_dict),
        start_ms=start_ms if start_ms is not None else now_ms - MS_PER_DAY,
        end_ms=end_ms if end_ms is not None else now_ms,
        cursor=cursor,
    )


@router.get("/v1/traces/{trace_id}", response_model=None)
async def get_agent_trace(
    trace_id: str,
    format: Literal["json", "md"] = Query("json", description="`md` returns Markdown for pasting into Claude / Codex"),
    span_id: str | None = Query(None, description="With format=md: only the subtree under this span"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Trace | Response:
    scope: Final = scope_for(user_api_key_dict)
    if format == "md":
        markdown: Final = await get_receiver().get_trace_markdown(trace_id, scope, span_id)
        if markdown is None:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        return Response(content=markdown, media_type="text/markdown; charset=utf-8")
    trace: Final = await get_receiver().get_trace(trace_id, scope)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@router.get("/v1/traces/{trace_id}/spans/{span_id}", response_model=None)
async def get_agent_trace_span(
    trace_id: str,
    span_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SpanDetail:
    span: Final = await get_receiver().get_span(trace_id, span_id, scope_for(user_api_key_dict))
    if span is None:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not found")
    return span
