"""
Agent tracing endpoints. Thin wrappers over `TraceReceiver`: auth -> tenant/scope -> one call.

POST /v1/traces                              OTLP/HTTP trace export (protobuf or JSON)
GET  /v1/traces                              TracePage
GET  /v1/traces/{trace_id}                   Trace
GET  /v1/traces/{trace_id}/spans/{span_id}   SpanDetail
"""

import time
from dataclasses import dataclass
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from litellm.constants import OTLP_MAX_BODY_BYTES, OTLP_RETRY_AFTER_SECONDS
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.tracing import (
    Tenant,
    TraceReceiver,
    TracingPayloadTooLargeError,
)
from litellm.tracing.decode import InvalidOTLPPayloadError, encode_otlp_response
from litellm.tracing.types import SpanDetail, Trace, TracePage, TraceScope

router = APIRouter(tags=["agent tracing"])  # mutable-ok: FastAPI copies the mutable tags list

MS_PER_DAY: Final = 24 * 60 * 60 * 1000
_ADMIN_ROLES: Final = (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

receiver: TraceReceiver | None = None


def get_receiver() -> TraceReceiver:
    return _require_receiver(receiver)


def _require_receiver(tracing: TraceReceiver | None) -> TraceReceiver:
    if tracing is None:
        raise HTTPException(
            status_code=501,
            detail="Agent tracing is not enabled. Set `tracing:` in general_settings and CLICKHOUSE_URL.",
        )
    return tracing


def tenant_for(user_api_key_dict: UserAPIKeyAuth) -> Tenant:
    return Tenant(
        team_id=user_api_key_dict.team_id or "",
        api_key_hash=user_api_key_dict.token or "",
        org_id=user_api_key_dict.org_id or "",
    )


def scope_for(user_api_key_dict: UserAPIKeyAuth) -> TraceScope:
    """Admins see everything; team members see their team; team-less keys see their own traces."""
    if user_api_key_dict.user_role in _ADMIN_ROLES:
        return TraceScope(team_ids=(), api_key_hash="")
    if user_api_key_dict.team_id:
        return TraceScope(team_ids=(user_api_key_dict.team_id,), api_key_hash="")
    if not user_api_key_dict.token:
        raise HTTPException(status_code=403, detail="Not allowed to view agent traces")
    return TraceScope(team_ids=("",), api_key_hash=user_api_key_dict.token)


@dataclass(frozen=True, slots=True)
class TraceAccessContext:
    auth: UserAPIKeyAuth
    receiver: TraceReceiver | None

    def reader(self) -> tuple[TraceReceiver, TraceScope]:
        return _require_receiver(self.receiver), scope_for(self.auth)

    def writer(self) -> tuple[TraceReceiver, Tenant]:
        if self.auth.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
            raise HTTPException(status_code=403, detail="Not allowed to ingest agent traces")
        return _require_receiver(self.receiver), tenant_for(self.auth)


async def provide_receiver() -> TraceReceiver | None:
    return receiver


async def provide_trace_access(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tracing: Annotated[TraceReceiver | None, Depends(provide_receiver)],
) -> TraceAccessContext:
    return TraceAccessContext(auth=auth, receiver=tracing)


async def _read_otlp_body(request: Request) -> bytes:
    body: Final = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > OTLP_MAX_BODY_BYTES:
            raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
        body.extend(chunk)
    return bytes(body)


@router.post("/v1/traces", include_in_schema=False)
async def ingest_otlp_traces(
    request: Request,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> Response:
    tracing, tenant = context.writer()
    content_type: Final = request.headers.get("content-type")
    try:
        await tracing.ingest(
            body=await _read_otlp_body(request),
            content_type=content_type,
            content_encoding=request.headers.get("content-encoding"),
            tenant=tenant,
        )
    except TracingPayloadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except InvalidOTLPPayloadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            headers={"Retry-After": str(OTLP_RETRY_AFTER_SECONDS)},  # mutable-ok: FastAPI requires dict headers
        )
    body, media_type = encode_otlp_response(content_type)
    return Response(content=body, media_type=media_type)


@router.get("/v1/traces", response_model=None)
async def list_agent_traces(
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    start_ms: Annotated[int | None, Query(description="Window start, unix ms. Default: 24h ago")] = None,
    end_ms: Annotated[int | None, Query(description="Window end, unix ms. Default: now")] = None,
    cursor: Annotated[str | None, Query()] = None,
) -> TracePage:
    now_ms: Final = int(time.time() * 1000)
    try:
        tracing, scope = context.reader()
        return await tracing.list_traces(
            scope=scope,
            start_ms=start_ms if start_ms is not None else now_ms - MS_PER_DAY,
            end_ms=end_ms if end_ms is not None else now_ms,
            cursor=cursor,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/v1/traces/{trace_id}", response_model=None)
async def get_agent_trace(
    trace_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
) -> Trace:
    tracing, scope = context.reader()
    trace: Final = await tracing.get_trace(trace_id, scope, trace_ref)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@router.get("/v1/traces/{trace_id}/spans/{span_id}", response_model=None)
async def get_agent_trace_span(
    trace_id: str,
    span_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
) -> SpanDetail:
    tracing, scope = context.reader()
    span: Final = await tracing.get_span(trace_id, span_id, scope, trace_ref)
    if span is None:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not found")
    return span
