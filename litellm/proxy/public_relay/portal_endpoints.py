from __future__ import annotations

from datetime import datetime
from typing import Annotated

import stripe
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from litellm.proxy.public_relay.api_types import (
    AccountResponse,
    ApiKeyCreateRequest,
    ApiKeyListResponse,
    ApiKeyResponse,
    CheckoutRequest,
    CheckoutResponse,
    LedgerListResponse,
    MessageResponse,
    PaymentListResponse,
    RequestLogListResponse,
    RequestLogResponse,
    UsageResponse,
    WalletResponse,
)
from litellm.proxy.public_relay.repository import (
    attach_checkout_session,
    create_api_key,
    create_checkout_order,
    delete_api_key,
    fail_checkout_creation,
    get_usage_summary,
    get_wallet,
    list_api_keys,
    list_ledger,
    list_payments,
    list_request_logs,
)
from litellm.proxy.public_relay.runtime import (
    PortalContext,
    account_response,
    database,
    key_response,
    ledger_response,
    money_response,
    payment_response,
    relay_cache,
    remote_ip,
    require_portal,
    require_portal_write,
    settings,
    wallet_response,
)
from litellm.proxy.public_relay.stripe_client import StripeSdkGateway
from litellm.proxy.public_relay.turnstile import TurnstileVerifier

router = APIRouter(prefix="/v1/portal", tags=["public relay portal"])


@router.get("/me", response_model=AccountResponse)
async def me(
    response: Response,
    context: Annotated[PortalContext, Depends(require_portal)],
) -> AccountResponse:
    response.headers["x-csrf-token"] = context.session.csrf_token
    return account_response(context.account)


@router.get("/wallet", response_model=WalletResponse)
async def wallet(context: Annotated[PortalContext, Depends(require_portal)]) -> WalletResponse:
    value = await get_wallet(database(), context.account.account_id)
    if value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wallet not found")
    return wallet_response(value)


@router.get("/ledger", response_model=LedgerListResponse)
async def ledger(
    context: Annotated[PortalContext, Depends(require_portal)],
    cursor: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> LedgerListResponse:
    entries = await list_ledger(database(), context.account.account_id, cursor, limit)
    next_cursor = entries[-1].created_at.isoformat() if len(entries) == limit else None
    return LedgerListResponse(data=tuple(ledger_response(entry) for entry in entries), next_cursor=next_cursor)


@router.get("/usage", response_model=UsageResponse)
async def usage(context: Annotated[PortalContext, Depends(require_portal)]) -> UsageResponse:
    summary = await get_usage_summary(database(), context.account.account_id)
    return UsageResponse(
        request_count=summary.request_count,
        input_tokens=summary.input_tokens,
        cached_input_tokens=summary.cached_input_tokens,
        output_tokens=summary.output_tokens,
        charged=money_response(summary.charged_micros),
        upstream_cost=money_response(summary.upstream_cost_micros),
    )


@router.get("/logs", response_model=RequestLogListResponse)
async def logs(
    context: Annotated[PortalContext, Depends(require_portal)],
    cursor: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> RequestLogListResponse:
    entries = await list_request_logs(database(), context.account.account_id, cursor, limit)
    data = tuple(
        RequestLogResponse(
            request_id=entry.request_id,
            model=entry.model_name,
            input_tokens=entry.input_tokens,
            cached_input_tokens=entry.cached_input_tokens,
            output_tokens=entry.output_tokens,
            charged=money_response(entry.charged_micros),
            status=entry.status,
            request_duration_ms=entry.request_duration_ms,
            created_at=entry.created_at,
        )
        for entry in entries
    )
    next_cursor = entries[-1].created_at.isoformat() if len(entries) == limit else None
    return RequestLogListResponse(data=data, next_cursor=next_cursor)


@router.get("/keys", response_model=ApiKeyListResponse)
async def keys(context: Annotated[PortalContext, Depends(require_portal)]) -> ApiKeyListResponse:
    values = await list_api_keys(database(), context.account)
    return ApiKeyListResponse(data=tuple(key_response(value) for value in values))


@router.post("/keys", response_model=ApiKeyResponse)
async def create_key(
    payload: ApiKeyCreateRequest,
    context: Annotated[PortalContext, Depends(require_portal_write)],
) -> ApiKeyResponse:
    value = settings()
    try:
        created = await create_api_key(
            database(),
            context.account,
            payload.alias,
            payload.log_content,
            value.max_api_keys,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
        ) from exc
    return ApiKeyResponse(
        key_id=created.key_id,
        alias=payload.alias,
        key=created.raw_key,
        log_content=payload.log_content,
    )


@router.delete("/keys/{key_id}", response_model=MessageResponse)
async def remove_key(
    key_id: str,
    context: Annotated[PortalContext, Depends(require_portal_write)],
) -> MessageResponse:
    deleted = await delete_api_key(database(), context.account, key_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    from litellm.proxy.proxy_server import user_api_key_cache

    await user_api_key_cache.async_delete_cache(key_id)
    return MessageResponse(message="API key deleted")


@router.post("/billing/checkout", response_model=CheckoutResponse)
async def checkout(
    payload: CheckoutRequest,
    request: Request,
    context: Annotated[PortalContext, Depends(require_portal_write)],
) -> CheckoutResponse:
    value = settings()
    if payload.amount_cents < value.min_checkout_cents or payload.amount_cents > value.max_checkout_cents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {"message": "Checkout amount is outside the allowed range", "type": "invalid_request_error"}
            },
        )
    ip_address = remote_ip(request)
    if value.turnstile_verify_url is None or not await TurnstileVerifier(value.turnstile_verify_url).verify(
        payload.turnstile_token,
        ip_address,
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Human verification failed")
    cache = relay_cache(value)
    try:
        await cache.enforce_limit(f"public-relay:checkout:{context.account.account_id}", 10, 3600)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests") from exc
    order = await create_checkout_order(database(), context.account.account_id, payload.amount_cents * 10_000)
    gateway = StripeSdkGateway(
        secret_key=value.stripe_secret_key or "",
        success_url=value.checkout_success_url or "",
        cancel_url=value.checkout_cancel_url or "",
    )
    try:
        session = await gateway.create_checkout(order, context.account.normalized_email)
    except (stripe.StripeError, RuntimeError) as exc:
        await fail_checkout_creation(database(), order.payment_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Unable to start Stripe Checkout") from exc
    await attach_checkout_session(database(), order.payment_id, session.session_id)
    return CheckoutResponse(payment_id=order.payment_id, checkout_url=session.url)


@router.get("/billing/payments", response_model=PaymentListResponse)
async def payments(
    context: Annotated[PortalContext, Depends(require_portal)],
    cursor: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> PaymentListResponse:
    values = await list_payments(database(), context.account.account_id, cursor, limit)
    next_cursor = values[-1].created_at.isoformat() if len(values) == limit else None
    return PaymentListResponse(
        data=tuple(payment_response(value) for value in values),
        next_cursor=next_cursor,
    )
