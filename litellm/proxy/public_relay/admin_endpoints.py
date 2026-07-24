from __future__ import annotations

from datetime import datetime
from typing import Annotated

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.public_relay.api_types import (
    AccountStatusRequest,
    AdminAccountListResponse,
    AdminAccountResponse,
    AdminPaymentListResponse,
    AdminPaymentResponse,
    MarginResponse,
    MessageResponse,
    ModelPriceCreateRequest,
    ModelPriceResponse,
    PricingResponse,
    RefundRequest,
    RefundResponse,
    WalletAdjustmentRequest,
    WalletResponse,
)
from litellm.proxy.public_relay.db_types import PriceRow, RefundRow
from litellm.proxy.public_relay.repository import (
    adjust_wallet,
    attach_refund_submission,
    begin_refund,
    complete_refund,
    fail_refund,
    get_account_by_id,
    get_margin_summary,
    list_active_prices,
    list_admin_accounts,
    list_admin_payments,
    list_api_keys,
    publish_price,
    set_account_status,
)
from litellm.proxy.public_relay.runtime import database, money_response, settings, wallet_response
from litellm.proxy.public_relay.stripe_client import StripeSdkGateway

router = APIRouter(prefix="/v1/admin/relay", tags=["public relay administration"])


@router.get("/accounts", response_model=AdminAccountListResponse)
async def accounts(
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    cursor: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> AdminAccountListResponse:
    _require_admin(user)
    settings()
    values = await list_admin_accounts(database(), cursor, limit)
    data = tuple(
        AdminAccountResponse(
            account_id=value.account_id,
            user_id=value.user_id,
            email=value.normalized_email,
            status=value.status,
            created_at=value.created_at,
            wallet_id=value.wallet_id,
            available=money_response(value.available_micros),
            reserved=money_response(value.reserved_micros),
            debt=money_response(value.debt_micros),
        )
        for value in values
    )
    next_cursor = values[-1].created_at.isoformat() if len(values) == limit else None
    return AdminAccountListResponse(data=data, next_cursor=next_cursor)


@router.get("/payments", response_model=AdminPaymentListResponse)
async def admin_payments(
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    cursor: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> AdminPaymentListResponse:
    _require_admin(user)
    settings()
    values = await list_admin_payments(database(), cursor, limit)
    data = tuple(
        AdminPaymentResponse(
            payment_id=value.payment_id,
            account_id=value.account_id,
            wallet_id=value.wallet_id,
            email=value.normalized_email,
            amount=money_response(value.amount_micros),
            refunded=money_response(value.refunded_micros),
            status=value.status,
            created_at=value.created_at,
        )
        for value in values
    )
    next_cursor = values[-1].created_at.isoformat() if len(values) == limit else None
    return AdminPaymentListResponse(data=data, next_cursor=next_cursor)


@router.get("/margin", response_model=MarginResponse)
async def margin(
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> MarginResponse:
    _require_admin(user)
    settings()
    value = await get_margin_summary(database())
    return MarginResponse(
        charged=money_response(value.charged_micros),
        upstream_cost=money_response(value.upstream_cost_micros),
        gross_margin=money_response(value.charged_micros - value.upstream_cost_micros),
    )


@router.get("/prices", response_model=PricingResponse)
async def admin_prices(
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> PricingResponse:
    _require_admin(user)
    settings()
    values = await list_active_prices(database())
    return PricingResponse(models=tuple(_price_response(value) for value in values))


@router.post("/prices", response_model=ModelPriceResponse)
async def create_price(
    payload: ModelPriceCreateRequest,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ModelPriceResponse:
    _require_admin(user)
    settings()
    if payload.default_max_output_tokens > payload.max_output_tokens:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Default output limit exceeds maximum")
    created = await publish_price(database(), payload, user.user_id or "proxy-admin")
    return _price_response(created)


@router.post("/accounts/{account_id}/status", response_model=MessageResponse)
async def account_status(
    account_id: str,
    payload: AccountStatusRequest,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> MessageResponse:
    _require_admin(user)
    settings()
    account = await get_account_by_id(database(), account_id)
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")
    key_values = await list_api_keys(database(), account)
    await set_account_status(database(), account_id, payload.status)
    from litellm.proxy.proxy_server import user_api_key_cache

    for key in key_values:
        await user_api_key_cache.async_delete_cache(key.token)
    return MessageResponse(message="Account status updated")


@router.post("/wallets/{wallet_id}/adjust", response_model=WalletResponse)
async def wallet_adjustment(
    wallet_id: str,
    payload: WalletAdjustmentRequest,
    idempotency_key: Annotated[str, Header(alias="idempotency-key", min_length=8, max_length=255)],
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> WalletResponse:
    _require_admin(user)
    settings()
    try:
        value = await adjust_wallet(database(), wallet_id, payload.amount_micros, payload.reason, idempotency_key)
    except (ArithmeticError, LookupError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return wallet_response(value)


@router.post("/payments/{payment_id}/refund", response_model=RefundResponse)
async def refund_payment(
    payment_id: str,
    payload: RefundRequest,
    idempotency_key: Annotated[str, Header(alias="idempotency-key", min_length=8, max_length=255)],
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> RefundResponse:
    _require_admin(user)
    value = settings()
    try:
        operation = await begin_refund(
            database(),
            payment_id,
            payload.amount_micros,
            payload.reason,
            idempotency_key,
        )
    except (ArithmeticError, LookupError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if operation.refund.status == "SUCCEEDED":
        return _refund_response(operation.refund)
    gateway = StripeSdkGateway(
        secret_key=value.stripe_secret_key or "",
        success_url=value.checkout_success_url or "",
        cancel_url=value.checkout_cancel_url or "",
    )
    try:
        stripe_refund = await gateway.create_refund(
            operation.refund.refund_id,
            operation.payment_intent_id,
            operation.refund.amount_micros,
            operation.refund.idempotency_key,
            operation.refund.reason,
        )
    except stripe.StripeError as exc:
        await fail_refund(database(), operation.refund.refund_id, str(exc))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Stripe refund failed") from exc
    submitted = await attach_refund_submission(
        database(),
        operation.refund.refund_id,
        stripe_refund.refund_id,
    )
    if stripe_refund.status == "failed":
        await fail_refund(database(), operation.refund.refund_id, "Stripe marked the refund as failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Stripe refund failed")
    if stripe_refund.status != "succeeded":
        return _refund_response(submitted)
    completed = await complete_refund(database(), operation.refund.refund_id, stripe_refund.refund_id)
    return _refund_response(completed)


def _require_admin(user: UserAPIKeyAuth) -> None:
    if user.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Proxy admin access required")


def _price_response(value: PriceRow) -> ModelPriceResponse:
    return ModelPriceResponse(
        price_id=value.price_id,
        model_name=value.model_name,
        version=value.version,
        input_micros_per_million=value.input_micros_per_million,
        cached_input_micros_per_million=value.cached_input_micros_per_million,
        output_micros_per_million=value.output_micros_per_million,
        embedding_micros_per_million=value.embedding_micros_per_million,
        default_max_output_tokens=value.default_max_output_tokens,
        max_output_tokens=value.max_output_tokens,
        enabled=value.enabled,
        effective_at=value.effective_at,
    )


def _refund_response(value: RefundRow) -> RefundResponse:
    return RefundResponse(
        refund_id=value.refund_id,
        payment_id=value.payment_id,
        amount=money_response(value.amount_micros),
        status=value.status,
    )
