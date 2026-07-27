from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.public_relay.api_types import (
    AccountStatusRequest,
    AdminAccountListResponse,
    AdminAccountResponse,
    AuthLinkResponse,
    EnterpriseCreateRequest,
    EnterpriseCreateResponse,
    MarginResponse,
    MessageResponse,
    ModelPriceCreateRequest,
    ModelPriceResponse,
    PricingResponse,
    WalletAdjustmentRequest,
    WalletResponse,
)
from litellm.proxy.public_relay.db_types import PriceRow
from litellm.proxy.public_relay.repository import (
    adjust_wallet,
    create_auth_token,
    create_enterprise,
    get_account_by_id,
    get_margin_summary,
    list_active_prices,
    list_admin_accounts,
    list_api_keys,
    publish_price,
    set_account_status,
)
from litellm.proxy.public_relay.runtime import (
    account_response,
    database,
    money_response,
    settings,
    wallet_response,
)
from litellm.proxy.public_relay.security import hash_auth_token, new_auth_token, normalize_email

router = APIRouter(prefix="/v1/admin/relay", tags=["public relay administration"])


@router.post("/accounts", response_model=EnterpriseCreateResponse)
async def create_account(
    payload: EnterpriseCreateRequest,
    idempotency_key: Annotated[str, Header(alias="idempotency-key", min_length=8, max_length=255)],
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> EnterpriseCreateResponse:
    _require_admin(user)
    value = settings()
    raw_token = new_auth_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=72)
    try:
        normalized_email = normalize_email(payload.admin_email)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid enterprise details") from exc
    created = await create_enterprise(
        database(),
        normalized_email,
        payload.company_name.strip(),
        payload.notes,
        payload.initial_credit_micros,
        idempotency_key,
        hash_auth_token(value.session_secret, raw_token),
        expires_at,
    )
    return EnterpriseCreateResponse(
        account=account_response(created.account),
        wallet=wallet_response(created.wallet),
        activation=_auth_link(created.account.account_id, raw_token, "activate", expires_at),
    )


@router.post("/accounts/{account_id}/activation-link", response_model=AuthLinkResponse)
async def activation_link(
    account_id: str,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> AuthLinkResponse:
    _require_admin(user)
    account = await get_account_by_id(database(), account_id)
    if account is None or account.status != "INVITED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Account is not awaiting activation")
    return await _new_auth_link(account_id, "ACTIVATION", "activate", timedelta(hours=72))


@router.post("/accounts/{account_id}/password-reset-link", response_model=AuthLinkResponse)
async def password_reset_link(
    account_id: str,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> AuthLinkResponse:
    _require_admin(user)
    account = await get_account_by_id(database(), account_id)
    if account is None or account.status != "ACTIVE":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Account is not active")
    return await _new_auth_link(account_id, "PASSWORD_RESET", "password-reset", timedelta(minutes=30))


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
            company_name=value.company_name,
            notes=value.notes,
            status=value.status,
            created_at=value.created_at,
            wallet_id=value.wallet_id,
            available=money_response(value.available_micros),
            reserved=money_response(value.reserved_micros),
        )
        for value in values
    )
    next_cursor = values[-1].created_at.isoformat() if len(values) == limit else None
    return AdminAccountListResponse(data=data, next_cursor=next_cursor)


@router.get("/margin", response_model=MarginResponse)
async def margin(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> MarginResponse:
    _require_admin(user)
    settings()
    value = await get_margin_summary(database())
    return MarginResponse(
        charged=money_response(value.charged_micros),
        upstream_cost=money_response(value.upstream_cost_micros),
        gross_margin=money_response(value.charged_micros - value.upstream_cost_micros),
    )


@router.get("/prices", response_model=PricingResponse)
async def admin_prices(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> PricingResponse:
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


async def _new_auth_link(
    account_id: str,
    purpose: str,
    page: str,
    lifetime: timedelta,
) -> AuthLinkResponse:
    value = settings()
    raw_token = new_auth_token()
    expires_at = datetime.now(timezone.utc) + lifetime
    await create_auth_token(
        database(),
        account_id,
        hash_auth_token(value.session_secret, raw_token),
        purpose,
        expires_at,
    )
    return _auth_link(account_id, raw_token, page, expires_at)


def _auth_link(account_id: str, raw_token: str, page: str, expires_at: datetime) -> AuthLinkResponse:
    value = settings()
    base_url = (value.base_url or "").rstrip("/")
    url = f"{base_url}/{page}?{urlencode({'token': raw_token})}"
    return AuthLinkResponse(account_id=account_id, expires_at=expires_at, url=url)


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
