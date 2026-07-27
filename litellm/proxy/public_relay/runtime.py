from __future__ import annotations

from dataclasses import dataclass
from typing import cast  # noqa: TID251, RUF100  # Virtual-key metadata requires runtime boundary narrowing.

from fastapi import HTTPException, Request, status

from litellm.proxy.public_relay.api_types import (
    AccountResponse,
    ApiKeyResponse,
    LedgerEntryResponse,
    MoneyResponse,
    WalletResponse,
)
from litellm.proxy.public_relay.config import PublicRelaySettings
from litellm.proxy.public_relay.db_types import AccountRow, KeyRow, LedgerRow, WalletRow
from litellm.proxy.public_relay.money import display_usd
from litellm.proxy.public_relay.session_store import PortalSession, RelayStore
from litellm.proxy.utils import PrismaClient, get_prisma_client_or_throw

SESSION_COOKIE = "public_relay_session"


@dataclass(frozen=True, slots=True)
class PortalContext:
    account: AccountRow
    session: PortalSession
    token: str


def settings(require_operational: bool = True) -> PublicRelaySettings:
    value = PublicRelaySettings.from_env()
    if not value.enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if require_operational and value.missing_runtime_configuration():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Public relay is not configured", "type": "service_unavailable"}},
        )
    return value


def database() -> PrismaClient:
    return get_prisma_client_or_throw("Public relay requires a database")


def relay_store(value: PublicRelaySettings) -> RelayStore:
    return RelayStore(prisma_client=database(), settings=value)


async def require_portal(request: Request) -> PortalContext:
    from litellm.proxy.public_relay.repository import get_account_by_id

    value = settings()
    token = request.cookies.get(SESSION_COOKIE)
    if token is None:
        raise _unauthorized()
    session = await relay_store(value).get_session(token)
    if session is None:
        raise _unauthorized()
    account = await get_account_by_id(database(), session.account_id)
    if (
        account is None
        or account.status != "ACTIVE"
        or account.session_version != session.session_version
        or account.user_id != session.user_id
    ):
        raise _unauthorized()
    return PortalContext(account=account, session=session, token=token)


async def require_portal_write(request: Request) -> PortalContext:
    context = await require_portal(request)
    if request.headers.get("x-csrf-token") != context.session.csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": {"message": "Invalid CSRF token", "type": "invalid_request_error"}},
        )
    return context


def remote_ip(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip")
    if forwarded:
        return forwarded
    return request.client.host if request.client is not None else "unknown"


def account_response(account: AccountRow) -> AccountResponse:
    return AccountResponse(
        account_id=account.account_id,
        user_id=account.user_id,
        email=account.normalized_email,
        company_name=account.company_name,
        notes=account.notes,
        status=account.status,
        created_at=account.created_at,
    )


def money_response(amount_micros: int) -> MoneyResponse:
    return MoneyResponse(amount_micros=str(amount_micros), display=display_usd(amount_micros))


def wallet_response(wallet: WalletRow) -> WalletResponse:
    return WalletResponse(
        available=money_response(wallet.available_micros),
        reserved=money_response(wallet.reserved_micros),
    )


def ledger_response(entry: LedgerRow) -> LedgerEntryResponse:
    return LedgerEntryResponse(
        entry_id=entry.entry_id,
        entry_type=entry.entry_type,
        amount=money_response(entry.amount_micros),
        available_after=money_response(entry.available_after_micros),
        reserved_after=money_response(entry.reserved_after_micros),
        request_id=entry.request_id,
        created_at=entry.created_at,
    )


def key_response(key: KeyRow, raw_key: str | None = None) -> ApiKeyResponse:
    metadata = (
        cast(dict[str, object], key.metadata)  # cast-ok: isinstance validates virtual-key metadata.
        if isinstance(key.metadata, dict)
        else {}
    )
    return ApiKeyResponse(
        key_id=key.token,
        alias=key.key_alias,
        key=raw_key,
        created_at=key.created_at,
        log_content=metadata.get("public_relay_log_content") is True,
    )


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {"message": "Invalid credentials", "type": "authentication_error"}},
    )
