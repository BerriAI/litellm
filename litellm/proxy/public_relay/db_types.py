from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DatabaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class AccountRow(DatabaseModel):
    account_id: str
    user_id: str
    normalized_email: str
    company_name: str
    notes: str | None
    status: Literal["INVITED", "ACTIVE", "FROZEN", "CLOSED"]
    password: str | None = None
    session_version: int
    created_at: datetime


class WalletRow(DatabaseModel):
    wallet_id: str
    account_id: str
    available_micros: int
    reserved_micros: int


class PriceRow(DatabaseModel):
    price_id: str
    model_name: str
    version: int
    input_micros_per_million: int
    cached_input_micros_per_million: int | None
    output_micros_per_million: int | None
    embedding_micros_per_million: int | None
    default_max_output_tokens: int
    max_output_tokens: int
    enabled: bool
    effective_at: datetime


class ReservationRow(DatabaseModel):
    reservation_id: str
    request_id: str
    account_id: str
    wallet_id: str
    price_id: str
    reserved_micros: int
    input_tokens: int
    max_output_tokens: int
    status: Literal["OPEN", "FINALIZED", "RELEASED"]


class ReservationSettlementRow(ReservationRow):
    input_micros_per_million: int
    cached_input_micros_per_million: int | None
    output_micros_per_million: int | None
    embedding_micros_per_million: int | None


class LedgerRow(DatabaseModel):
    entry_id: str
    entry_type: Literal["RESERVE", "RELEASE", "USAGE", "ADJUSTMENT"]
    amount_micros: int
    available_after_micros: int
    reserved_after_micros: int
    request_id: str | None
    created_at: datetime


class PortalSessionRow(DatabaseModel):
    session_id: str
    account_id: str
    user_id: str
    normalized_email: str
    session_version: int
    csrf_token: str
    expires_at: datetime


class AuthTokenRow(DatabaseModel):
    auth_token_id: str
    account_id: str
    purpose: Literal["ACTIVATION", "PASSWORD_RESET"]
    expires_at: datetime
    consumed_at: datetime | None


class KeyRow(DatabaseModel):
    token: str
    key_alias: str | None
    metadata: object
    created_at: datetime | None


class CountRow(DatabaseModel):
    count: int


class UsageSummaryRow(DatabaseModel):
    request_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    charged_micros: int
    upstream_cost_micros: int


class RequestLogRow(DatabaseModel):
    request_id: str
    model_name: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    charged_micros: int
    upstream_cost_micros: int
    status: str | None
    request_duration_ms: int | None
    created_at: datetime


class AdminAccountRow(AccountRow):
    wallet_id: str
    available_micros: int
    reserved_micros: int


class MarginSummaryRow(DatabaseModel):
    charged_micros: int
    upstream_cost_micros: int
