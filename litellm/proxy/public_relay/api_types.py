from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MoneyResponse(BaseModel):
    currency: Literal["USD"] = "USD"
    amount_micros: str
    display: str


class ActivateRequest(BaseModel):
    token: str = Field(min_length=32, max_length=512)
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class PasswordResetRequest(BaseModel):
    token: str = Field(min_length=32, max_length=512)
    password: str


class MessageResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    enabled: bool
    operational: bool


class AccountResponse(BaseModel):
    account_id: str
    user_id: str
    email: str
    company_name: str
    notes: str | None
    status: Literal["INVITED", "ACTIVE", "FROZEN", "CLOSED"]
    created_at: datetime


class WalletResponse(BaseModel):
    available: MoneyResponse
    reserved: MoneyResponse


class LedgerEntryResponse(BaseModel):
    entry_id: str
    entry_type: str
    amount: MoneyResponse
    available_after: MoneyResponse
    reserved_after: MoneyResponse
    created_at: datetime
    request_id: str | None = None


class LedgerListResponse(BaseModel):
    data: tuple[LedgerEntryResponse, ...]
    next_cursor: str | None = None


class ModelPriceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1, max_length=256)
    input_micros_per_million: int = Field(ge=0)
    cached_input_micros_per_million: int | None = Field(default=None, ge=0)
    output_micros_per_million: int | None = Field(default=None, ge=0)
    embedding_micros_per_million: int | None = Field(default=None, ge=0)
    default_max_output_tokens: int = Field(default=4096, ge=1)
    max_output_tokens: int = Field(default=4096, ge=1)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_price_shape(self) -> ModelPriceCreateRequest:
        if self.default_max_output_tokens > self.max_output_tokens:
            raise ValueError("default output limit exceeds maximum")
        if self.output_micros_per_million is None and self.embedding_micros_per_million is None:
            raise ValueError("output or embedding pricing is required")
        return self


class ModelPriceResponse(ModelPriceCreateRequest):
    price_id: str
    version: int
    effective_at: datetime


class PricingResponse(BaseModel):
    currency: Literal["USD"] = "USD"
    models: tuple[ModelPriceResponse, ...]


class ApiKeyCreateRequest(BaseModel):
    alias: str = Field(min_length=1, max_length=128)
    log_content: bool = False


class ApiKeyResponse(BaseModel):
    key_id: str
    alias: str | None
    key: str | None = None
    created_at: datetime | None = None
    log_content: bool


class ApiKeyListResponse(BaseModel):
    data: tuple[ApiKeyResponse, ...]


class SessionResponse(BaseModel):
    account: AccountResponse
    csrf_token: str
    default_key: ApiKeyResponse | None = None


class UsageResponse(BaseModel):
    request_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    charged: MoneyResponse
    upstream_cost: MoneyResponse


class RequestLogResponse(BaseModel):
    request_id: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    charged: MoneyResponse
    status: str | None
    request_duration_ms: int | None
    created_at: datetime


class RequestLogListResponse(BaseModel):
    data: tuple[RequestLogResponse, ...]
    next_cursor: str | None = None


class WalletAdjustmentRequest(BaseModel):
    amount_micros: int
    reason: str = Field(min_length=1, max_length=500)


class EnterpriseCreateRequest(BaseModel):
    company_name: str = Field(min_length=1, max_length=200)
    admin_email: str
    initial_credit_micros: int = Field(default=0, ge=0)
    notes: str | None = Field(default=None, max_length=2000)


class AccountStatusRequest(BaseModel):
    status: Literal["ACTIVE", "FROZEN", "CLOSED"]


class AuthLinkResponse(BaseModel):
    account_id: str
    expires_at: datetime
    url: str


class EnterpriseCreateResponse(BaseModel):
    account: AccountResponse
    wallet: WalletResponse
    activation: AuthLinkResponse


class AdminAccountResponse(AccountResponse):
    wallet_id: str
    available: MoneyResponse
    reserved: MoneyResponse


class AdminAccountListResponse(BaseModel):
    data: tuple[AdminAccountResponse, ...]
    next_cursor: str | None = None


class MarginResponse(BaseModel):
    charged: MoneyResponse
    upstream_cost: MoneyResponse
    gross_margin: MoneyResponse
