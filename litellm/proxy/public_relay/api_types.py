from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MoneyResponse(BaseModel):
    currency: Literal["USD"] = "USD"
    amount_micros: str
    display: str


class VerificationCodeRequest(BaseModel):
    email: str
    turnstile_token: str = Field(min_length=1, max_length=4096)


class RegisterRequest(BaseModel):
    email: str
    code: str = Field(pattern=r"^\d{6}$")
    password: str
    turnstile_token: str = Field(min_length=1, max_length=4096)


class LoginRequest(BaseModel):
    email: str
    password: str
    turnstile_token: str = Field(min_length=1, max_length=4096)


class PasswordResetRequest(BaseModel):
    email: str
    code: str = Field(pattern=r"^\d{6}$")
    password: str
    turnstile_token: str = Field(min_length=1, max_length=4096)


class MessageResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    enabled: bool
    operational: bool


class AccountResponse(BaseModel):
    account_id: str
    user_id: str
    email: str
    status: Literal["ACTIVE", "FROZEN", "CLOSED"]
    created_at: datetime


class WalletResponse(BaseModel):
    available: MoneyResponse
    reserved: MoneyResponse
    debt: MoneyResponse


class LedgerEntryResponse(BaseModel):
    entry_id: str
    entry_type: str
    amount: MoneyResponse
    available_after: MoneyResponse
    reserved_after: MoneyResponse
    created_at: datetime
    request_id: str | None = None
    payment_id: str | None = None


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
    log_content: bool = True


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


class CheckoutRequest(BaseModel):
    amount_cents: int
    turnstile_token: str = Field(min_length=1, max_length=4096)


class CheckoutResponse(BaseModel):
    payment_id: str
    checkout_url: str


class PaymentResponse(BaseModel):
    payment_id: str
    amount: MoneyResponse
    refunded: MoneyResponse
    status: str
    created_at: datetime


class PaymentListResponse(BaseModel):
    data: tuple[PaymentResponse, ...]
    next_cursor: str | None = None


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


class RefundRequest(BaseModel):
    amount_micros: int = Field(gt=0, multiple_of=10_000)
    reason: str = Field(min_length=1, max_length=500)


class AccountStatusRequest(BaseModel):
    status: Literal["ACTIVE", "FROZEN", "CLOSED"]


class RefundResponse(BaseModel):
    refund_id: str
    payment_id: str
    amount: MoneyResponse
    status: str


class AdminAccountResponse(AccountResponse):
    wallet_id: str
    available: MoneyResponse
    reserved: MoneyResponse
    debt: MoneyResponse


class AdminAccountListResponse(BaseModel):
    data: tuple[AdminAccountResponse, ...]
    next_cursor: str | None = None


class AdminPaymentResponse(PaymentResponse):
    account_id: str
    wallet_id: str
    email: str


class AdminPaymentListResponse(BaseModel):
    data: tuple[AdminPaymentResponse, ...]
    next_cursor: str | None = None


class MarginResponse(BaseModel):
    charged: MoneyResponse
    upstream_cost: MoneyResponse
    gross_margin: MoneyResponse
