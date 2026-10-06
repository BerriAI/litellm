from datetime import date
from typing import Final

from pydantic import BaseModel, Field

from litellm.models.subscription_account import LiteLLM_SubscriptionAccountTable

CURRENCY_CODE_PATTERN: Final = r"^[A-Z]{3}$"


class SubscriptionAccountFeeRequest(BaseModel):
    custom_llm_provider: str
    account_id: str
    monthly_fee: float = Field(gt=0)
    currency: str = Field(pattern=CURRENCY_CODE_PATTERN)
    billing_period_start: date
    label: str | None = None


class UpdateSubscriptionAccountFeeRequest(BaseModel):
    subscription_account_id: str
    monthly_fee: float | None = Field(default=None, gt=0)
    currency: str | None = Field(default=None, pattern=CURRENCY_CODE_PATTERN)
    billing_period_start: date | None = None
    label: str | None = None


class DeleteSubscriptionAccountRequest(BaseModel):
    subscription_account_id: str


class BillingPeriod(BaseModel):
    start: date
    end: date


class SubscriptionAccountUsage(BaseModel):
    custom_llm_provider: str
    account_id: str | None
    deployments: list[str]
    fee: LiteLLM_SubscriptionAccountTable | None
    billing_periods: list[BillingPeriod]
    fixed_cost: float | None


class SubscriptionUsageResponse(BaseModel):
    start_date: date
    end_date: date
    subscription_providers: list[str]
    accounts: list[SubscriptionAccountUsage]
