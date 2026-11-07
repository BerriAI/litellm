from datetime import date
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from litellm.models.subscription_account import LiteLLM_SubscriptionAccountTable

CURRENCY_CODE_PATTERN: Final = r"^[A-Z]{3}$"


class SubscriptionAccountFeeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    custom_llm_provider: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    monthly_fee: float = Field(gt=0, allow_inf_nan=False)
    currency: str = Field(pattern=CURRENCY_CODE_PATTERN)
    billing_period_start: date
    label: str | None = None


class UpdateSubscriptionAccountFeeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    subscription_account_id: str
    monthly_fee: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    currency: str | None = Field(default=None, pattern=CURRENCY_CODE_PATTERN)
    billing_period_start: date | None = None
    label: str | None = None


class DeleteSubscriptionAccountRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    subscription_account_id: str


class BillingPeriod(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: date
    end: date


class SubscriptionAccountUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    custom_llm_provider: str
    account_id: str | None
    deployments: tuple[str, ...]
    fee: LiteLLM_SubscriptionAccountTable | None
    billing_periods: tuple[BillingPeriod, ...]
    fixed_cost: float | None


class SubscriptionUsageResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_date: date
    end_date: date
    subscription_providers: tuple[str, ...]
    accounts: tuple[SubscriptionAccountUsage, ...]
