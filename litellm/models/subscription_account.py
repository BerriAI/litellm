"""
Subscription account table model.

Canonical definition for ``litellm_subscriptionaccounttable``: the operator-entered
recurring fee of a provider account whose requests are covered by a flat subscription.
"""

from datetime import datetime

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_SubscriptionAccountTable(LiteLLMPydanticObjectBase):
    subscription_account_id: str
    custom_llm_provider: str
    account_id: str
    label: str | None = None
    monthly_fee: float
    currency: str
    billing_period_start: str
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
