from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import ConfigDict

from litellm.types.llms.base_model import LiteLLMBaseModel

CustomerBudgetScope: TypeAlias = Literal["customer_org", "customer_team", "customer_member"]
CustomerBudgetEventType: TypeAlias = Literal["soft_budget_crossed", "budget_crossed"]


class CustomerBudgetEvent(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    event: CustomerBudgetEventType
    scope: CustomerBudgetScope
    entity_id: str
    customer_org_id: str | None = None
    customer_org_alias: str | None = None
    customer_team_id: str | None = None
    customer_team_alias: str | None = None
    customer_member_id: str | None = None
    customer_member_alias: str | None = None
    spend: float
    budget: float
    period: str
    timestamp: datetime
