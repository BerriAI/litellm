from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

AppBudgetScope: TypeAlias = Literal["app", "app_customer_org", "app_customer_member"]
CustomerBudgetScope: TypeAlias = Literal["customer_org", "customer_team", "customer_member"] | AppBudgetScope
CustomerBudgetEventType: TypeAlias = Literal["soft_budget_crossed", "budget_crossed"]


class CustomerBudgetEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event: CustomerBudgetEventType
    scope: CustomerBudgetScope
    entity_id: str
    app_id: str | None = None
    app_alias: str | None = None
    budget_duration: str | None = None
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
