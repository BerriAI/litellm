"""Budget fields auth already resolved, carried on the request so success logging does no object lookups.

Auth pins one snapshot per entity on ``UserAPIKeyAuth`` (request-scoped, never cached), the pre-call
setup writes them into the request metadata under the aliased ``user_api_key_*`` names, and a logger
reads them back with ``from_metadata``. ``None`` means this request never carried that entity
(unauthenticated route, custom auth, budget check skipped) and the logger keeps its own lookup.
"""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import Self


class _BudgetSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    def metadata_entries(self) -> Mapping[str, object]:
        return MappingProxyType(self.model_dump(by_alias=True, mode="json"))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Self | None:
        try:
            return cls.model_validate(metadata)
        except ValidationError:
            return None


class KeyBudgetSnapshot(_BudgetSnapshot):
    """Read-only view of the ``user_api_key_budget_reset_at`` entry ``add_user_api_key_auth_to_request_metadata`` writes."""

    budget_reset_at: datetime | None = Field(
        validation_alias="user_api_key_budget_reset_at", serialization_alias="user_api_key_budget_reset_at"
    )


class TeamBudgetSnapshot(_BudgetSnapshot):
    budget_reset_at: datetime | None = Field(
        validation_alias="user_api_key_team_budget_reset_at", serialization_alias="user_api_key_team_budget_reset_at"
    )
    max_budget: float | None = Field(
        validation_alias="user_api_key_team_table_max_budget", serialization_alias="user_api_key_team_table_max_budget"
    )


class UserBudgetSnapshot(_BudgetSnapshot):
    budget_reset_at: datetime | None = Field(
        validation_alias="user_api_key_user_budget_reset_at", serialization_alias="user_api_key_user_budget_reset_at"
    )
    max_budget: float | None = Field(
        validation_alias="user_api_key_user_table_max_budget", serialization_alias="user_api_key_user_table_max_budget"
    )
    user_alias: str | None = Field(
        validation_alias="user_api_key_user_alias", serialization_alias="user_api_key_user_alias"
    )


class OrgBudgetSnapshot(_BudgetSnapshot):
    spend: float = Field(validation_alias="user_api_key_org_spend", serialization_alias="user_api_key_org_spend")
    max_budget: float | None = Field(
        validation_alias="user_api_key_org_max_budget", serialization_alias="user_api_key_org_max_budget"
    )
