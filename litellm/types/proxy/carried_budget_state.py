"""Budget fields auth already resolved, carried on the request so success logging does no object lookups.

Auth pins one snapshot per entity on ``UserAPIKeyAuth`` (request-scoped, never cached), the pre-call
setup writes them into the request metadata under the aliased ``user_api_key_*`` names, and a logger
reads them back with ``from_metadata``. ``None`` means this request never carried that entity
(unauthenticated route, custom auth, budget check skipped) and the logger keeps its own lookup.
"""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final

from pydantic import ConfigDict, Field, ValidationError
from typing_extensions import Self

from litellm.types.llms.base import LiteLLMBaseModel

END_USER_COUNTER_PREFIX: Final = "spend:end_user:"


class _BudgetSnapshot(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    def metadata_entries(self) -> Mapping[str, object]:
        return MappingProxyType(self.model_dump(by_alias=True, mode="json"))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Self | None:
        try:
            return cls.model_validate(metadata, by_name=False)
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


class EntitySpendCounterKeys(LiteLLMBaseModel):
    """Spend counter of each entity whose remaining budget a gauge reports. ``None`` for an entity with no budget."""

    model_config = ConfigDict(frozen=True)

    key: str | None = None
    team: str | None = None
    user: str | None = None
    org: str | None = None
    end_user: str | None = None

    @classmethod
    def of(
        cls,
        token: str | None,
        team_id: str | None,
        user_id: str | None,
        org_id: str | None,
        end_user_id: str | None,
    ) -> Self:
        return cls(
            key=f"spend:key:{token}" if token else None,
            team=f"spend:team:{team_id}" if team_id else None,
            user=f"spend:user:{user_id}" if user_id else None,
            org=f"spend:org:{org_id}" if org_id else None,
            end_user=f"{END_USER_COUNTER_PREFIX}{end_user_id}" if end_user_id else None,
        )

    def tracked(self) -> tuple[str, ...]:
        counter_keys: Final = (self.key, self.team, self.user, self.org, self.end_user)
        return tuple(counter_key for counter_key in counter_keys if counter_key is not None)


class CommittedBudgetSpend(LiteLLMBaseModel):
    """Spend each entity was charged, budget holds excluded, read from the committed spend counters for the
    remaining budget gauges only. ``None`` when the entity has no committed counter. ``counter_keys`` names the
    budgeted entities whose committed counters this request charges and reads."""

    model_config = ConfigDict(frozen=True)

    counter_keys: EntitySpendCounterKeys = EntitySpendCounterKeys()
    key: float | None = None
    team: float | None = None
    user: float | None = None
    org: float | None = None
    end_user: float | None = None
