from typing import Any, Optional

from pydantic import Field, model_validator

from litellm.types.utils import BudgetConfig, GenericBudgetConfigType, LiteLLMPydanticObjectBase


class ConfigTeamEntry(LiteLLMPydanticObjectBase):
    """Declarative team definition from proxy YAML `teams:` list."""

    team_id: Optional[str] = None
    team_alias: Optional[str] = None
    team_member_budget: Optional[float] = None
    team_member_budget_duration: Optional[str] = None
    model_max_budget: Optional[GenericBudgetConfigType] = None
    models: Optional[list] = None
    max_budget: Optional[float] = Field(
        default=None,
        description=(
            "Alias for team_member_budget. Config teams never use a shared team pool; "
            "max_budget is remapped to a per-user/SA member cap."
        ),
    )
    budget_duration: Optional[str] = Field(
        default=None,
        description="Alias for team_member_budget_duration when max_budget is used as the member cap.",
    )
    max_budget_was_aliased: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def alias_max_budget_to_member_budget(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        max_budget = data.get("max_budget")
        if max_budget is None:
            return data
        aliased = dict(data)
        aliased["max_budget_was_aliased"] = True
        if aliased.get("team_member_budget") is None:
            aliased["team_member_budget"] = max_budget
        if aliased.get("budget_duration") is not None and aliased.get("team_member_budget_duration") is None:
            aliased["team_member_budget_duration"] = aliased.get("budget_duration")
        aliased["max_budget"] = None
        aliased["budget_duration"] = None
        return aliased

    @model_validator(mode="after")
    def require_team_id_or_alias(self) -> "ConfigTeamEntry":
        if not self.team_id and not self.team_alias:
            raise ValueError("each teams: entry requires team_id or team_alias")
        return self


def normalize_budget_config_dict(raw: dict) -> dict[str, BudgetConfig]:
    return {
        model_name: BudgetConfig(**config) if not isinstance(config, BudgetConfig) else config
        for model_name, config in raw.items()
    }
