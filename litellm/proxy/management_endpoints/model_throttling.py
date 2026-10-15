from typing import Final

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.model_throttling import ModelThrottling, model_throttling_budget_error


class _BudgetState(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    max_budget: float | None = None
    model_throttling: ModelThrottling | None = None


def validate_model_throttling_update(data: BaseModel, existing: object | None = None) -> None:
    if not {"max_budget", "model_throttling"}.intersection(data.model_fields_set):
        return
    previous: Final = _BudgetState.model_validate(existing) if existing is not None else _BudgetState()
    requested: Final = _BudgetState.model_validate(data)
    config: Final = (
        requested.model_throttling if "model_throttling" in data.model_fields_set else previous.model_throttling
    )
    max_budget: Final = requested.max_budget if "max_budget" in data.model_fields_set else previous.max_budget
    error: Final = model_throttling_budget_error(config, max_budget)
    if error is not None:
        raise HTTPException(status_code=400, detail={"error": error})


def check_model_throttling_permission(data: BaseModel, is_proxy_admin: bool) -> None:
    if "model_throttling" in data.model_fields_set and not is_proxy_admin:
        raise HTTPException(status_code=403, detail={"error": "Only proxy admins can change model throttling."})
