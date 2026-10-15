from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from litellm.proxy._types import (
    BudgetNewRequest,
    GenerateKeyRequest,
    UpdateKeyRequest,
    UpdateTeamRequest,
    UpdateUserRequest,
)
from litellm.proxy.management_endpoints.model_throttling import (
    check_model_throttling_permission,
    validate_model_throttling_update,
)
from litellm.types.proxy.model_throttling import ModelThrottling, ModelThrottlingThreshold

CONFIG: Final = ModelThrottling(thresholds=(ModelThrottlingThreshold(spend=60, models=("premium",)),))


@pytest.mark.parametrize("request_type", (UpdateKeyRequest, UpdateTeamRequest, UpdateUserRequest, BudgetNewRequest))
def test_partial_budget_edits_cannot_leave_threshold_above_final_cap(request_type: type[BaseModel]) -> None:
    existing: Final = {"max_budget": 100, "model_throttling": CONFIG}
    invalid: Final = request_type.model_validate({"key": "key", "team_id": "team", "user_id": "user", "max_budget": 50})
    with pytest.raises(HTTPException, match="below the final budget") as error:
        validate_model_throttling_update(invalid, existing)
    assert error.value.status_code == 400
    cleared: Final = request_type.model_validate(
        {"key": "key", "team_id": "team", "user_id": "user", "max_budget": None, "model_throttling": None}
    )
    assert validate_model_throttling_update(cleared, existing) is None
    unchanged: Final = request_type.model_validate({"key": "key", "team_id": "team", "user_id": "user"})
    assert validate_model_throttling_update(unchanged, existing) is None


@pytest.mark.parametrize("limit", (None, 0, -1, 60))
def test_throttling_requires_a_later_positive_final_budget(limit: float | None) -> None:
    request: Final = GenerateKeyRequest(max_budget=limit, model_throttling=CONFIG)
    with pytest.raises(HTTPException) as error:
        validate_model_throttling_update(request)
    assert error.value.status_code == 400


def test_only_admins_can_change_or_clear_throttling() -> None:
    for config in (CONFIG, None):
        request = UpdateKeyRequest(key="key", model_throttling=config)
        with pytest.raises(HTTPException) as error:
            check_model_throttling_permission(request, False)
        assert error.value.status_code == 403
        assert check_model_throttling_permission(request, True) is None
    assert check_model_throttling_permission(UpdateKeyRequest(key="key"), False) is None


@pytest.mark.parametrize("amount", (0, -1, float("nan"), float("inf")))
def test_invalid_threshold_amount_is_rejected(amount: float) -> None:
    with pytest.raises(ValidationError):
        ModelThrottlingThreshold(spend=amount, models=("premium",))


@pytest.mark.parametrize("models", ((), ("",), ("  ",)))
def test_empty_model_selection_is_rejected(models: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        ModelThrottlingThreshold(spend=60, models=models)


@pytest.mark.parametrize("second", (ModelThrottlingThreshold(spend=60, models=("standard",)), CONFIG.thresholds[0]))
def test_duplicate_thresholds_or_models_are_rejected(second: ModelThrottlingThreshold) -> None:
    with pytest.raises(ValidationError):
        ModelThrottling(thresholds=(*CONFIG.thresholds, second))
