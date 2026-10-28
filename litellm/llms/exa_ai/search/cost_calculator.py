from collections.abc import Mapping
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.constants import INTERNAL_KWARG_PREFIX

EXA_AI_COST_DOLLARS_PARAM: Final = f"{INTERNAL_KWARG_PREFIX}exa_ai_cost_dollars"


class _ExaCostDollars(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


def exa_ai_reported_cost(optional_params: Mapping[str, object]) -> float | None:
    """The `costDollars.total` Exa reported for the request, when it is a finite non-negative number."""
    try:
        return _ExaCostDollars.model_validate(optional_params.get(EXA_AI_COST_DOLLARS_PARAM)).total
    except ValidationError:
        return None
