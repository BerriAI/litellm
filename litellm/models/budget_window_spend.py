"""
Budget window spend model.

The two columns of ``litellm_budgetwindowspend`` that budget enforcement reads: when the window started and the
spend recorded in it so far.
"""

from datetime import datetime

from pydantic import ConfigDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class BudgetWindowSpend(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    window_start: datetime
    spend: float
