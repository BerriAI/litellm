# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union, Optional
from typing_extensions import Annotated, TypeAlias

from ....._utils import PropertyInfo
from ....._models import BaseModel
from .tool_call_delta_object import ToolCallDeltaObject
from .run_step_delta_message_delta import RunStepDeltaMessageDelta

__all__ = ["RunStepDelta", "StepDetails"]

StepDetails: TypeAlias = Annotated[
    Union[RunStepDeltaMessageDelta, ToolCallDeltaObject],
    PropertyInfo(discriminator="type"),
]


class RunStepDelta(BaseModel):
    step_details: Optional[StepDetails] = None
    """The details of the run step."""
