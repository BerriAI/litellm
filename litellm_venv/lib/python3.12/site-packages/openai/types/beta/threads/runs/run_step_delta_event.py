# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ....._models import BaseModel
from .run_step_delta import RunStepDelta

__all__ = ["RunStepDeltaEvent"]


class RunStepDeltaEvent(BaseModel):
    id: str
    """The identifier of the run step, which can be referenced in API endpoints."""

    delta: RunStepDelta
    """The delta containing the fields that have changed on the run step."""

    object: Literal["thread.run.step.delta"]
    """The object type, which is always `thread.run.step.delta`."""
