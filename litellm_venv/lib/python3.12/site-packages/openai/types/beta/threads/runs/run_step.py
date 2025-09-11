# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union, Optional
from typing_extensions import Literal, Annotated, TypeAlias

from ....._utils import PropertyInfo
from ....._models import BaseModel
from ....shared.metadata import Metadata
from .tool_calls_step_details import ToolCallsStepDetails
from .message_creation_step_details import MessageCreationStepDetails

__all__ = ["RunStep", "LastError", "StepDetails", "Usage"]


class LastError(BaseModel):
    code: Literal["server_error", "rate_limit_exceeded"]
    """One of `server_error` or `rate_limit_exceeded`."""

    message: str
    """A human-readable description of the error."""


StepDetails: TypeAlias = Annotated[
    Union[MessageCreationStepDetails, ToolCallsStepDetails],
    PropertyInfo(discriminator="type"),
]


class Usage(BaseModel):
    completion_tokens: int
    """Number of completion tokens used over the course of the run step."""

    prompt_tokens: int
    """Number of prompt tokens used over the course of the run step."""

    total_tokens: int
    """Total number of tokens used (prompt + completion)."""


class RunStep(BaseModel):
    id: str
    """The identifier of the run step, which can be referenced in API endpoints."""

    assistant_id: str
    """
    The ID of the
    [assistant](https://platform.openai.com/docs/api-reference/assistants)
    associated with the run step.
    """

    cancelled_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the run step was cancelled."""

    completed_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the run step completed."""

    created_at: int
    """The Unix timestamp (in seconds) for when the run step was created."""

    expired_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the run step expired.

    A step is considered expired if the parent run is expired.
    """

    failed_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the run step failed."""

    last_error: Optional[LastError] = None
    """The last error associated with this run step.

    Will be `null` if there are no errors.
    """

    metadata: Optional[Metadata] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    object: Literal["thread.run.step"]
    """The object type, which is always `thread.run.step`."""

    run_id: str
    """
    The ID of the [run](https://platform.openai.com/docs/api-reference/runs) that
    this run step is a part of.
    """

    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"]
    """
    The status of the run step, which can be either `in_progress`, `cancelled`,
    `failed`, `completed`, or `expired`.
    """

    step_details: StepDetails
    """The details of the run step."""

    thread_id: str
    """
    The ID of the [thread](https://platform.openai.com/docs/api-reference/threads)
    that was run.
    """

    type: Literal["message_creation", "tool_calls"]
    """The type of run step, which can be either `message_creation` or `tool_calls`."""

    usage: Optional[Usage] = None
    """Usage statistics related to the run step.

    This value will be `null` while the run step's status is `in_progress`.
    """
