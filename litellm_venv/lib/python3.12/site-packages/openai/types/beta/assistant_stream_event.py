# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union, Optional
from typing_extensions import Literal, Annotated, TypeAlias

from .thread import Thread
from ..._utils import PropertyInfo
from ..._models import BaseModel
from .threads.run import Run
from .threads.message import Message
from ..shared.error_object import ErrorObject
from .threads.runs.run_step import RunStep
from .threads.message_delta_event import MessageDeltaEvent
from .threads.runs.run_step_delta_event import RunStepDeltaEvent

__all__ = [
    "AssistantStreamEvent",
    "ThreadCreated",
    "ThreadRunCreated",
    "ThreadRunQueued",
    "ThreadRunInProgress",
    "ThreadRunRequiresAction",
    "ThreadRunCompleted",
    "ThreadRunIncomplete",
    "ThreadRunFailed",
    "ThreadRunCancelling",
    "ThreadRunCancelled",
    "ThreadRunExpired",
    "ThreadRunStepCreated",
    "ThreadRunStepInProgress",
    "ThreadRunStepDelta",
    "ThreadRunStepCompleted",
    "ThreadRunStepFailed",
    "ThreadRunStepCancelled",
    "ThreadRunStepExpired",
    "ThreadMessageCreated",
    "ThreadMessageInProgress",
    "ThreadMessageDelta",
    "ThreadMessageCompleted",
    "ThreadMessageIncomplete",
    "ErrorEvent",
]


class ThreadCreated(BaseModel):
    data: Thread
    """
    Represents a thread that contains
    [messages](https://platform.openai.com/docs/api-reference/messages).
    """

    event: Literal["thread.created"]

    enabled: Optional[bool] = None
    """Whether to enable input audio transcription."""


class ThreadRunCreated(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.created"]


class ThreadRunQueued(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.queued"]


class ThreadRunInProgress(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.in_progress"]


class ThreadRunRequiresAction(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.requires_action"]


class ThreadRunCompleted(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.completed"]


class ThreadRunIncomplete(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.incomplete"]


class ThreadRunFailed(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.failed"]


class ThreadRunCancelling(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.cancelling"]


class ThreadRunCancelled(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.cancelled"]


class ThreadRunExpired(BaseModel):
    data: Run
    """
    Represents an execution run on a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.run.expired"]


class ThreadRunStepCreated(BaseModel):
    data: RunStep
    """Represents a step in execution of a run."""

    event: Literal["thread.run.step.created"]


class ThreadRunStepInProgress(BaseModel):
    data: RunStep
    """Represents a step in execution of a run."""

    event: Literal["thread.run.step.in_progress"]


class ThreadRunStepDelta(BaseModel):
    data: RunStepDeltaEvent
    """Represents a run step delta i.e.

    any changed fields on a run step during streaming.
    """

    event: Literal["thread.run.step.delta"]


class ThreadRunStepCompleted(BaseModel):
    data: RunStep
    """Represents a step in execution of a run."""

    event: Literal["thread.run.step.completed"]


class ThreadRunStepFailed(BaseModel):
    data: RunStep
    """Represents a step in execution of a run."""

    event: Literal["thread.run.step.failed"]


class ThreadRunStepCancelled(BaseModel):
    data: RunStep
    """Represents a step in execution of a run."""

    event: Literal["thread.run.step.cancelled"]


class ThreadRunStepExpired(BaseModel):
    data: RunStep
    """Represents a step in execution of a run."""

    event: Literal["thread.run.step.expired"]


class ThreadMessageCreated(BaseModel):
    data: Message
    """
    Represents a message within a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.message.created"]


class ThreadMessageInProgress(BaseModel):
    data: Message
    """
    Represents a message within a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.message.in_progress"]


class ThreadMessageDelta(BaseModel):
    data: MessageDeltaEvent
    """Represents a message delta i.e.

    any changed fields on a message during streaming.
    """

    event: Literal["thread.message.delta"]


class ThreadMessageCompleted(BaseModel):
    data: Message
    """
    Represents a message within a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.message.completed"]


class ThreadMessageIncomplete(BaseModel):
    data: Message
    """
    Represents a message within a
    [thread](https://platform.openai.com/docs/api-reference/threads).
    """

    event: Literal["thread.message.incomplete"]


class ErrorEvent(BaseModel):
    data: ErrorObject

    event: Literal["error"]


AssistantStreamEvent: TypeAlias = Annotated[
    Union[
        ThreadCreated,
        ThreadRunCreated,
        ThreadRunQueued,
        ThreadRunInProgress,
        ThreadRunRequiresAction,
        ThreadRunCompleted,
        ThreadRunIncomplete,
        ThreadRunFailed,
        ThreadRunCancelling,
        ThreadRunCancelled,
        ThreadRunExpired,
        ThreadRunStepCreated,
        ThreadRunStepInProgress,
        ThreadRunStepDelta,
        ThreadRunStepCompleted,
        ThreadRunStepFailed,
        ThreadRunStepCancelled,
        ThreadRunStepExpired,
        ThreadMessageCreated,
        ThreadMessageInProgress,
        ThreadMessageDelta,
        ThreadMessageCompleted,
        ThreadMessageIncomplete,
        ErrorEvent,
    ],
    PropertyInfo(discriminator="event"),
]
