from datetime import datetime
from typing import Annotated, TypeAlias

from pydantic import Field

from litellm.constants import LENS_FEEDBACK_MAX_COMMENT_CHARS, LENS_FEEDBACK_MAX_SCORE
from litellm.proxy.lens.models import Record, TraceIdentity

FeedbackScore: TypeAlias = Annotated[int, Field(ge=0, le=LENS_FEEDBACK_MAX_SCORE)]


class FeedbackInput(Record):
    score: FeedbackScore
    comment: str = Field(default="", max_length=LENS_FEEDBACK_MAX_COMMENT_CHARS)
    user: str = Field(default="", max_length=256)


class Feedback(TraceIdentity):
    score: FeedbackScore
    comment: str
    author: str
    created_at: datetime
    updated_at: datetime


class TraceFeedback(TraceIdentity):
    feedback: tuple[Feedback, ...]


class TraceFeedbackSummary(TraceIdentity):
    count: int = Field(ge=0)
    average: float | None = Field(ge=0, le=LENS_FEEDBACK_MAX_SCORE)
    lowest: FeedbackScore | None


class TraceFeedbackRequest(Record):
    traces: tuple[TraceIdentity, ...] = Field(min_length=1, max_length=500)
