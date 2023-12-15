# File generated from our OpenAPI spec by Stainless.

from typing import List
from typing_extensions import Literal

from .._models import BaseModel
from .completion_usage import CompletionUsage

__all__ = ["Edit", "Choice"]


class Choice(BaseModel):
    finish_reason: Literal["stop", "length"]
    """The reason the model stopped generating tokens.

    This will be `stop` if the model hit a natural stop point or a provided stop
    sequence, `length` if the maximum number of tokens specified in the request was
    reached, or `content_filter` if content was omitted due to a flag from our
    content filters.
    """

    index: int
    """The index of the choice in the list of choices."""

    text: str
    """The edited result."""


class Edit(BaseModel):
    choices: List[Choice]
    """A list of edit choices. Can be more than one if `n` is greater than 1."""

    created: int
    """The Unix timestamp (in seconds) of when the edit was created."""

    object: Literal["edit"]
    """The object type, which is always `edit`."""

    usage: CompletionUsage
    """Usage statistics for the completion request."""
