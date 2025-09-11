# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from .._models import BaseModel
from .batch_error import BatchError
from .shared.metadata import Metadata
from .batch_request_counts import BatchRequestCounts

__all__ = ["Batch", "Errors"]


class Errors(BaseModel):
    data: Optional[List[BatchError]] = None

    object: Optional[str] = None
    """The object type, which is always `list`."""


class Batch(BaseModel):
    id: str

    completion_window: str
    """The time frame within which the batch should be processed."""

    created_at: int
    """The Unix timestamp (in seconds) for when the batch was created."""

    endpoint: str
    """The OpenAI API endpoint used by the batch."""

    input_file_id: str
    """The ID of the input file for the batch."""

    object: Literal["batch"]
    """The object type, which is always `batch`."""

    status: Literal[
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
    ]
    """The current status of the batch."""

    cancelled_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch was cancelled."""

    cancelling_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch started cancelling."""

    completed_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch was completed."""

    error_file_id: Optional[str] = None
    """The ID of the file containing the outputs of requests with errors."""

    errors: Optional[Errors] = None

    expired_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch expired."""

    expires_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch will expire."""

    failed_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch failed."""

    finalizing_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch started finalizing."""

    in_progress_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch started processing."""

    metadata: Optional[Metadata] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    output_file_id: Optional[str] = None
    """The ID of the file containing the outputs of successfully executed requests."""

    request_counts: Optional[BatchRequestCounts] = None
    """The request counts for different statuses within the batch."""
