# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from .._models import BaseModel

__all__ = ["BatchError"]


class BatchError(BaseModel):
    code: Optional[str] = None
    """An error code identifying the error type."""

    line: Optional[int] = None
    """The line number of the input file where the error occurred, if applicable."""

    message: Optional[str] = None
    """A human-readable message providing more details about the error."""

    param: Optional[str] = None
    """The name of the parameter that caused the error, if applicable."""
