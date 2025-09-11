# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["PermissionDeleteResponse"]


class PermissionDeleteResponse(BaseModel):
    id: str
    """The ID of the fine-tuned model checkpoint permission that was deleted."""

    deleted: bool
    """Whether the fine-tuned model checkpoint permission was successfully deleted."""

    object: Literal["checkpoint.permission"]
    """The object type, which is always "checkpoint.permission"."""
