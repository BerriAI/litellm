# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, TypedDict

__all__ = ["PermissionRetrieveParams"]


class PermissionRetrieveParams(TypedDict, total=False):
    after: str
    """Identifier for the last permission ID from the previous pagination request."""

    limit: int
    """Number of permissions to retrieve."""

    order: Literal["ascending", "descending"]
    """The order in which to retrieve permissions."""

    project_id: str
    """The ID of the project to get permissions for."""
