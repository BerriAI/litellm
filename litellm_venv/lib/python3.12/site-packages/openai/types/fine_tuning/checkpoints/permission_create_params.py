# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List
from typing_extensions import Required, TypedDict

__all__ = ["PermissionCreateParams"]


class PermissionCreateParams(TypedDict, total=False):
    project_ids: Required[List[str]]
    """The project identifiers to grant access to."""
