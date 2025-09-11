# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List
from typing_extensions import Literal, Required, TypedDict

__all__ = ["ContainerCreateParams", "ExpiresAfter"]


class ContainerCreateParams(TypedDict, total=False):
    name: Required[str]
    """Name of the container to create."""

    expires_after: ExpiresAfter
    """Container expiration time in seconds relative to the 'anchor' time."""

    file_ids: List[str]
    """IDs of files to copy to the container."""


class ExpiresAfter(TypedDict, total=False):
    anchor: Required[Literal["last_active_at"]]
    """Time anchor for the expiration time.

    Currently only 'last_active_at' is supported.
    """

    minutes: Required[int]
