# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List
from typing_extensions import Required, TypedDict

__all__ = ["UploadCompleteParams"]


class UploadCompleteParams(TypedDict, total=False):
    part_ids: Required[List[str]]
    """The ordered list of Part IDs."""

    md5: str
    """
    The optional md5 checksum for the file contents to verify if the bytes uploaded
    matches what you expect.
    """
