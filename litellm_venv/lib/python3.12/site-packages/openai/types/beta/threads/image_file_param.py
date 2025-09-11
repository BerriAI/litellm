# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ImageFileParam"]


class ImageFileParam(TypedDict, total=False):
    file_id: Required[str]
    """
    The [File](https://platform.openai.com/docs/api-reference/files) ID of the image
    in the message content. Set `purpose="vision"` when uploading the File if you
    need to later display the file content.
    """

    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image if specified by the user.

    `low` uses fewer tokens, you can opt in to high resolution using `high`.
    """
