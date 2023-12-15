# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, TypedDict

__all__ = ["ChatCompletionSystemMessageParam"]


class ChatCompletionSystemMessageParam(TypedDict, total=False):
    content: Required[Optional[str]]
    """The contents of the system message."""

    role: Required[Literal["system"]]
    """The role of the messages author, in this case `system`."""
