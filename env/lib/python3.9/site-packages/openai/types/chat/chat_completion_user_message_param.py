# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

from typing import List, Union
from typing_extensions import Literal, Required, TypedDict

from .chat_completion_content_part_param import ChatCompletionContentPartParam

__all__ = ["ChatCompletionUserMessageParam"]


class ChatCompletionUserMessageParam(TypedDict, total=False):
    content: Required[Union[str, List[ChatCompletionContentPartParam], None]]
    """The contents of the user message."""

    role: Required[Literal["user"]]
    """The role of the messages author, in this case `user`."""
