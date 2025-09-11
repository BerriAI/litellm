# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .chat_completion_allowed_tools_param import ChatCompletionAllowedToolsParam

__all__ = ["ChatCompletionAllowedToolChoiceParam"]


class ChatCompletionAllowedToolChoiceParam(TypedDict, total=False):
    allowed_tools: Required[ChatCompletionAllowedToolsParam]
    """Constrains the tools available to the model to a pre-defined set."""

    type: Required[Literal["allowed_tools"]]
    """Allowed tool configuration type. Always `allowed_tools`."""
