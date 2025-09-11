# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ToolChoiceCustomParam"]


class ToolChoiceCustomParam(TypedDict, total=False):
    name: Required[str]
    """The name of the custom tool to call."""

    type: Required[Literal["custom"]]
    """For custom tool calling, the type is always `custom`."""
