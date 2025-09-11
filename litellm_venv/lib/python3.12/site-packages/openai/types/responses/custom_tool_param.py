# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from ..shared_params.custom_tool_input_format import CustomToolInputFormat

__all__ = ["CustomToolParam"]


class CustomToolParam(TypedDict, total=False):
    name: Required[str]
    """The name of the custom tool, used to identify it in tool calls."""

    type: Required[Literal["custom"]]
    """The type of the custom tool. Always `custom`."""

    description: str
    """Optional description of the custom tool, used to provide more context."""

    format: CustomToolInputFormat
    """The input format for the custom tool. Default is unconstrained text."""
