# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel
from ..shared.custom_tool_input_format import CustomToolInputFormat

__all__ = ["CustomTool"]


class CustomTool(BaseModel):
    name: str
    """The name of the custom tool, used to identify it in tool calls."""

    type: Literal["custom"]
    """The type of the custom tool. Always `custom`."""

    description: Optional[str] = None
    """Optional description of the custom tool, used to provide more context."""

    format: Optional[CustomToolInputFormat] = None
    """The input format for the custom tool. Default is unconstrained text."""
