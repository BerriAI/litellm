# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ....._models import BaseModel

__all__ = ["FunctionToolCallDelta", "Function"]


class Function(BaseModel):
    arguments: Optional[str] = None
    """The arguments passed to the function."""

    name: Optional[str] = None
    """The name of the function."""

    output: Optional[str] = None
    """The output of the function.

    This will be `null` if the outputs have not been
    [submitted](https://platform.openai.com/docs/api-reference/runs/submitToolOutputs)
    yet.
    """


class FunctionToolCallDelta(BaseModel):
    index: int
    """The index of the tool call in the tool calls array."""

    type: Literal["function"]
    """The type of tool call.

    This is always going to be `function` for this type of tool call.
    """

    id: Optional[str] = None
    """The ID of the tool call object."""

    function: Optional[Function] = None
    """The definition of the function that was called."""
