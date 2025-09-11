# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Union, Optional
from typing_extensions import Literal, Annotated, TypeAlias

from ....._utils import PropertyInfo
from ....._models import BaseModel
from .code_interpreter_logs import CodeInterpreterLogs
from .code_interpreter_output_image import CodeInterpreterOutputImage

__all__ = ["CodeInterpreterToolCallDelta", "CodeInterpreter", "CodeInterpreterOutput"]

CodeInterpreterOutput: TypeAlias = Annotated[
    Union[CodeInterpreterLogs, CodeInterpreterOutputImage],
    PropertyInfo(discriminator="type"),
]


class CodeInterpreter(BaseModel):
    input: Optional[str] = None
    """The input to the Code Interpreter tool call."""

    outputs: Optional[List[CodeInterpreterOutput]] = None
    """The outputs from the Code Interpreter tool call.

    Code Interpreter can output one or more items, including text (`logs`) or images
    (`image`). Each of these are represented by a different object type.
    """


class CodeInterpreterToolCallDelta(BaseModel):
    index: int
    """The index of the tool call in the tool calls array."""

    type: Literal["code_interpreter"]
    """The type of tool call.

    This is always going to be `code_interpreter` for this type of tool call.
    """

    id: Optional[str] = None
    """The ID of the tool call."""

    code_interpreter: Optional[CodeInterpreter] = None
    """The Code Interpreter tool call definition."""
