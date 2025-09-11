# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Union, Optional
from typing_extensions import Required, TypeAlias, TypedDict

from .response_input_file_param import ResponseInputFileParam
from .response_input_text_param import ResponseInputTextParam
from .response_input_image_param import ResponseInputImageParam

__all__ = ["ResponsePromptParam", "Variables"]

Variables: TypeAlias = Union[
    str, ResponseInputTextParam, ResponseInputImageParam, ResponseInputFileParam
]


class ResponsePromptParam(TypedDict, total=False):
    id: Required[str]
    """The unique identifier of the prompt template to use."""

    variables: Optional[Dict[str, Variables]]
    """Optional map of values to substitute in for variables in your prompt.

    The substitution values can either be strings, or other Response input types
    like images or files.
    """

    version: Optional[str]
    """Optional version of the prompt template."""
