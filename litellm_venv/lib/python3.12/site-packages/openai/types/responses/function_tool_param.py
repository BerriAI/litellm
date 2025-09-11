# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Optional
from typing_extensions import Literal, Required, TypedDict

__all__ = ["FunctionToolParam"]


class FunctionToolParam(TypedDict, total=False):
    name: Required[str]
    """The name of the function to call."""

    parameters: Required[Optional[Dict[str, object]]]
    """A JSON schema object describing the parameters of the function."""

    strict: Required[Optional[bool]]
    """Whether to enforce strict parameter validation. Default `true`."""

    type: Required[Literal["function"]]
    """The type of the function tool. Always `function`."""

    description: Optional[str]
    """A description of the function.

    Used by the model to determine whether or not to call the function.
    """
