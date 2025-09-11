# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Literal, TypeAlias

from .assistant_tool_choice import AssistantToolChoice

__all__ = ["AssistantToolChoiceOption"]

AssistantToolChoiceOption: TypeAlias = Union[
    Literal["none", "auto", "required"], AssistantToolChoice
]
