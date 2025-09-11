# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseFormatTextGrammar"]


class ResponseFormatTextGrammar(BaseModel):
    grammar: str
    """The custom grammar for the model to follow."""

    type: Literal["grammar"]
    """The type of response format being defined. Always `grammar`."""
