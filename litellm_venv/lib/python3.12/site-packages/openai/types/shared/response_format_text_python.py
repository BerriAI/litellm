# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseFormatTextPython"]


class ResponseFormatTextPython(BaseModel):
    type: Literal["python"]
    """The type of response format being defined. Always `python`."""
