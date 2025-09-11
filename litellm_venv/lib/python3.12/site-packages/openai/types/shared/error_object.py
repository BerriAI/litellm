# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from ..._models import BaseModel

__all__ = ["ErrorObject"]


class ErrorObject(BaseModel):
    code: Optional[str] = None

    message: str

    param: Optional[str] = None

    type: str
