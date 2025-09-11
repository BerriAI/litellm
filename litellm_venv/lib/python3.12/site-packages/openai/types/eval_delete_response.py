# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from .._models import BaseModel

__all__ = ["EvalDeleteResponse"]


class EvalDeleteResponse(BaseModel):
    deleted: bool

    eval_id: str

    object: str
