# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["MessageDeleted"]


class MessageDeleted(BaseModel):
    id: str

    deleted: bool

    object: Literal["thread.message.deleted"]
