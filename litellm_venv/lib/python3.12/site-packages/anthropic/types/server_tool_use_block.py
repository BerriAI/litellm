# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["ServerToolUseBlock"]


class ServerToolUseBlock(BaseModel):
    id: str

    input: object

    name: Literal["web_search"]

    type: Literal["server_tool_use"]
