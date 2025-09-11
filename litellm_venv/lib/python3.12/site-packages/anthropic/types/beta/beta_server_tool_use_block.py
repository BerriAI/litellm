# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["BetaServerToolUseBlock"]


class BetaServerToolUseBlock(BaseModel):
    id: str

    input: object

    name: Literal["web_search", "code_execution"]

    type: Literal["server_tool_use"]
