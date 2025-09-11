# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .._models import BaseModel
from .web_search_tool_result_block_content import WebSearchToolResultBlockContent

__all__ = ["WebSearchToolResultBlock"]


class WebSearchToolResultBlock(BaseModel):
    content: WebSearchToolResultBlockContent

    tool_use_id: str

    type: Literal["web_search_tool_result"]
