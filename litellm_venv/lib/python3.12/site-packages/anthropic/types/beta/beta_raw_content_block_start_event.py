# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Literal, Annotated, TypeAlias

from ..._utils import PropertyInfo
from ..._models import BaseModel
from .beta_text_block import BetaTextBlock
from .beta_thinking_block import BetaThinkingBlock
from .beta_tool_use_block import BetaToolUseBlock
from .beta_mcp_tool_use_block import BetaMCPToolUseBlock
from .beta_mcp_tool_result_block import BetaMCPToolResultBlock
from .beta_server_tool_use_block import BetaServerToolUseBlock
from .beta_container_upload_block import BetaContainerUploadBlock
from .beta_redacted_thinking_block import BetaRedactedThinkingBlock
from .beta_web_search_tool_result_block import BetaWebSearchToolResultBlock
from .beta_code_execution_tool_result_block import BetaCodeExecutionToolResultBlock

__all__ = ["BetaRawContentBlockStartEvent", "ContentBlock"]

ContentBlock: TypeAlias = Annotated[
    Union[
        BetaTextBlock,
        BetaToolUseBlock,
        BetaServerToolUseBlock,
        BetaWebSearchToolResultBlock,
        BetaCodeExecutionToolResultBlock,
        BetaMCPToolUseBlock,
        BetaMCPToolResultBlock,
        BetaContainerUploadBlock,
        BetaThinkingBlock,
        BetaRedactedThinkingBlock,
    ],
    PropertyInfo(discriminator="type"),
]


class BetaRawContentBlockStartEvent(BaseModel):
    content_block: ContentBlock
    """Response model for a file uploaded to the container."""

    index: int

    type: Literal["content_block_start"]
