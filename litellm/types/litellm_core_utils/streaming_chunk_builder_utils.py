from typing import TYPE_CHECKING, Optional

from typing_extensions import TypedDict

from ..utils import CompletionTokensDetails, PromptTokensDetailsWrapper, ServerToolUse


class UsagePerChunk(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]
    server_tool_use: Optional[ServerToolUse]
    web_search_requests: Optional[int]
    completion_tokens_details: Optional[CompletionTokensDetails]
    prompt_tokens_details: Optional[PromptTokensDetailsWrapper]
