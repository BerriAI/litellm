from typing_extensions import TypedDict

from ..utils import CompletionTokensDetails, PromptTokensDetailsWrapper, ServerToolUse


class UsagePerChunk(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    server_tool_use: ServerToolUse | None
    web_search_requests: int | None
    completion_tokens_details: CompletionTokensDetails | None
    prompt_tokens_details: PromptTokensDetailsWrapper | None
    cost: float | None
