from typing_extensions import ReadOnly, TypedDict

from ..utils import CompletionTokensDetails, PromptTokensDetailsWrapper, ServerToolUse


class UsagePerChunk(TypedDict):
    prompt_tokens: ReadOnly[int | None]
    completion_tokens: ReadOnly[int | None]
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    server_tool_use: ServerToolUse | None
    web_search_requests: int | None
    google_maps_grounding_requests: ReadOnly[int | None]
    completion_tokens_details: CompletionTokensDetails | None
    prompt_tokens_details: PromptTokensDetailsWrapper | None
    cost: float | None
    inference_geo: str | None
    speed: str | None
