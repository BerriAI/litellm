from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import anthropic
import httpx
from typing_extensions import TypeAlias


class AnthropicMessagesRequestParams(TypedDict, total=False):
    """
    Anthropic Messages API Request Params: https://docs.anthropic.com/en/api/messages
    """

    max_tokens: int
    messages: List[Dict]
    model: str
    metadata: Optional[Dict[str, Any]]
    stop_sequences: Optional[List[str]]
    stream: Literal[False]
    system: Optional[str]
    temperature: float
    thinking: Optional[Dict[str, Any]]
    tool_choice: Optional[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    top_k: Optional[int]
    top_p: Optional[float]
    timeout: Optional[Union[float, httpx.Timeout]]
