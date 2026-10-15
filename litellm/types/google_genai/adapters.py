from collections.abc import Mapping

from openai.types.chat import ChatCompletionStreamOptionsParam
from typing_extensions import ReadOnly, TypedDict

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolChoiceStringValues,
    ChatCompletionToolParam,
)


class GenerateContentCompletionKwargs(TypedDict, total=False):
    model: str
    messages: list[AllMessageValues]
    temperature: float
    max_tokens: int
    top_p: float
    stop: str | list[str]
    tools: list[ChatCompletionToolParam]
    tool_choice: ChatCompletionToolChoiceStringValues
    stream: bool
    metadata: dict[str, object]
    litellm_metadata: ReadOnly[Mapping[str, object]]
    stream_options: ReadOnly[ChatCompletionStreamOptionsParam]
    extra_headers: dict[str, str] | None
    proxy_server_request: dict[str, object] | None
