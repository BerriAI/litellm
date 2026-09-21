from typing_extensions import TypedDict

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
    extra_headers: dict[str, str] | None
