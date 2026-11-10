from typing_extensions import (
    NotRequired,
    ReadOnly,
    Required,
    TypedDict,
)


class OllamaToolCallFunction(
    TypedDict
):  # follows - https://github.com/ollama/ollama/blob/6bd8a4b0a1ac15d5718f52bbe1cd56f827beb694/api/types.go#L148
    name: str
    arguments: dict


class OllamaToolCall(TypedDict):
    id: NotRequired[ReadOnly[str]]
    function: OllamaToolCallFunction


class OllamaVisionModelObject(TypedDict):
    prompt: str
    images: list[str]


class OllamaChatCompletionMessage(TypedDict, total=False):
    role: Required[str]
    content: str
    thinking: str
    images: list[str]
    tool_calls: list[OllamaToolCall]
    tool_name: str
    tool_call_id: str
