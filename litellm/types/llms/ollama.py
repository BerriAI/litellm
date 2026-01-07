import json
from typing import Any, List, Optional, Union

from pydantic import BaseModel
from typing_extensions import (
    Protocol,
    Required,
    Self,
    TypedDict,
    TypeGuard,
    get_origin,
    override,
    runtime_checkable,
)


class OllamaToolCallFunction(
    TypedDict
):  # follows - https://github.com/ollama/ollama/blob/6bd8a4b0a1ac15d5718f52bbe1cd56f827beb694/api/types.go#L148
    name: str
    arguments: dict


class OllamaToolCall(TypedDict):
    function: OllamaToolCallFunction


class OllamaVisionModelObject(TypedDict):
    prompt: str
    images: List[str]


class OllamaChatCompletionMessage(TypedDict, total=False):
    role: Required[str]
    content: str
    thinking: str
    images: List[str]
    tool_calls: List[OllamaToolCall]
    tool_name: str
