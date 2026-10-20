from typing import Literal

from typing_extensions import ReadOnly, TypedDict


class FunctionCall(TypedDict):
    name: str | None
    arguments: str | dict | None


class MistralToolCallMessage(TypedDict):
    id: str | None
    type: Literal["function"]
    function: FunctionCall | None


class MistralTextBlock(TypedDict):
    type: Literal["text"]
    text: str


class MistralThinkingBlock(TypedDict):
    type: Literal["thinking"]
    thinking: list[MistralTextBlock]


class MistralRequestRouting(TypedDict, total=False):
    api_base: ReadOnly[object]
    drop_params: ReadOnly[object]
    allowed_openai_params: ReadOnly[object]
    custom_llm_provider: ReadOnly[object]


class MistralCacheKeyParams(TypedDict, total=False):
    prompt_cache_key: ReadOnly[object]


class MistralReasoningEffortParams(TypedDict, total=False):
    reasoning_effort: ReadOnly[object]


class MistralFilePart(TypedDict, total=False):
    file_id: ReadOnly[object]
