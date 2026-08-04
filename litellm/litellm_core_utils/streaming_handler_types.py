from collections.abc import Mapping, Sequence
from typing import Protocol, Union, runtime_checkable

from pydantic import BaseModel
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Function,
    ModelResponseStream,
)


class CompletionUsageLike(Protocol):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ParsedProviderChunk(TypedDict, total=False):
    text: ReadOnly[str | None]
    is_finished: ReadOnly[bool]
    finish_reason: ReadOnly[str | None]
    logprobs: ReadOnly[object]
    original_chunk: ReadOnly[ModelResponseStream | None]
    usage: ReadOnly[object]
    tool_use: ReadOnly[ChatCompletionToolCallChunk | None]
    tool_calls: ReadOnly[list[ChatCompletionDeltaToolCall] | None]
    provider_specific_fields: ReadOnly[dict[str, object] | None]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    index: ReadOnly[int]


class SimpleParsedChunk(ParsedProviderChunk):
    text: Required[ReadOnly[str | None]]
    is_finished: Required[ReadOnly[bool]]
    finish_reason: Required[ReadOnly[str | None]]


class TextCompletionParsedChunk(SimpleParsedChunk):
    usage: Required[ReadOnly[CompletionUsageLike | None]]


class OpenAIChatParsedChunk(SimpleParsedChunk):
    logprobs: Required[ReadOnly[object]]
    original_chunk: Required[ReadOnly[ModelResponseStream | None]]
    usage: Required[ReadOnly[object]]


class StreamingCompletionObj(TypedDict, total=False):
    content: Required[str | None]
    role: str
    tool_calls: list[ChatCompletionToolCallChunk] | list[ChatCompletionDeltaToolCall] | None
    function_call: ChatCompletionToolCallFunctionChunk | Function | None
    provider_specific_fields: dict[str, object] | None
    index: int


class TextCompletionChoiceLike(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def finish_reason(self) -> str | None: ...


@runtime_checkable
class HasTextChoices(Protocol):
    @property
    def choices(self) -> Sequence[TextCompletionChoiceLike]: ...


@runtime_checkable
class HasCompletionUsage(Protocol):
    usage: CompletionUsageLike | None


@runtime_checkable
class HasFunctionsAttr(Protocol):
    functions: object


@runtime_checkable
class HasStatusCode(Protocol):
    @property
    def status_code(self) -> int: ...


VertexArgValue = Union[
    str,
    int,
    float,
    bool,
    None,
    Sequence["VertexArgValue"],
    Mapping[str, "VertexArgValue"],
]


class VertexFunctionCallLike(Protocol):
    @property
    def args(self) -> Mapping[str, VertexArgValue]: ...

    @property
    def name(self) -> str: ...


class VertexPartLike(Protocol):
    @property
    def function_call(self) -> VertexFunctionCallLike: ...


class VertexContentLike(Protocol):
    @property
    def parts(self) -> Sequence[VertexPartLike]: ...


class VertexEnumNameLike(Protocol):
    @property
    def name(self) -> str: ...


class VertexCandidateLike(Protocol):
    @property
    def content(self) -> VertexContentLike: ...

    @property
    def finish_reason(self) -> VertexEnumNameLike: ...


@runtime_checkable
class VertexProtoChunkLike(Protocol):
    @property
    def candidates(self) -> Sequence[VertexCandidateLike]: ...


class VllmCompletionOutputLike(Protocol):
    @property
    def text(self) -> str: ...


@runtime_checkable
class VllmRequestOutputLike(Protocol):
    @property
    def outputs(self) -> Sequence[VllmCompletionOutputLike]: ...


@runtime_checkable
class HasModelAttr(Protocol):
    model: str | None


class PredibaseStreamToken(BaseModel):
    text: str | None = None


class PredibaseStreamDetails(BaseModel):
    finish_reason: str | None = None


class PredibaseStreamChunk(BaseModel):
    token: PredibaseStreamToken | None = None
    details: PredibaseStreamDetails | None = None
    generated_text: str | None = None
    error: str | None = None


class AI21StreamData(BaseModel):
    text: str


class AI21StreamCompletion(BaseModel):
    data: AI21StreamData


class AI21StreamChunk(BaseModel):
    completions: list[AI21StreamCompletion]


class MaritalkStreamChunk(BaseModel):
    answer: str


class NlpCloudStreamChunk(BaseModel):
    generated_text: str


class AlephAlphaStreamCompletion(BaseModel):
    completion: str


class AlephAlphaStreamChunk(BaseModel):
    completions: list[AlephAlphaStreamCompletion]


class AzureStreamDelta(BaseModel):
    content: str | None = ""


class AzureStreamChoice(BaseModel):
    delta: AzureStreamDelta | None = None
    finish_reason: str | None = None


class AzureStreamChunk(BaseModel):
    choices: list[AzureStreamChoice]


class BasetenStreamToken(BaseModel):
    text: str | None = None


class BasetenModelOutput(BaseModel):
    data: list[str] | None = None


class BasetenStreamChunk(BaseModel):
    token: BasetenStreamToken | None = None
    model_output: BasetenModelOutput | str | None = None
    completion: str | None = None


class TritonStreamChunk(BaseModel):
    text_output: str | None = ""
    stop_reason: str | None = None
    is_finished: bool = False
    input_token_count: int = 0
    generated_token_count: int = 0
