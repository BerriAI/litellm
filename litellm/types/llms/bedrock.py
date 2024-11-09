import json
from typing import Any, List, Literal, Optional, TypedDict, Union

from typing_extensions import (
    Protocol,
    Required,
    Self,
    TypeGuard,
    get_origin,
    override,
    runtime_checkable,
)

from .openai import ChatCompletionToolCallChunk


class SystemContentBlock(TypedDict):
    text: str


class ImageSourceBlock(TypedDict):
    bytes: Optional[str]  # base 64 encoded string


class ImageBlock(TypedDict):
    format: Literal["png", "jpeg", "gif", "webp"]
    source: ImageSourceBlock


class ToolResultContentBlock(TypedDict, total=False):
    image: ImageBlock
    json: dict
    text: str


class ToolResultBlock(TypedDict, total=False):
    content: Required[List[ToolResultContentBlock]]
    toolUseId: Required[str]
    status: Literal["success", "error"]


class ToolUseBlock(TypedDict):
    input: dict
    name: str
    toolUseId: str


class ContentBlock(TypedDict, total=False):
    text: str
    image: ImageBlock
    toolResult: ToolResultBlock
    toolUse: ToolUseBlock


class MessageBlock(TypedDict):
    content: List[ContentBlock]
    role: Literal["user", "assistant"]


class ConverseMetricsBlock(TypedDict):
    latencyMs: float  # time in ms


class ConverseResponseOutputBlock(TypedDict):
    message: Optional[MessageBlock]


class ConverseTokenUsageBlock(TypedDict):
    inputTokens: int
    outputTokens: int
    totalTokens: int


class ConverseResponseBlock(TypedDict):
    additionalModelResponseFields: dict
    metrics: ConverseMetricsBlock
    output: ConverseResponseOutputBlock
    stopReason: (
        str  # end_turn | tool_use | max_tokens | stop_sequence | content_filtered
    )
    usage: ConverseTokenUsageBlock


class ToolInputSchemaBlock(TypedDict):
    json: Optional[dict]


class ToolSpecBlock(TypedDict, total=False):
    inputSchema: Required[ToolInputSchemaBlock]
    name: Required[str]
    description: str


class ToolBlock(TypedDict):
    toolSpec: Optional[ToolSpecBlock]


class SpecificToolChoiceBlock(TypedDict):
    name: str


class ToolChoiceValuesBlock(TypedDict, total=False):
    any: dict
    auto: dict
    tool: SpecificToolChoiceBlock


class ToolConfigBlock(TypedDict, total=False):
    tools: Required[List[ToolBlock]]
    toolChoice: Union[str, ToolChoiceValuesBlock]


class GuardrailConfigBlock(TypedDict, total=False):
    guardrailIdentifier: str
    guardrailVersion: str
    trace: Literal["enabled", "disabled"]


class InferenceConfig(TypedDict, total=False):
    maxTokens: int
    stopSequences: List[str]
    temperature: float
    topP: float


class ToolBlockDeltaEvent(TypedDict):
    input: str


class ToolUseBlockStartEvent(TypedDict):
    name: str
    toolUseId: str


class ContentBlockStartEvent(TypedDict, total=False):
    toolUse: Optional[ToolUseBlockStartEvent]


class ContentBlockDeltaEvent(TypedDict, total=False):
    """
    Either 'text' or 'toolUse' will be specified for Converse API streaming response.
    """

    text: str
    toolUse: ToolBlockDeltaEvent


class RequestObject(TypedDict, total=False):
    additionalModelRequestFields: dict
    additionalModelResponseFieldPaths: List[str]
    inferenceConfig: InferenceConfig
    messages: Required[List[MessageBlock]]
    system: List[SystemContentBlock]
    toolConfig: ToolConfigBlock
    guardrailConfig: Optional[GuardrailConfigBlock]


class GenericStreamingChunk(TypedDict):
    text: Required[str]
    tool_use: Optional[ChatCompletionToolCallChunk]
    is_finished: Required[bool]
    finish_reason: Required[str]
    usage: Optional[ConverseTokenUsageBlock]
    index: int


class Document(TypedDict):
    title: str
    snippet: str


class ServerSentEvent:
    def __init__(
        self,
        *,
        event: Optional[str] = None,
        data: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> None:
        if data is None:
            data = ""

        self._id = id
        self._data = data
        self._event = event or None
        self._retry = retry

    @property
    def event(self) -> Optional[str]:
        return self._event

    @property
    def id(self) -> Optional[str]:
        return self._id

    @property
    def retry(self) -> Optional[int]:
        return self._retry

    @property
    def data(self) -> str:
        return self._data

    def json(self) -> Any:
        return json.loads(self.data)

    @override
    def __repr__(self) -> str:
        return f"ServerSentEvent(event={self.event}, data={self.data}, id={self.id}, retry={self.retry})"


COHERE_EMBEDDING_INPUT_TYPES = Literal[
    "search_document", "search_query", "classification", "clustering", "image"
]


class CohereEmbeddingRequest(TypedDict, total=False):
    texts: List[str]
    images: List[str]
    input_type: Required[COHERE_EMBEDDING_INPUT_TYPES]
    truncate: Literal["NONE", "START", "END"]
    embedding_types: Literal["float", "int8", "uint8", "binary", "ubinary"]


class CohereEmbeddingRequestWithModel(CohereEmbeddingRequest):
    model: Required[str]


class CohereEmbeddingResponse(TypedDict):
    embeddings: List[List[float]]
    id: str
    response_type: Literal["embedding_floats"]
    texts: List[str]


class AmazonTitanV2EmbeddingRequest(TypedDict):
    inputText: str
    dimensions: int
    normalize: bool


class AmazonTitanV2EmbeddingResponse(TypedDict):
    embedding: List[float]
    inputTextTokenCount: int


class AmazonTitanG1EmbeddingRequest(TypedDict):
    inputText: str


class AmazonTitanG1EmbeddingResponse(TypedDict):
    embedding: List[float]
    inputTextTokenCount: int


class AmazonTitanMultimodalEmbeddingConfig(TypedDict):
    outputEmbeddingLength: Literal[256, 384, 1024]


class AmazonTitanMultimodalEmbeddingRequest(TypedDict, total=False):
    inputText: str
    inputImage: str
    embeddingConfig: AmazonTitanMultimodalEmbeddingConfig


class AmazonTitanMultimodalEmbeddingResponse(TypedDict):
    embedding: List[float]
    inputTextTokenCount: int
    message: str  # Specifies any errors that occur during generation.


AmazonEmbeddingRequest = Union[
    AmazonTitanMultimodalEmbeddingRequest,
    AmazonTitanV2EmbeddingRequest,
    AmazonTitanG1EmbeddingRequest,
]


class AmazonStability3TextToImageRequest(TypedDict, total=False):
    """
    Request for Amazon Stability 3 Text to Image API

    Ref here: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-text-image.html
    """

    prompt: str
    aspect_ratio: Literal[
        "16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"
    ]
    mode: Literal["image-to-image", "text-to-image"]
    output_format: Literal["JPEG", "PNG"]
    seed: int
    negative_prompt: str


class AmazonStability3TextToImageResponse(TypedDict, total=False):
    """
    Response for Amazon Stability 3 Text to Image API

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-text-image.html
    """

    images: List[str]
    seeds: List[str]
    finish_reasons: List[str]
