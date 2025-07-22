import json
from typing import Any, List, Literal, Optional, TypedDict, Union

from typing_extensions import (
    TYPE_CHECKING,
    Protocol,
    Required,
    Self,
    TypeGuard,
    get_origin,
    override,
    runtime_checkable,
)

from .openai import ChatCompletionToolCallChunk


class CachePointBlock(TypedDict, total=False):
    type: Literal["default"]


class SystemContentBlock(TypedDict, total=False):
    text: str
    cachePoint: CachePointBlock


class SourceBlock(TypedDict):
    bytes: Optional[str]  # base 64 encoded string


BedrockImageTypes = Literal["png", "jpeg", "gif", "webp"]


class ImageBlock(TypedDict):
    format: Union[BedrockImageTypes, str]
    source: SourceBlock


BedrockVideoTypes = Literal["mp4", "mov", "mkv", "webm", "flv", "mpeg", "mpg", "wmv", "3gp"]


class VideoBlock(TypedDict):
    format: Union[BedrockVideoTypes, str]
    source: SourceBlock


BedrockDocumentTypes = Literal[
    "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"
]


class DocumentBlock(TypedDict):
    format: Union[BedrockDocumentTypes, str]
    source: SourceBlock
    name: str


class ToolResultContentBlock(TypedDict, total=False):
    image: ImageBlock
    document: DocumentBlock
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


class BedrockConverseReasoningTextBlock(TypedDict, total=False):
    text: Required[str]
    signature: str


class BedrockConverseReasoningContentBlock(TypedDict, total=False):
    reasoningText: BedrockConverseReasoningTextBlock
    redactedContent: str


class BedrockConverseReasoningContentBlockDelta(TypedDict, total=False):
    signature: str
    redactedContent: str
    text: str


class ContentBlock(TypedDict, total=False):
    text: str
    image: ImageBlock
    video: VideoBlock
    document: DocumentBlock
    toolResult: ToolResultBlock
    toolUse: ToolUseBlock
    cachePoint: CachePointBlock
    reasoningContent: BedrockConverseReasoningContentBlock


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
    cacheReadInputTokenCount: int
    cacheReadInputTokens: int
    cacheWriteInputTokenCount: int
    cacheWriteInputTokens: int


class ConverseResponseBlock(TypedDict):
    additionalModelResponseFields: dict
    metrics: ConverseMetricsBlock
    output: ConverseResponseOutputBlock
    stopReason: (
        str  # end_turn | tool_use | max_tokens | stop_sequence | content_filtered
    )
    usage: ConverseTokenUsageBlock


class ToolJsonSchemaBlock(TypedDict, total=False):
    type: Literal["object"]
    properties: dict
    required: List[str]


class ToolInputSchemaBlock(TypedDict):
    json: Optional[ToolJsonSchemaBlock]


class ToolSpecBlock(TypedDict, total=False):
    inputSchema: Required[ToolInputSchemaBlock]
    name: Required[str]
    description: str


class ToolBlock(TypedDict, total=False):
    toolSpec: Optional[ToolSpecBlock]
    cachePoint: Optional[CachePointBlock]


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
    topK: int


class ToolBlockDeltaEvent(TypedDict):
    input: str


class ToolUseBlockStartEvent(TypedDict):
    name: str
    toolUseId: str


class ContentBlockStartEvent(TypedDict, total=False):
    toolUse: Optional[ToolUseBlockStartEvent]
    reasoningContent: BedrockConverseReasoningContentBlockDelta


class ContentBlockDeltaEvent(TypedDict, total=False):
    """
    Either 'text' or 'toolUse' will be specified for Converse API streaming response.
    """

    text: str
    toolUse: ToolBlockDeltaEvent
    reasoningContent: BedrockConverseReasoningContentBlockDelta


class PerformanceConfigBlock(TypedDict):
    latency: Literal["optimized", "throughput"]


class CommonRequestObject(
    TypedDict, total=False
):  # common request object across sync + async flows
    additionalModelRequestFields: dict
    additionalModelResponseFieldPaths: List[str]
    inferenceConfig: InferenceConfig
    system: List[SystemContentBlock]
    toolConfig: ToolConfigBlock
    guardrailConfig: Optional[GuardrailConfigBlock]
    performanceConfig: Optional[PerformanceConfigBlock]


class RequestObject(CommonRequestObject, total=False):
    messages: Required[List[MessageBlock]]


class BedrockInvokeNovaRequest(TypedDict, total=False):
    """
    Request object for sending `nova` requests to `/bedrock/invoke/`
    """

    messages: List[MessageBlock]
    inferenceConfig: InferenceConfig
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


class AmazonNovaCanvasRequestBase(TypedDict, total=False):
    """
    Base class for Amazon Nova Canvas API requests
    """

    pass


class AmazonNovaCanvasImageGenerationConfig(TypedDict, total=False):
    """
    Config for Amazon Nova Canvas Text to Image API

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
    """

    cfgScale: int
    seed: int
    quality: Literal["standard", "premium"]
    width: int
    height: int
    numberOfImages: int


class AmazonNovaCanvasTextToImageParams(TypedDict, total=False):
    """
    Params for Amazon Nova Canvas Text to Image API
    """

    text: str
    negativeText: str
    controlStrength: float
    controlMode: Literal["CANNY_EDIT", "SEGMENTATION"]
    conditionImage: str


class AmazonNovaCanvasTextToImageRequest(
    AmazonNovaCanvasRequestBase, TypedDict, total=False
):
    """
    Request for Amazon Nova Canvas Text to Image API

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
    """

    textToImageParams: AmazonNovaCanvasTextToImageParams
    taskType: Literal["TEXT_IMAGE"]
    imageGenerationConfig: AmazonNovaCanvasImageGenerationConfig


class AmazonNovaCanvasColorGuidedGenerationParams(TypedDict, total=False):
    """
    Params for Amazon Nova Canvas Color Guided Generation API
    """

    colors: List[str]
    referenceImage: str
    text: str
    negativeText: str


class AmazonNovaCanvasColorGuidedRequest(
    AmazonNovaCanvasRequestBase, TypedDict, total=False
):
    """
    Request for Amazon Nova Canvas Color Guided Generation API

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
    """

    taskType: Literal["COLOR_GUIDED_GENERATION"]
    colorGuidedGenerationParams: AmazonNovaCanvasColorGuidedGenerationParams
    imageGenerationConfig: AmazonNovaCanvasImageGenerationConfig


class AmazonNovaCanvasTextToImageResponse(TypedDict, total=False):
    """
    Response for Amazon Nova Canvas Text to Image API

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
    """

    images: List[str]


if TYPE_CHECKING:
    from botocore.awsrequest import AWSPreparedRequest
else:
    AWSPreparedRequest = Any

from pydantic import BaseModel


class BedrockPreparedRequest(TypedDict):
    """
    Internal/Helper class for preparing the request for bedrock image generation
    """

    endpoint_url: str
    prepped: AWSPreparedRequest
    body: bytes
    data: dict


class BedrockRerankTextQuery(TypedDict):
    text: str


class BedrockRerankQuery(TypedDict):
    textQuery: BedrockRerankTextQuery
    type: Literal["TEXT"]


class BedrockRerankModelConfiguration(TypedDict, total=False):
    modelArn: Required[str]
    modelConfiguration: dict


class BedrockRerankBedrockRerankingConfiguration(TypedDict):
    modelConfiguration: BedrockRerankModelConfiguration
    numberOfResults: int


class BedrockRerankConfiguration(TypedDict):
    bedrockRerankingConfiguration: BedrockRerankBedrockRerankingConfiguration
    type: Literal["BEDROCK_RERANKING_MODEL"]


class BedrockRerankTextDocument(TypedDict, total=False):
    text: str


class BedrockRerankInlineDocumentSource(TypedDict, total=False):
    jsonDocument: dict
    textDocument: BedrockRerankTextDocument
    type: Literal["TEXT", "JSON"]


class BedrockRerankSource(TypedDict):
    inlineDocumentSource: BedrockRerankInlineDocumentSource
    type: Literal["INLINE"]


class BedrockRerankRequest(TypedDict):
    """
    Request for Bedrock Rerank API
    """

    queries: List[BedrockRerankQuery]
    rerankingConfiguration: BedrockRerankConfiguration
    sources: List[BedrockRerankSource]


class AmazonDeepSeekR1StreamingResponse(TypedDict):
    generation: str
    generation_token_count: int
    stop_reason: Optional[str]
    prompt_token_count: int
