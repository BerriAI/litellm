import json
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Literal

from typing_extensions import ReadOnly, Required, TypedDict, override

from .openai import ChatCompletionToolCallChunk


class CachePointBlock(TypedDict, total=False):
    type: Literal["default"]
    ttl: str


class SystemContentBlock(TypedDict, total=False):
    text: str
    cachePoint: CachePointBlock


class SourceBlock(TypedDict):
    bytes: str | None  # base 64 encoded string


BedrockImageTypes = Literal["png", "jpeg", "gif", "webp"]


class ImageBlock(TypedDict):
    format: BedrockImageTypes | str
    source: SourceBlock


BedrockVideoTypes = Literal["mp4", "mov", "mkv", "webm", "flv", "mpeg", "mpg", "wmv", "3gp"]


class VideoBlock(TypedDict):
    format: BedrockVideoTypes | str
    source: SourceBlock


BedrockDocumentTypes = Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]


class DocumentBlock(TypedDict):
    format: BedrockDocumentTypes | str
    source: SourceBlock
    name: str


class SearchResultBlock(TypedDict, total=False):
    """
    Search result block used in Bedrock toolResult content.

    Reference:
    https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SearchResultBlock.html
    """

    source: str
    title: str
    content: list[dict]
    citations: dict


class ToolResultContentBlock(TypedDict, total=False):
    image: ImageBlock
    document: DocumentBlock
    searchResult: SearchResultBlock
    json: dict
    text: str


class ToolResultBlock(TypedDict, total=False):
    content: Required[list[ToolResultContentBlock]]
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


class BedrockConverseGptReasoningEffortBlock(TypedDict):
    effort: ReadOnly[str]


class GuardrailConverseTextBlock(TypedDict, total=False):
    text: str


class GuardrailConverseContentBlock(TypedDict, total=False):
    """Content block for selective guardrail evaluation in Bedrock Converse API"""

    text: GuardrailConverseTextBlock


class CitationWebLocationBlock(TypedDict, total=False):
    """
    Web location block for Nova grounding citations.
    Contains the URL and domain from web search results.

    Reference: https://docs.aws.amazon.com/nova/latest/userguide/grounding.html
    """

    url: str
    domain: str


class CitationSearchResultLocationBlock(TypedDict, total=False):
    """
    Character span of a Nova grounding citation within the cited content,
    plus the index of the search result it refers to.
    """

    start: int
    end: int
    searchResultIndex: int


class CitationLocationBlock(TypedDict, total=False):
    """
    Location block describing where a citation points to.
    """

    web: CitationWebLocationBlock
    searchResultLocation: CitationSearchResultLocationBlock


class CitationReferenceBlock(TypedDict, total=False):
    """
    Citation reference block containing a single citation with its location,
    source URL and title.
    """

    location: CitationLocationBlock
    source: str
    title: str


class CitationGeneratedContentBlock(TypedDict, total=False):
    """A piece of generated text associated with a citationsContent block."""

    text: str


class CitationsContentBlock(TypedDict, total=False):
    """
    Citations content block returned by Nova grounding (web search) tool.

    When Nova grounding is enabled via systemTool, the model may return
    citationsContent blocks containing the grounded text and its citation
    references.

    Reference: https://docs.aws.amazon.com/nova/latest/userguide/grounding.html

    Example response structure:
        {
            "citationsContent": {
                "content": [{"text": "The grounded answer text ..."}],
                "citations": [
                    {
                        "location": {
                            "searchResultLocation": {
                                "start": 0,
                                "end": 42,
                                "searchResultIndex": 0
                            }
                        },
                        "source": "https://example.com/article",
                        "title": "Example Article"
                    }
                ]
            }
        }
    """

    content: list[CitationGeneratedContentBlock]
    citations: list[CitationReferenceBlock]


class ContentBlock(TypedDict, total=False):
    text: str
    image: ImageBlock
    video: VideoBlock
    document: DocumentBlock
    toolResult: ToolResultBlock
    toolUse: ToolUseBlock
    cachePoint: CachePointBlock
    reasoningContent: BedrockConverseReasoningContentBlock
    guardContent: GuardrailConverseContentBlock
    citationsContent: CitationsContentBlock


class MessageBlock(TypedDict):
    content: list[ContentBlock]
    role: Literal["user", "assistant"]


class ConverseMetricsBlock(TypedDict):
    latencyMs: float  # time in ms


class ConverseResponseOutputBlock(TypedDict):
    message: MessageBlock | None


class CacheDetailBlock(TypedDict):
    """Per-TTL cache-write breakdown, read-only AWS response data. https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CacheDetail.html"""

    inputTokens: ReadOnly[int]
    ttl: ReadOnly[Literal["5m", "1h"]]


class ConverseTokenUsageBlock(TypedDict, total=False):
    inputTokens: Required[ReadOnly[int]]
    outputTokens: Required[ReadOnly[int]]
    totalTokens: Required[ReadOnly[int]]
    cacheReadInputTokenCount: ReadOnly[int]
    cacheReadInputTokens: ReadOnly[int]
    cacheWriteInputTokenCount: ReadOnly[int]
    cacheWriteInputTokens: ReadOnly[int]
    cacheDetails: ReadOnly[list[CacheDetailBlock]]  # mutable-ok: AWS response array, never mutated after parsing


class ServiceTierBlock(TypedDict):
    type: Literal["priority", "default", "flex"]


class ConverseResponseBlock(TypedDict, total=False):
    additionalModelResponseFields: dict
    metrics: ConverseMetricsBlock
    output: Required[ConverseResponseOutputBlock]
    stopReason: Required[str]  # end_turn | tool_use | max_tokens | stop_sequence | content_filtered
    usage: Required[ConverseTokenUsageBlock]
    serviceTier: ServiceTierBlock  # Optional - only present when serviceTier was sent in request


class ToolJsonSchemaBlock(TypedDict, total=False):
    type: Literal["object"]
    properties: dict
    required: list[str]
    additionalProperties: bool


class ToolInputSchemaBlock(TypedDict):
    json: ToolJsonSchemaBlock | None


class ToolSpecBlock(TypedDict, total=False):
    inputSchema: Required[ToolInputSchemaBlock]
    name: Required[str]
    description: str
    strict: bool


class SystemToolBlock(TypedDict, total=False):
    """
    System tool block for Nova grounding and other built-in tools.

    Example:
        {
            "systemTool": {
                "name": "nova_grounding"
            }
        }
    """

    name: Required[str]


class ToolBlock(TypedDict, total=False):
    toolSpec: ToolSpecBlock | None
    systemTool: SystemToolBlock | None
    cachePoint: CachePointBlock | None


class BedrockToolSpec(dict):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict,
        strict: bool | None,
        supports_strict_tools: bool,
    ) -> None:
        json_schema: Final[ToolJsonSchemaBlock] = {
            "type": parameters["type"],
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
        }
        additional_properties: Final = parameters.get("additionalProperties")
        if supports_strict_tools and additional_properties is not None:
            json_schema["additionalProperties"] = additional_properties

        tool_spec: Final[ToolSpecBlock] = {
            "inputSchema": {"json": json_schema},
            "name": name,
            "description": description,
        }
        if supports_strict_tools and strict:
            tool_spec["strict"] = strict

        super().__init__(toolSpec=tool_spec)


class SpecificToolChoiceBlock(TypedDict):
    name: str


class ToolChoiceValuesBlock(TypedDict, total=False):
    any: dict
    auto: dict
    tool: SpecificToolChoiceBlock


class ToolConfigBlock(TypedDict, total=False):
    tools: Required[list[ToolBlock]]
    toolChoice: str | ToolChoiceValuesBlock


class GuardrailConfigBlock(TypedDict, total=False):
    guardrailIdentifier: str
    guardrailVersion: str
    trace: Literal["enabled", "disabled", "enabled_full"]


class InferenceConfig(TypedDict, total=False):
    maxTokens: int
    stopSequences: list[str]
    temperature: float
    topP: float
    topK: int


class ToolBlockDeltaEvent(TypedDict):
    input: str


class ToolUseBlockStartEvent(TypedDict):
    name: str
    toolUseId: str


class ContentBlockStartEvent(TypedDict, total=False):
    toolUse: ToolUseBlockStartEvent | None
    reasoningContent: BedrockConverseReasoningContentBlockDelta


class ContentBlockDeltaEvent(TypedDict, total=False):
    """
    Either 'text' or 'toolUse' will be specified for Converse API streaming response.
    May also include 'citationsContent' when Nova grounding is enabled.
    """

    text: str
    toolUse: ToolBlockDeltaEvent
    reasoningContent: BedrockConverseReasoningContentBlockDelta
    citationsContent: CitationsContentBlock


class PerformanceConfigBlock(TypedDict):
    latency: Literal["optimized", "throughput"]


class JsonSchemaDefinition(TypedDict, total=False):
    """JSON schema structured output format options for Bedrock Converse API."""

    schema: Required[str]  # JSON string, not dict
    name: str
    description: str


class OutputFormatStructure(TypedDict, total=False):
    """The structure that the model's output must adhere to (union type)."""

    jsonSchema: Required[JsonSchemaDefinition]


class OutputFormat(TypedDict):
    """Structured output parameters to control the model's response."""

    type: Literal["json_schema"]
    structure: OutputFormatStructure


class OutputConfigBlock(TypedDict, total=False):
    """Output configuration for a model response in Converse/ConverseStream."""

    textFormat: OutputFormat


class CommonRequestObject(TypedDict, total=False):  # common request object across sync + async flows
    additionalModelRequestFields: dict
    additionalModelResponseFieldPaths: Sequence[str]
    inferenceConfig: InferenceConfig
    system: list[SystemContentBlock]
    toolConfig: ToolConfigBlock
    guardrailConfig: GuardrailConfigBlock | None
    performanceConfig: PerformanceConfigBlock | None
    serviceTier: ServiceTierBlock | None
    requestMetadata: dict[str, str] | None
    outputConfig: OutputConfigBlock | None


class RequestObject(CommonRequestObject, total=False):
    messages: Required[list[MessageBlock]]


class BedrockInvokeNovaRequest(TypedDict, total=False):
    """
    Request object for sending `nova` requests to `/bedrock/invoke/`
    """

    messages: list[MessageBlock]
    inferenceConfig: InferenceConfig
    system: list[SystemContentBlock]
    toolConfig: ToolConfigBlock
    guardrailConfig: GuardrailConfigBlock | None


class GenericStreamingChunk(TypedDict):
    text: Required[str]
    tool_use: ChatCompletionToolCallChunk | None
    is_finished: Required[bool]
    finish_reason: Required[str]
    usage: ConverseTokenUsageBlock | None
    index: int


class Document(TypedDict):
    title: str
    snippet: str


class ServerSentEvent:
    def __init__(
        self,
        *,
        event: str | None = None,
        data: str | None = None,
        id: str | None = None,
        retry: int | None = None,
    ) -> None:
        if data is None:
            data = ""

        self._id = id
        self._data = data
        self._event = event or None
        self._retry = retry

    @property
    def event(self) -> str | None:
        return self._event

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def retry(self) -> int | None:
        return self._retry

    @property
    def data(self) -> str:
        return self._data

    def json(self) -> Any:
        return json.loads(self.data)

    @override
    def __repr__(self) -> str:
        return f"ServerSentEvent(event={self.event}, data={self.data}, id={self.id}, retry={self.retry})"


COHERE_EMBEDDING_INPUT_TYPES = Literal["search_document", "search_query", "classification", "clustering", "image"]


class CohereEmbeddingRequest(TypedDict, total=False):
    texts: list[str]
    images: list[str]
    input_type: Required[COHERE_EMBEDDING_INPUT_TYPES]
    truncate: Literal["NONE", "START", "END"]
    embedding_types: Literal["float", "int8", "uint8", "binary", "ubinary"]
    output_dimension: int


class CohereEmbeddingRequestWithModel(CohereEmbeddingRequest):
    model: Required[str]


class CohereEmbeddingResponse(TypedDict):
    embeddings: list[list[float]]
    id: str
    response_type: Literal["embedding_floats"]
    texts: list[str]


class AmazonTitanV2EmbeddingRequest(TypedDict, total=False):
    inputText: Required[str]
    dimensions: int
    normalize: bool
    embeddingTypes: list[Literal["float", "binary"]]


class AmazonTitanV2EmbeddingsByType(TypedDict, total=False):
    binary: list[int]  # Array of integers for binary format
    float: list[float]  # Array of floats for float format


class AmazonTitanV2EmbeddingResponse(TypedDict, total=False):
    embedding: list[float]  # Legacy field - array of floats (backward compatibility)
    embeddingsByType: AmazonTitanV2EmbeddingsByType  # New format per AWS schema
    inputTextTokenCount: Required[int]  # Always present in AWS response


class AmazonTitanG1EmbeddingRequest(TypedDict):
    inputText: str


class AmazonTitanG1EmbeddingResponse(TypedDict):
    embedding: list[float]
    inputTextTokenCount: int


class AmazonTitanMultimodalEmbeddingConfig(TypedDict):
    outputEmbeddingLength: Literal[256, 384, 1024]


class AmazonTitanMultimodalEmbeddingRequest(TypedDict, total=False):
    inputText: str
    inputImage: str
    embeddingConfig: AmazonTitanMultimodalEmbeddingConfig


class AmazonTitanMultimodalEmbeddingResponse(TypedDict):
    embedding: list[float]
    inputTextTokenCount: int
    message: str  # Specifies any errors that occur during generation.


# TwelveLabs Marengo Embed 2.7 types
TWELVELABS_EMBEDDING_INPUT_TYPES = Literal["text", "image", "video", "audio"]
TWELVELABS_EMBEDDING_OPTIONS = Literal["visual-text", "visual-image", "audio"]


class TwelveLabsS3Location(TypedDict, total=False):
    uri: str
    bucketOwner: str


class TwelveLabsMediaSource(TypedDict, total=False):
    base64String: str
    s3Location: TwelveLabsS3Location


class TwelveLabsMarengoEmbeddingRequest(TypedDict, total=False):
    inputType: Required[TWELVELABS_EMBEDDING_INPUT_TYPES]
    inputText: str
    mediaSource: TwelveLabsMediaSource
    textTruncate: Literal["end", "none"]
    startSec: float
    lengthSec: float
    useFixedLengthSec: float
    minClipSec: int
    embeddingOption: list[TWELVELABS_EMBEDDING_OPTIONS]


class TwelveLabsMarengoEmbeddingResponse(TypedDict):
    embedding: list[float]
    embeddingOption: TWELVELABS_EMBEDDING_OPTIONS
    startSec: float
    endSec: float


class TwelveLabsS3OutputDataConfig(TypedDict):
    s3Uri: str


class TwelveLabsOutputDataConfig(TypedDict):
    s3OutputDataConfig: TwelveLabsS3OutputDataConfig


class TwelveLabsAsyncInvokeRequest(TypedDict):
    modelId: str
    modelInput: TwelveLabsMarengoEmbeddingRequest
    outputDataConfig: TwelveLabsOutputDataConfig


class TwelveLabsAsyncInvokeStatusResponse(TypedDict):
    invocationArn: str
    modelArn: str
    status: str  # "InProgress" | "Completed" | "Failed"
    submitTime: str
    lastModifiedTime: str
    endTime: str | None
    outputDataConfig: TwelveLabsOutputDataConfig
    clientRequestToken: str | None
    failureMessage: str | None


# Amazon Nova Multimodal Embeddings types
NOVA_EMBEDDING_PURPOSES = Literal[
    "GENERIC_INDEX",
    "GENERIC_RETRIEVAL",
    "TEXT_RETRIEVAL",
    "IMAGE_RETRIEVAL",
    "VIDEO_RETRIEVAL",
    "DOCUMENT_RETRIEVAL",
    "AUDIO_RETRIEVAL",
    "CLASSIFICATION",
    "CLUSTERING",
]

NOVA_EMBEDDING_DIMENSIONS = Literal[256, 384, 1024, 3072]

NOVA_TRUNCATION_MODES = Literal["START", "END", "NONE"]

NOVA_DETAIL_LEVELS = Literal["STANDARD_IMAGE", "DOCUMENT_IMAGE"]

NOVA_EMBEDDING_MODES = Literal["AUDIO_VIDEO_COMBINED", "AUDIO_VIDEO_SEPARATE"]

NOVA_EMBEDDING_TYPES = Literal["TEXT", "IMAGE", "VIDEO", "AUDIO", "AUDIO_VIDEO_COMBINED"]


class NovaSourceS3Location(TypedDict):
    uri: str


class NovaSourceObject(TypedDict, total=False):
    bytes: str  # base64 encoded
    s3Location: NovaSourceS3Location


class NovaTextParams(TypedDict, total=False):
    truncationMode: NOVA_TRUNCATION_MODES
    value: str
    source: NovaSourceObject


class NovaImageParams(TypedDict, total=False):
    format: str  # png, jpeg, gif, webp
    source: Required[NovaSourceObject]
    detailLevel: NOVA_DETAIL_LEVELS


class NovaVideoParams(TypedDict, total=False):
    format: str  # mp4, mov, mkv, webm, flv, mpeg, mpg, wmv, 3gp
    source: Required[NovaSourceObject]
    embeddingMode: Required[NOVA_EMBEDDING_MODES]


class NovaAudioParams(TypedDict, total=False):
    format: str  # mp3, wav, ogg
    source: Required[NovaSourceObject]


class NovaTextSegmentationConfig(TypedDict, total=False):
    maxLengthChars: int  # 800-50,000, default 32,000


class NovaMediaSegmentationConfig(TypedDict, total=False):
    durationSeconds: int  # 1-30, default 5


class NovaTextParamsWithSegmentation(NovaTextParams, total=False):
    segmentationConfig: NovaTextSegmentationConfig


class NovaVideoParamsWithSegmentation(NovaVideoParams, total=False):
    segmentationConfig: NovaMediaSegmentationConfig


class NovaAudioParamsWithSegmentation(NovaAudioParams, total=False):
    segmentationConfig: NovaMediaSegmentationConfig


class NovaSingleEmbeddingParams(TypedDict, total=False):
    embeddingPurpose: Required[NOVA_EMBEDDING_PURPOSES]
    embeddingDimension: NOVA_EMBEDDING_DIMENSIONS
    text: NovaTextParams
    image: NovaImageParams
    video: NovaVideoParams
    audio: NovaAudioParams


class NovaSegmentedEmbeddingParams(TypedDict, total=False):
    embeddingPurpose: Required[NOVA_EMBEDDING_PURPOSES]
    embeddingDimension: NOVA_EMBEDDING_DIMENSIONS
    text: NovaTextParamsWithSegmentation
    image: NovaImageParams
    video: NovaVideoParamsWithSegmentation
    audio: NovaAudioParamsWithSegmentation


class NovaEmbeddingRequest(TypedDict, total=False):
    schemaVersion: str  # "nova-multimodal-embed-v1"
    taskType: Literal["SINGLE_EMBEDDING", "SEGMENTED_EMBEDDING"]
    singleEmbeddingParams: NovaSingleEmbeddingParams
    segmentedEmbeddingParams: NovaSegmentedEmbeddingParams


class NovaEmbeddingItem(TypedDict, total=False):
    embeddingType: NOVA_EMBEDDING_TYPES
    embedding: Required[list[float]]
    truncatedCharLength: int  # Only for text


class NovaEmbeddingResponse(TypedDict):
    embeddings: list[NovaEmbeddingItem]


class NovaS3OutputDataConfig(TypedDict):
    s3Uri: str


class NovaOutputDataConfig(TypedDict):
    s3OutputDataConfig: NovaS3OutputDataConfig


class NovaAsyncInvokeRequest(TypedDict):
    modelId: str
    modelInput: NovaEmbeddingRequest
    outputDataConfig: NovaOutputDataConfig


AmazonEmbeddingRequest = (
    AmazonTitanMultimodalEmbeddingRequest | AmazonTitanV2EmbeddingRequest | AmazonTitanG1EmbeddingRequest
)


class AmazonStability3TextToImageRequest(TypedDict, total=False):
    """
    Request for Amazon Stability 3 Text to Image API

    Ref here: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-text-image.html
    """

    prompt: str
    aspect_ratio: Literal["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
    mode: Literal["image-to-image", "text-to-image"]
    output_format: Literal["JPEG", "PNG"]
    seed: int
    negative_prompt: str


class AmazonStability3TextToImageResponse(TypedDict, total=False):
    """
    Response for Amazon Stability 3 Text to Image API

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-text-image.html
    """

    images: list[str]
    seeds: list[str]
    finish_reasons: list[str]


class AmazonTitanTextToImageParams(TypedDict, total=False):
    """
    Params for Amazon Titan Text to Image API
    """

    text: Required[str]
    negativeText: str


class AmazonNovaCanvasRequestBase(TypedDict, total=False):
    """
    Base class for Amazon Nova Canvas API requests
    """


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


class AmazonNovaCanvasTextToImageRequest(AmazonNovaCanvasRequestBase, TypedDict, total=False):
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

    colors: list[str]
    referenceImage: str
    text: str
    negativeText: str


class AmazonNovaCanvasColorGuidedRequest(AmazonNovaCanvasRequestBase, TypedDict, total=False):
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

    images: list[str]


class AmazonNovaCanvasInpaintingParams(TypedDict, total=False):
    """
    Params for Amazon Nova Canvas Inpainting API
    """

    text: str
    maskImage: str
    inputImage: str
    negativeText: str


class AmazonNovaCanvasInpaintingRequest(AmazonNovaCanvasRequestBase, TypedDict, total=False):
    """
    Request for Amazon Nova Canvas Inpainting API

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
    """

    taskType: Literal["INPAINTING"]
    inpaintingParams: AmazonNovaCanvasInpaintingParams
    imageGenerationConfig: AmazonNovaCanvasImageGenerationConfig


class AmazonTitanImageGenerationRequestBody(TypedDict, total=False):
    """
    Config for Amazon Titan Image Generation API
    """

    taskType: Literal["TEXT_IMAGE", "COLOR_GUIDED_GENERATION", "INPAINTING"]
    textToImageParams: AmazonTitanTextToImageParams
    imageGenerationConfig: AmazonNovaCanvasImageGenerationConfig


if TYPE_CHECKING:
    from botocore.awsrequest import AWSPreparedRequest
else:
    AWSPreparedRequest = Any


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

    queries: list[BedrockRerankQuery]
    rerankingConfiguration: BedrockRerankConfiguration
    sources: list[BedrockRerankSource]


class AmazonDeepSeekR1StreamingResponse(TypedDict):
    generation: str
    generation_token_count: int
    stop_reason: str | None
    prompt_token_count: int


################ Bedrock Batch Types #################


class BedrockS3InputDataConfig(TypedDict):
    """S3 input data configuration for Bedrock batch jobs."""

    s3Uri: str


class BedrockInputDataConfig(TypedDict):
    """Input data configuration for Bedrock batch jobs."""

    s3InputDataConfig: BedrockS3InputDataConfig


class BedrockS3OutputDataConfig(TypedDict, total=False):
    """S3 output data configuration for Bedrock batch jobs."""

    s3Uri: str
    s3EncryptionKeyId: str | None


class BedrockOutputDataConfig(TypedDict):
    """Output data configuration for Bedrock batch jobs."""

    s3OutputDataConfig: BedrockS3OutputDataConfig


class BedrockTag(TypedDict):
    key: str
    value: str


class BedrockCreateBatchRequest(TypedDict, total=False):
    """
    Request structure for creating a Bedrock batch inference job.

    Reference: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_CreateModelInvocationJob.html
    """

    jobName: str
    roleArn: str
    modelId: str
    inputDataConfig: BedrockInputDataConfig
    outputDataConfig: BedrockOutputDataConfig
    timeoutDurationInHours: int | None
    clientRequestToken: str | None
    tags: list[BedrockTag] | None


BedrockBatchJobStatus = Literal["Submitted", "InProgress", "Completed", "Failed", "Stopping", "Stopped"]


class BedrockCreateBatchResponse(TypedDict):
    """
    Response structure from creating a Bedrock batch inference job.

    Reference: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_CreateModelInvocationJob.html
    """

    jobArn: str
    jobName: str
    status: BedrockBatchJobStatus


class BedrockGetBatchResponse(TypedDict, total=False):
    """
    Response structure from getting a Bedrock batch inference job.

    Reference: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_GetModelInvocationJob.html
    """

    jobArn: str
    jobName: str
    modelId: str
    roleArn: str
    status: BedrockBatchJobStatus
    message: str | None
    submitTime: str | None
    lastModifiedTime: str | None
    endTime: str | None
    inputDataConfig: BedrockInputDataConfig
    outputDataConfig: BedrockOutputDataConfig
    timeoutDurationInHours: int | None
    clientRequestToken: str | None


class BedrockToolBlock(TypedDict, total=False):
    toolSpec: ToolSpecBlock | None
    systemTool: SystemToolBlock | None  # For Nova grounding
    cachePoint: CachePointBlock | None


class BedrockInvokeAnthropicMessagesRequest(TypedDict, total=False):
    """
    Top-level request body accepted by AWS Bedrock `InvokeModel` /
    `InvokeModelWithResponseStream` when calling an Anthropic Claude model with
    the Messages API format. The LiteLLM /v1/messages → Bedrock Invoke
    transformation filters outgoing requests to the keys of this TypedDict; any
    other field (Anthropic-only extension, internal metadata, future addition)
    is dropped before signing so Bedrock doesn't 400 with
    "Extra inputs are not permitted".

    Reference:
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html

    Editing this type is the single source of truth — the runtime allowlist in
    `AmazonAnthropicClaudeMessagesConfig.BEDROCK_INVOKE_ALLOWED_TOP_LEVEL_FIELDS`
    is derived from `__annotations__`, and a test asserts the resolved set
    exactly, so any edit forces a conscious review.

    Value types are intentionally loose (`list`, `dict`) — this type exists to
    pin the allowed field names, not to validate nested structure.
    """

    # Required by Bedrock
    anthropic_version: str
    max_tokens: int
    messages: list

    # Documented optional fields
    anthropic_beta: list[str]
    system: object  # str or list[TextBlock]
    stop_sequences: list[str]
    temperature: float
    top_p: float
    top_k: int
    tools: list
    tool_choice: dict

    # `thinking` is required for Opus 4.5 / Sonnet 4 extended thinking,
    # `metadata` is part of the common Anthropic Messages API shape.
    thinking: dict
    metadata: dict
    output_config: dict

    # `context_management` is allowed for Bedrock InvokeModel only when it
    # carries `compact_20260112` edits paired with the `compact-2026-01-12`
    # anthropic-beta header. The Invoke transformation filters edits to the
    # supported subset and strips the field entirely when nothing remains, so
    # other edit types (e.g. `clear_thinking_20251015`) never reach Bedrock.
    context_management: dict


class BedrockBatchRecordKind(Enum):
    """
    Which OpenAI endpoint shape a line of a Bedrock managed-batch JSONL file
    carries. Bedrock batch `modelInput` is always the model's InvokeModel /
    Converse body, so every non-embedding shape is normalized to Chat
    Completions before being handed to the per-provider transformation.
    """

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    RESPONSES = "responses"
    EMBEDDING = "embedding"
