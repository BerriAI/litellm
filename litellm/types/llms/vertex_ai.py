import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from typing_extensions import (
    Protocol,
    Required,
    Self,
    TypeGuard,
    get_origin,
    override,
    runtime_checkable,
)


class FunctionResponse(TypedDict):
    name: str
    response: Optional[dict]


class FunctionCall(TypedDict):
    name: str
    args: Optional[dict]


class FileDataType(TypedDict):
    mime_type: str
    file_uri: str  # the cloud storage uri of storing this file


class BlobType(TypedDict):
    mime_type: Required[str]
    data: Required[str]


class PartType(TypedDict, total=False):
    text: str
    inline_data: BlobType
    file_data: FileDataType
    function_call: FunctionCall
    function_response: FunctionResponse
    thought: bool


class HttpxFunctionCall(TypedDict):
    name: str
    args: dict


class HttpxExecutableCode(TypedDict):
    code: str
    language: str


class HttpxCodeExecutionResult(TypedDict):
    outcome: str
    output: str


class HttpxBlobType(TypedDict):
    mimeType: str
    data: str


class HttpxPartType(TypedDict, total=False):
    text: str
    inlineData: HttpxBlobType
    fileData: FileDataType
    functionCall: HttpxFunctionCall
    functionResponse: FunctionResponse
    executableCode: HttpxExecutableCode
    codeExecutionResult: HttpxCodeExecutionResult
    thought: bool


class HttpxContentType(TypedDict, total=False):
    role: Literal["user", "model"]
    parts: List[HttpxPartType]


class ContentType(TypedDict, total=False):
    role: Literal["user", "model"]
    parts: Required[List[PartType]]


class SystemInstructions(TypedDict):
    parts: Required[List[PartType]]


class Schema(TypedDict, total=False):
    type: Literal["STRING", "INTEGER", "BOOLEAN", "NUMBER", "ARRAY", "OBJECT"]
    format: str
    title: str
    description: str
    nullable: bool
    default: Any
    items: "Schema"
    minItems: str
    maxItems: str
    enum: List[str]
    properties: Dict[str, "Schema"]
    propertyOrdering: List[str]
    required: List[str]
    minProperties: str
    maxProperties: str
    minimum: float
    maximum: float
    minLength: str
    maxLength: str
    pattern: str
    example: Any
    anyOf: List["Schema"]


class FunctionDeclaration(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: Union[Schema, dict]
    response: Schema


class VertexAISearch(TypedDict, total=False):
    datastore: Required[str]


class Retrieval(TypedDict):
    source: VertexAISearch


class FunctionCallingConfig(TypedDict, total=False):
    mode: Literal["ANY", "AUTO", "NONE"]
    allowed_function_names: List[str]


HarmCategory = Literal[
    "HARM_CATEGORY_UNSPECIFIED",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
]
HarmBlockThreshold = Literal[
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_NONE",
]
HarmBlockMethod = Literal["HARM_BLOCK_METHOD_UNSPECIFIED", "SEVERITY", "PROBABILITY"]

HarmProbability = Literal[
    "HARM_PROBABILITY_UNSPECIFIED", "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"
]

HarmSeverity = Literal[
    "HARM_SEVERITY_UNSPECIFIED",
    "HARM_SEVERITY_NEGLIGIBLE",
    "HARM_SEVERITY_LOW",
    "HARM_SEVERITY_MEDIUM",
    "HARM_SEVERITY_HIGH",
]


class SafetSettingsConfig(TypedDict, total=False):
    category: HarmCategory
    threshold: HarmBlockThreshold
    max_influential_terms: int
    method: HarmBlockMethod


class GeminiThinkingConfig(TypedDict, total=False):
    includeThoughts: bool
    thinkingBudget: int


GeminiResponseModalities = Literal["TEXT", "IMAGE", "AUDIO", "VIDEO"]


class PrebuiltVoiceConfig(TypedDict):
    voiceName: str


class VoiceConfig(TypedDict):
    prebuiltVoiceConfig: PrebuiltVoiceConfig


class SpeechConfig(TypedDict, total=False):
    voiceConfig: VoiceConfig


class GenerationConfig(TypedDict, total=False):
    temperature: float
    top_p: float
    top_k: float
    candidate_count: int
    max_output_tokens: int
    stop_sequences: List[str]
    presence_penalty: float
    frequency_penalty: float
    response_mime_type: Literal["text/plain", "application/json"]
    response_schema: dict
    seed: int
    responseLogprobs: bool
    logprobs: int
    responseModalities: List[GeminiResponseModalities]
    thinkingConfig: GeminiThinkingConfig


class Tools(TypedDict, total=False):
    function_declarations: List[FunctionDeclaration]
    googleSearch: dict
    googleSearchRetrieval: dict
    enterpriseWebSearch: dict
    code_execution: dict
    retrieval: Retrieval


class ToolConfig(TypedDict):
    functionCallingConfig: FunctionCallingConfig


class TTL(TypedDict, total=False):
    seconds: Required[float]
    nano: float


class PromptTokensDetails(TypedDict):
    modality: Literal["TEXT", "AUDIO", "IMAGE", "VIDEO"]
    tokenCount: int


class UsageMetadata(TypedDict, total=False):
    promptTokenCount: int
    totalTokenCount: int
    candidatesTokenCount: int
    responseTokenCount: int
    cachedContentTokenCount: int
    promptTokensDetails: List[PromptTokensDetails]
    thoughtsTokenCount: int
    responseTokensDetails: List[PromptTokensDetails]


class CachedContent(TypedDict, total=False):
    ttl: TTL
    expire_time: str
    contents: List[ContentType]
    tools: List[Tools]
    createTime: str  # "2014-10-02T15:01:23Z" and "2014-10-02T15:01:23.045123456Z"
    updateTime: str  # "2014-10-02T15:01:23Z" and "2014-10-02T15:01:23.045123456Z"
    usageMetadata: UsageMetadata
    expireTime: str  # "2014-10-02T15:01:23Z" and "2014-10-02T15:01:23.045123456Z"
    name: str
    displayName: str
    model: str
    systemInstruction: ContentType
    toolConfig: ToolConfig


class RequestBody(TypedDict, total=False):
    contents: Required[List[ContentType]]
    system_instruction: SystemInstructions
    tools: Tools
    toolConfig: ToolConfig
    safetySettings: List[SafetSettingsConfig]
    generationConfig: GenerationConfig
    cachedContent: str
    speechConfig: SpeechConfig


class CachedContentRequestBody(TypedDict, total=False):
    contents: Required[List[ContentType]]
    system_instruction: SystemInstructions
    tools: Tools
    toolConfig: ToolConfig
    model: Required[str]  # Format: models/{model}
    ttl: str  # ending in 's' - Example: "3.5s".
    displayName: str


class CachedContentListAllResponseBody(TypedDict, total=False):
    cachedContents: List[CachedContent]
    nextPageToken: str


class SafetyRatings(TypedDict):
    category: HarmCategory
    probability: HarmProbability
    probabilityScore: int
    severity: HarmSeverity
    blocked: bool


class Date(TypedDict):
    year: int
    month: int
    date: int


class Citation(TypedDict):
    startIndex: int
    endIndex: int
    uri: str
    title: str
    license: str
    publicationDate: Date


class CitationMetadata(TypedDict):
    citations: List[Citation]


class SearchEntryPoint(TypedDict, total=False):
    renderedContent: str
    sdkBlob: str


class GroundingMetadata(TypedDict, total=False):
    webSearchQueries: List[str]
    searchEntryPoint: SearchEntryPoint
    groundingAttributions: List[dict]


class LogprobsCandidate(TypedDict):
    token: str
    tokenId: int
    logProbability: float


class LogprobsTopCandidate(TypedDict):
    candidates: List[LogprobsCandidate]


class LogprobsResult(TypedDict, total=False):
    topCandidates: List[LogprobsTopCandidate]
    chosenCandidates: List[LogprobsCandidate]


class Candidates(TypedDict, total=False):
    index: int
    content: HttpxContentType
    finishReason: Literal[
        "FINISH_REASON_UNSPECIFIED",
        "STOP",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
    ]
    safetyRatings: List[SafetyRatings]
    citationMetadata: CitationMetadata
    groundingMetadata: GroundingMetadata
    finishMessage: str
    logprobsResult: LogprobsResult


class PromptFeedback(TypedDict):
    blockReason: str
    safetyRatings: List[SafetyRatings]
    blockReasonMessage: str


class GenerateContentResponseBody(TypedDict, total=False):
    candidates: List[Candidates]
    promptFeedback: PromptFeedback
    usageMetadata: Required[UsageMetadata]


class FineTuneHyperparameters(TypedDict, total=False):
    epoch_count: Optional[int]
    learning_rate_multiplier: Optional[float]
    adapter_size: Optional[
        Literal[
            "ADAPTER_SIZE_UNSPECIFIED",
            "ADAPTER_SIZE_ONE",
            "ADAPTER_SIZE_FOUR",
            "ADAPTER_SIZE_EIGHT",
            "ADAPTER_SIZE_SIXTEEN",
        ]
    ]


class FineTunesupervisedTuningSpec(TypedDict, total=False):
    training_dataset_uri: str
    validation_dataset: Optional[str]
    tuned_model_display_name: Optional[str]
    hyperParameters: Optional[FineTuneHyperparameters]


class FineTuneJobCreate(TypedDict, total=False):
    baseModel: str
    supervisedTuningSpec: FineTunesupervisedTuningSpec
    tunedModelDisplayName: Optional[str]


class ResponseSupervisedTuningSpec(TypedDict, total=False):
    trainingDatasetUri: Optional[str]
    hyperParameters: Optional[FineTuneHyperparameters]


class ResponseTuningJob(TypedDict):
    name: Optional[str]
    tunedModelDisplayName: Optional[str]
    baseModel: Optional[str]
    supervisedTuningSpec: Optional[ResponseSupervisedTuningSpec]
    state: Optional[
        Literal[
            "JOB_STATE_PENDING",
            "JOB_STATE_RUNNING",
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        ]
    ]
    createTime: Optional[str]
    updateTime: Optional[str]


class VideoSegmentConfig(TypedDict, total=False):
    startOffsetSec: int
    endOffsetSec: int
    intervalSec: int


class InstanceVideo(TypedDict, total=False):
    gcsUri: str
    videoSegmentConfig: VideoSegmentConfig


class InstanceImage(TypedDict, total=False):
    gcsUri: Optional[str]
    bytesBase64Encoded: Optional[str]
    mimeType: Optional[str]


class Instance(TypedDict, total=False):
    text: str
    image: InstanceImage
    video: InstanceVideo


class VertexMultimodalEmbeddingRequest(TypedDict):
    instances: List[Instance]


class VideoEmbedding(TypedDict):
    startOffsetSec: int
    endOffsetSec: int
    embedding: List[float]


class MultimodalPrediction(TypedDict, total=False):
    textEmbedding: List[float]
    imageEmbedding: List[float]
    videoEmbeddings: List[VideoEmbedding]


class MultimodalPredictions(TypedDict):
    predictions: List[MultimodalPrediction]


class VertexAICachedContentResponseObject(TypedDict):
    name: str
    model: str


class TaskTypeEnum(Enum):
    TASK_TYPE_UNSPECIFIED = "TASK_TYPE_UNSPECIFIED"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


class VertexAITextEmbeddingsRequestBody(TypedDict, total=False):
    content: Required[ContentType]
    taskType: TaskTypeEnum
    title: str
    outputDimensionality: int


class ContentEmbeddings(TypedDict):
    values: List[int]


class VertexAITextEmbeddingsResponseObject(TypedDict):
    embedding: ContentEmbeddings


class EmbedContentRequest(VertexAITextEmbeddingsRequestBody):
    model: Required[str]


class VertexAIBatchEmbeddingsRequestBody(TypedDict, total=False):
    requests: List[EmbedContentRequest]


class VertexAIBatchEmbeddingsResponseObject(TypedDict):
    embeddings: List[ContentEmbeddings]


# Vertex AI Batch Prediction


class GcsSource(TypedDict):
    uris: str


class InputConfig(TypedDict):
    instancesFormat: str
    gcsSource: GcsSource


class GcsDestination(TypedDict):
    outputUriPrefix: str


class OutputConfig(TypedDict, total=False):
    predictionsFormat: str
    gcsDestination: GcsDestination


class GcsBucketResponse(TypedDict):
    """
    TypedDict for GCS bucket upload response

    Attributes:
        kind: The kind of item this is. For objects, this is always storage#object
        id: The ID of the object
        selfLink: The link to this object
        mediaLink: The link to download the object
        name: The name of the object
        bucket: The name of the bucket containing this object
        generation: The content generation of this object
        metageneration: The metadata generation of this object
        contentType: The content type of the object
        storageClass: The storage class of the object
        size: The size of the object in bytes
        md5Hash: The MD5 hash of the object
        crc32c: The CRC32c checksum of the object
        etag: The ETag of the object
        timeCreated: The creation time of the object
        updated: The last update time of the object
        timeStorageClassUpdated: The time the storage class was last updated
        timeFinalized: The time the object was finalized
    """

    kind: Literal["storage#object"]
    id: str
    selfLink: str
    mediaLink: str
    name: str
    bucket: str
    generation: str
    metageneration: str
    contentType: str
    storageClass: str
    size: str
    md5Hash: str
    crc32c: str
    etag: str
    timeCreated: str
    updated: str
    timeStorageClassUpdated: str
    timeFinalized: str


class VertexAIBatchPredictionJob(TypedDict):
    displayName: str
    model: str
    inputConfig: InputConfig
    outputConfig: OutputConfig


class VertexBatchPredictionResponse(TypedDict, total=False):
    name: str
    displayName: str
    model: str
    inputConfig: InputConfig
    outputConfig: OutputConfig
    state: str
    createTime: str
    updateTime: str
    modelVersionId: str


VERTEX_CREDENTIALS_TYPES = Union[str, Dict[str, str]]


class VertexPartnerProvider(str, Enum):
    mistralai = "mistralai"
    llama = "llama"
    ai21 = "ai21"
    claude = "claude"
