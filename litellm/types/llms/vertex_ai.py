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


class Field(TypedDict):
    key: str
    value: Dict[str, Any]


class FunctionCallArgs(TypedDict):
    fields: Field


class FunctionResponse(TypedDict):
    name: str
    response: FunctionCallArgs


class FunctionCall(TypedDict):
    name: str
    args: FunctionCallArgs


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


class HttpxFunctionCall(TypedDict):
    name: str
    args: dict


class HttpxPartType(TypedDict, total=False):
    text: str
    inline_data: BlobType
    file_data: FileDataType
    functionCall: HttpxFunctionCall
    function_response: FunctionResponse


class HttpxContentType(TypedDict, total=False):
    role: Literal["user", "model"]
    parts: Required[List[HttpxPartType]]


class ContentType(TypedDict, total=False):
    role: Literal["user", "model"]
    parts: Required[List[PartType]]


class SystemInstructions(TypedDict):
    parts: Required[List[PartType]]


class Schema(TypedDict, total=False):
    type: Literal["STRING", "INTEGER", "BOOLEAN", "NUMBER", "ARRAY", "OBJECT"]
    description: str
    enum: List[str]
    items: List["Schema"]
    properties: "Schema"
    required: List[str]
    nullable: bool


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
    seed: int


class Tools(TypedDict, total=False):
    function_declarations: List[FunctionDeclaration]
    googleSearchRetrieval: dict
    retrieval: Retrieval


class ToolConfig(TypedDict):
    functionCallingConfig: FunctionCallingConfig


class TTL(TypedDict, total=False):
    seconds: Required[float]
    nano: float


class UsageMetadata(TypedDict, total=False):
    promptTokenCount: int
    totalTokenCount: int
    candidatesTokenCount: int


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


class PromptFeedback(TypedDict):
    blockReason: str
    safetyRatings: List[SafetyRatings]
    blockReasonMessage: str


class GenerateContentResponseBody(TypedDict, total=False):
    candidates: List[Candidates]
    promptFeedback: PromptFeedback
    usageMetadata: Required[UsageMetadata]


class FineTunesupervisedTuningSpec(TypedDict, total=False):
    training_dataset_uri: str
    validation_dataset: Optional[str]
    epoch_count: Optional[int]
    learning_rate_multiplier: Optional[float]
    tuned_model_display_name: Optional[str]
    adapter_size: Optional[
        Literal[
            "ADAPTER_SIZE_UNSPECIFIED",
            "ADAPTER_SIZE_ONE",
            "ADAPTER_SIZE_FOUR",
            "ADAPTER_SIZE_EIGHT",
            "ADAPTER_SIZE_SIXTEEN",
        ]
    ]


class FineTuneJobCreate(TypedDict, total=False):
    baseModel: str
    supervisedTuningSpec: FineTunesupervisedTuningSpec
    tunedModelDisplayName: Optional[str]


class ResponseSupervisedTuningSpec(TypedDict):
    trainingDatasetUri: Optional[str]


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


class InstanceVideo(TypedDict, total=False):
    gcsUri: str
    videoSegmentConfig: Tuple[float, float, float]


class Instance(TypedDict, total=False):
    text: str
    image: Dict[str, str]
    video: InstanceVideo


class VertexMultimodalEmbeddingRequest(TypedDict, total=False):
    instances: List[Instance]


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
