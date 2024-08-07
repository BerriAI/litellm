import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

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
    parameters: Schema
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


class CachedContent(TypedDict, total=False):
    ttl: TTL
    expire_time: str


class RequestBody(TypedDict, total=False):
    contents: Required[List[ContentType]]
    system_instruction: SystemInstructions
    tools: Tools
    toolConfig: ToolConfig
    safetySettings: List[SafetSettingsConfig]
    generationConfig: GenerationConfig
    cachedContent: str


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


class UsageMetadata(TypedDict, total=False):
    promptTokenCount: int
    totalTokenCount: int
    candidatesTokenCount: int


class GenerateContentResponseBody(TypedDict, total=False):
    candidates: Required[List[Candidates]]
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
