from enum import Enum
from os import PathLike
from typing import IO, Any, Iterable, List, Literal, Mapping, Optional, Tuple, Union

import httpx
from openai._legacy_response import (
    HttpxBinaryResponseContent as _HttpxBinaryResponseContent,
)
from openai.lib.streaming._assistants import (
    AssistantEventHandler,
    AssistantStreamManager,
    AsyncAssistantEventHandler,
    AsyncAssistantStreamManager,
)
from openai.pagination import AsyncCursorPage, SyncCursorPage
from openai.types import Batch, EmbeddingCreateParams, FileObject
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_tool_param import AssistantToolParam
from openai.types.beta.thread_create_params import (
    Message as OpenAICreateThreadParamsMessage,
)
from openai.types.beta.threads.message import Message as OpenAIMessage
from openai.types.beta.threads.message_content import MessageContent
from openai.types.beta.threads.run import Run
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam,
)
from openai.types.chat.chat_completion_modality import ChatCompletionModality
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.embedding import Embedding as OpenAIEmbedding
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
from openai.types.responses.response import (
    IncompleteDetails,
    Response,
    ResponseOutputItem,
    Tool,
    ToolChoice,
)

# Handle OpenAI SDK version compatibility for Text type
try:
    from openai.types.responses.response_create_params import Text as ResponseText
except (ImportError, AttributeError):
    # Fall back to the concrete config type available in all SDK versions
    from openai.types.responses.response_text_config_param import ResponseTextConfigParam as ResponseText

from openai.types.responses.response_create_params import (
    Reasoning,
    ResponseIncludable,
    ResponseInputParam,
    ToolChoice,
    ToolParam,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Discriminator, Field, PrivateAttr
from typing_extensions import Annotated, Dict, Required, TypedDict, override

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    OutputFunctionToolCall,
)

FileContent = Union[IO[bytes], bytes, PathLike]

FileTypes = Union[
    # file (or bytes)
    FileContent,
    # (filename, file (or bytes))
    Tuple[Optional[str], FileContent],
    # (filename, file (or bytes), content_type)
    Tuple[Optional[str], FileContent, Optional[str]],
    # (filename, file (or bytes), content_type, headers)
    Tuple[Optional[str], FileContent, Optional[str], Mapping[str, str]],
]


EmbeddingInput = Union[str, List[str]]


class HttpxBinaryResponseContent(_HttpxBinaryResponseContent):
    _hidden_params: dict = {}
    pass


class NotGiven:
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response:
        ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


class ToolResourcesCodeInterpreter(TypedDict, total=False):
    file_ids: List[str]
    """
    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs made
    available to the `code_interpreter` tool. There can be a maximum of 20 files
    associated with the tool.
    """


class ToolResourcesFileSearchVectorStore(TypedDict, total=False):
    file_ids: List[str]
    """
    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs to
    add to the vector store. There can be a maximum of 10000 files in a vector
    store.
    """

    metadata: object
    """Set of 16 key-value pairs that can be attached to a vector store.

    This can be useful for storing additional information about the vector store in
    a structured format. Keys can be a maximum of 64 characters long and values can
    be a maxium of 512 characters long.
    """


class ToolResourcesFileSearch(TypedDict, total=False):
    vector_store_ids: List[str]
    """
    The
    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)
    attached to this thread. There can be a maximum of 1 vector store attached to
    the thread.
    """

    vector_stores: Iterable[ToolResourcesFileSearchVectorStore]
    """
    A helper to create a
    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)
    with file_ids and attach it to this thread. There can be a maximum of 1 vector
    store attached to the thread.
    """


class OpenAICreateThreadParamsToolResources(TypedDict, total=False):
    code_interpreter: ToolResourcesCodeInterpreter

    file_search: ToolResourcesFileSearch


class FileSearchToolParam(TypedDict, total=False):
    type: Required[Literal["file_search"]]
    """The type of tool being defined: `file_search`"""


class CodeInterpreterToolParam(TypedDict, total=False):
    type: Required[Literal["code_interpreter"]]
    """The type of tool being defined: `code_interpreter`"""


AttachmentTool = Union[CodeInterpreterToolParam, FileSearchToolParam]


class Attachment(TypedDict, total=False):
    file_id: str
    """The ID of the file to attach to the message."""

    tools: Iterable[AttachmentTool]
    """The tools to add this file to."""


class ImageFileObject(TypedDict):
    file_id: Required[str]
    detail: Optional[str]


class ImageURLObject(TypedDict):
    url: Required[str]
    detail: Optional[str]


class MessageContentTextObject(TypedDict):
    type: Required[Literal["text"]]
    text: str


class MessageContentImageFileObject(TypedDict):
    type: Literal["image_file"]
    image_file: ImageFileObject


class MessageContentImageURLObject(TypedDict):
    type: Required[str]
    image_url: ImageURLObject


class MessageData(TypedDict):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                MessageContentTextObject,
                MessageContentImageFileObject,
                MessageContentImageURLObject,
            ]
        ],
    ]
    attachments: Optional[List[Attachment]]
    metadata: Optional[dict]


class Thread(BaseModel):
    id: str
    """The identifier, which can be referenced in API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the thread was created."""

    metadata: Optional[object] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format. Keys can be a maximum of 64 characters long and values can be
    a maxium of 512 characters long.
    """

    object: Literal["thread"]
    """The object type, which is always `thread`."""


OpenAICreateFileRequestOptionalParams = Literal["purpose"]

OpenAIFilesPurpose = Literal[
    "assistants",
    "assistants_output",
    "batch",
    "batch_output",
    "fine-tune",
    "fine-tune-results",
    "vision",
    "user_data",
]


class OpenAIFileObject(BaseModel):
    id: str
    """The file identifier, which can be referenced in the API endpoints."""

    bytes: int
    """The size of the file, in bytes."""

    created_at: int
    """The Unix timestamp (in seconds) for when the file was created."""

    filename: str
    """The name of the file."""

    object: Literal["file"]
    """The object type, which is always `file`."""

    purpose: OpenAIFilesPurpose
    """The intended purpose of the file.

    Supported values are `assistants`, `assistants_output`, `batch`, `batch_output`,
    `fine-tune`, `fine-tune-results`, `vision`, and `user_data`.
    """

    status: Literal["uploaded", "processed", "error"]
    """Deprecated.

    The current status of the file, which can be either `uploaded`, `processed`, or
    `error`.
    """

    expires_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the file will expire."""

    status_details: Optional[str] = None
    """Deprecated.

    For details on why a fine-tuning training file failed validation, see the
    `error` field on `fine_tuning.job`.
    """

    _hidden_params: dict = {"response_cost": 0.0}  # no cost for writing a file

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


CREATE_FILE_REQUESTS_PURPOSE = Literal["assistants", "batch", "fine-tune"]


# OpenAI Files Types
class CreateFileRequest(TypedDict, total=False):
    """
    CreateFileRequest
    Used by Assistants API, Batches API, and Fine-Tunes API

    Required Params:
        file: FileTypes
        purpose: Literal['assistants', 'batch', 'fine-tune']

    Optional Params:
        extra_headers: Optional[Dict[str, str]]
        extra_body: Optional[Dict[str, str]] = None
        timeout: Optional[float] = None
    """

    file: Required[FileTypes]
    purpose: Required[CREATE_FILE_REQUESTS_PURPOSE]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class FileContentRequest(TypedDict, total=False):
    """
    FileContentRequest
    Used by Assistants API, Batches API, and Fine-Tunes API

    Required Params:
        file_id: str

    Optional Params:
        extra_headers: Optional[Dict[str, str]]
        extra_body: Optional[Dict[str, str]] = None
        timeout: Optional[float] = None
    """

    file_id: str
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


# OpenAI Batches Types
class CreateBatchRequest(TypedDict, total=False):
    """
    CreateBatchRequest
    """

    completion_window: Literal["24h"]
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"]
    input_file_id: str
    metadata: Optional[Dict[str, str]]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class LiteLLMBatchCreateRequest(CreateBatchRequest, total=False):
    model: str


class RetrieveBatchRequest(TypedDict, total=False):
    """
    RetrieveBatchRequest
    """

    batch_id: str
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class CancelBatchRequest(TypedDict, total=False):
    """
    CancelBatchRequest
    """

    batch_id: str
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class ListBatchRequest(TypedDict, total=False):
    """
    ListBatchRequest - List your organization's batches
    Calls https://api.openai.com/v1/batches
    """

    after: Union[str, NotGiven]
    limit: Union[int, NotGiven]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


BatchJobStatus = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
]


class ChatCompletionAudioDelta(TypedDict, total=False):
    data: str
    transcript: str
    expires_at: int
    id: str


class ChatCompletionToolCallFunctionChunk(TypedDict, total=False):
    name: Optional[str]
    arguments: str


class ChatCompletionAssistantToolCall(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk


class ChatCompletionToolCallChunk(TypedDict):  # result of /chat/completions call
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk
    index: int


class ChatCompletionDeltaToolCallChunk(TypedDict, total=False):
    id: str
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk
    index: int


class ChatCompletionCachedContent(TypedDict):
    type: Literal["ephemeral"]


class ChatCompletionThinkingBlock(TypedDict, total=False):
    type: Required[Literal["thinking"]]
    thinking: str
    signature: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class ChatCompletionRedactedThinkingBlock(TypedDict, total=False):
    type: Required[Literal["redacted_thinking"]]
    data: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class WebSearchOptionsUserLocationApproximate(TypedDict, total=False):
    city: str
    """Free text input for the city of the user, e.g. `San Francisco`."""

    country: str
    """
    The two-letter [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1) of
    the user, e.g. `US`.
    """

    region: str
    """Free text input for the region of the user, e.g. `California`."""

    timezone: str
    """
    The [IANA timezone](https://timeapi.io/documentation/iana-timezones) of the
    user, e.g. `America/Los_Angeles`.
    """


class WebSearchOptionsUserLocation(TypedDict, total=False):
    approximate: Required[WebSearchOptionsUserLocationApproximate]
    """Approximate location parameters for the search."""

    type: Required[Literal["approximate"]]
    """The type of location approximation. Always `approximate`."""


class WebSearchOptions(TypedDict, total=False):
    search_context_size: Literal["low", "medium", "high"]
    """
    High level guidance for the amount of context window space to use for the
    search. One of `low`, `medium`, or `high`. `medium` is the default.
    """

    user_location: Optional[WebSearchOptionsUserLocation]
    """Approximate location parameters for the search."""


class FileSearchTool(TypedDict, total=False):
    type: Literal["file_search"]
    """The type of tool being defined: `file_search`"""

    vector_store_ids: Optional[List[str]]
    """The IDs of the vector stores to search."""


class ChatCompletionAnnotationURLCitation(TypedDict, total=False):
    end_index: int
    """The index of the last character of the URL citation in the message."""

    start_index: int
    """The index of the first character of the URL citation in the message."""

    title: str
    """The title of the web resource."""

    url: str
    """The URL of the web resource."""


class ChatCompletionAnnotation(TypedDict, total=False):
    type: Literal["url_citation"]
    """The type of the URL citation. Always `url_citation`."""

    url_citation: ChatCompletionAnnotationURLCitation
    """A URL citation when using web search."""


class OpenAIChatCompletionTextObject(TypedDict):
    type: Literal["text"]
    text: str


class ChatCompletionTextObject(
    OpenAIChatCompletionTextObject, total=False
):  # litellm wrapper on top of openai object for handling cached content
    cache_control: ChatCompletionCachedContent


class ChatCompletionImageUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str
    format: str


class ChatCompletionImageObject(TypedDict):
    type: Literal["image_url"]
    image_url: Union[str, ChatCompletionImageUrlObject]


class ChatCompletionVideoUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str


class ChatCompletionVideoObject(TypedDict):
    type: Literal["video_url"]
    video_url: Union[str, ChatCompletionVideoUrlObject]


class ChatCompletionAudioObject(ChatCompletionContentPartInputAudioParam):
    pass


class DocumentObject(TypedDict):
    type: Literal["text"]
    media_type: str
    data: str


class CitationsObject(TypedDict):
    enabled: bool


class ChatCompletionDocumentObject(TypedDict):
    type: Literal["document"]
    source: DocumentObject
    title: str
    context: str
    citations: Optional[CitationsObject]


class ChatCompletionFileObjectFile(TypedDict, total=False):
    file_data: str
    file_id: str
    filename: str
    format: str


class ChatCompletionFileObject(TypedDict):
    type: Literal["file"]
    file: ChatCompletionFileObjectFile


OpenAIMessageContentListBlock = Union[
    ChatCompletionTextObject,
    ChatCompletionImageObject,
    ChatCompletionAudioObject,
    ChatCompletionDocumentObject,
    ChatCompletionVideoObject,
    ChatCompletionFileObject,
]

OpenAIMessageContent = Union[
    str,
    Iterable[OpenAIMessageContentListBlock],
]

# The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
AllPromptValues = Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None]


class OpenAIChatCompletionUserMessage(TypedDict):
    role: Literal["user"]
    content: OpenAIMessageContent


class OpenAITextCompletionUserMessage(TypedDict):
    role: Literal["user"]
    content: AllPromptValues


class ChatCompletionUserMessage(OpenAIChatCompletionUserMessage, total=False):
    cache_control: ChatCompletionCachedContent


class OpenAIChatCompletionAssistantMessage(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    content: Optional[
        Union[
            str, Iterable[Union[ChatCompletionTextObject, ChatCompletionThinkingBlock]]
        ]
    ]
    name: Optional[str]
    tool_calls: Optional[List[ChatCompletionAssistantToolCall]]
    function_call: Optional[ChatCompletionToolCallFunctionChunk]
    reasoning_content: Optional[str]


class ChatCompletionAssistantMessage(OpenAIChatCompletionAssistantMessage, total=False):
    cache_control: ChatCompletionCachedContent
    thinking_blocks: Optional[List[ChatCompletionThinkingBlock]]


class ChatCompletionToolMessage(TypedDict):
    role: Literal["tool"]
    content: Union[str, Iterable[ChatCompletionTextObject]]
    tool_call_id: str


class ChatCompletionFunctionMessage(TypedDict):
    role: Literal["function"]
    content: Optional[Union[str, Iterable[ChatCompletionTextObject]]]
    name: str
    tool_call_id: Optional[str]


class OpenAIChatCompletionSystemMessage(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[Union[str, List]]
    name: str


class OpenAIChatCompletionDeveloperMessage(TypedDict, total=False):
    role: Required[Literal["developer"]]
    content: Required[Union[str, List]]
    name: str


class ChatCompletionSystemMessage(OpenAIChatCompletionSystemMessage, total=False):
    cache_control: ChatCompletionCachedContent


class ChatCompletionDeveloperMessage(OpenAIChatCompletionDeveloperMessage, total=False):
    cache_control: ChatCompletionCachedContent


class GenericChatCompletionMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[Union[str, List]]


ValidUserMessageContentTypes = [
    "text",
    "image_url",
    "input_audio",
    "audio_url",
    "document",
    "video_url",
    "file",
]  # used for validating user messages. Prevent users from accidentally sending anthropic messages.

AllMessageValues = Union[
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionSystemMessage,
    ChatCompletionFunctionMessage,
    ChatCompletionDeveloperMessage,
]


class ChatCompletionToolChoiceFunctionParam(TypedDict):
    name: str


class ChatCompletionToolChoiceObjectParam(TypedDict):
    type: Literal["function"]
    function: ChatCompletionToolChoiceFunctionParam


ChatCompletionToolChoiceStringValues = Literal["none", "auto", "required"]

ChatCompletionToolChoiceValues = Union[
    ChatCompletionToolChoiceStringValues, ChatCompletionToolChoiceObjectParam
]


class ChatCompletionToolParamFunctionChunk(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: dict
    strict: bool


class OpenAIChatCompletionToolParam(TypedDict):
    type: Union[Literal["function"], str]
    function: ChatCompletionToolParamFunctionChunk


class ChatCompletionToolParam(OpenAIChatCompletionToolParam, total=False):
    cache_control: ChatCompletionCachedContent


class Function(TypedDict, total=False):
    name: Required[str]
    """The name of the function to call."""


class ChatCompletionNamedToolChoiceParam(TypedDict, total=False):
    function: Required[Function]

    type: Required[Literal["function"]]
    """The type of the tool. Currently, only `function` is supported."""


class ChatCompletionRequest(TypedDict, total=False):
    model: Required[str]
    messages: Required[List[AllMessageValues]]
    frequency_penalty: float
    logit_bias: dict
    logprobs: bool
    top_logprobs: int
    max_tokens: int
    n: int
    presence_penalty: float
    response_format: dict
    seed: int
    service_tier: str
    stop: Union[str, List[str]]
    stream_options: dict
    temperature: float
    top_p: float
    tools: List[ChatCompletionToolParam]
    tool_choice: ChatCompletionToolChoiceValues
    parallel_tool_calls: bool
    function_call: Union[str, dict]
    functions: List
    user: str
    metadata: dict  # litellm specific param


class ChatCompletionDeltaChunk(TypedDict, total=False):
    content: Optional[str]
    tool_calls: List[ChatCompletionDeltaToolCallChunk]
    role: str


ChatCompletionAssistantContentValue = (
    str  # keep as var, used in stream_chunk_builder as well
)


class ChatCompletionResponseMessage(TypedDict, total=False):
    content: Optional[ChatCompletionAssistantContentValue]
    tool_calls: Optional[List[ChatCompletionToolCallChunk]]
    role: Literal["assistant"]
    function_call: Optional[ChatCompletionToolCallFunctionChunk]
    provider_specific_fields: Optional[dict]
    reasoning_content: Optional[str]
    thinking_blocks: Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ]


class ChatCompletionUsageBlock(TypedDict, total=False):
    prompt_tokens: Required[int]
    completion_tokens: Required[int]
    total_tokens: Required[int]
    prompt_tokens_details: Optional[dict]
    completion_tokens_details: Optional[dict]


class OpenAIChatCompletionChunk(ChatCompletionChunk):
    def __init__(self, **kwargs):
        # Set the 'object' kwarg to 'chat.completion.chunk'
        kwargs["object"] = "chat.completion.chunk"
        super().__init__(**kwargs)


class Hyperparameters(BaseModel):
    batch_size: Optional[Union[str, int]] = None  # "Number of examples in each batch."
    learning_rate_multiplier: Optional[Union[str, float]] = (
        None  # Scaling factor for the learning rate
    )
    n_epochs: Optional[Union[str, int]] = (
        None  # "The number of epochs to train the model for"
    )


class FineTuningJobCreate(BaseModel):
    """
    FineTuningJobCreate - Create a fine-tuning job

    Example Request
    ```
    {
        "model": "gpt-3.5-turbo",
        "training_file": "file-abc123",
        "hyperparameters": {
            "batch_size": "auto",
            "learning_rate_multiplier": 0.1,
            "n_epochs": 3
        },
        "suffix": "custom-model-name",
        "validation_file": "file-xyz789",
        "integrations": ["slack"],
        "seed": 42
    }
    ```
    """

    model: str  # "The name of the model to fine-tune."
    training_file: str  # "The ID of an uploaded file that contains training data."
    hyperparameters: Optional[Hyperparameters] = (
        None  # "The hyperparameters used for the fine-tuning job."
    )
    suffix: Optional[str] = (
        None  # "A string of up to 18 characters that will be added to your fine-tuned model name."
    )
    validation_file: Optional[str] = (
        None  # "The ID of an uploaded file that contains validation data."
    )
    integrations: Optional[List[str]] = (
        None  # "A list of integrations to enable for your fine-tuning job."
    )
    seed: Optional[int] = None  # "The seed controls the reproducibility of the job."


class LiteLLMFineTuningJobCreate(FineTuningJobCreate):
    custom_llm_provider: Optional[Literal["openai", "azure", "vertex_ai"]] = None

    model_config = {
        "extra": "allow"
    }  # This allows the model to accept additional fields


AllEmbeddingInputValues = Union[str, List[str], List[int], List[List[int]]]

OpenAIAudioTranscriptionOptionalParams = Literal[
    "language",
    "prompt",
    "temperature",
    "response_format",
    "timestamp_granularities",
    "include",
]


OpenAIImageVariationOptionalParams = Literal["n", "size", "response_format", "user"]


OpenAIImageGenerationOptionalParams = Literal[
    "background",
    "input_fidelity",
    "moderation",
    "n",
    "output_compression",
    "output_format",
    "quality",
    "response_format",
    "size",
    "style",
    "user",
]


class ComputerToolParam(TypedDict, total=False):
    display_height: Required[float]
    """The height of the computer display."""

    display_width: Required[float]
    """The width of the computer display."""

    environment: Required[Union[Literal["mac", "windows", "ubuntu", "browser"], str]]
    """The type of computer environment to control."""

    type: Required[Union[Literal["computer_use_preview"], str]]


ALL_RESPONSES_API_TOOL_PARAMS = Union[ToolParam, ComputerToolParam]


class PromptObject(TypedDict, total=False):
    """Reference to a stored prompt template."""

    id: Required[str]
    """The unique identifier of the prompt template to use."""

    variables: Optional[Dict]
    """Variables to substitute into the prompt template."""

    version: Optional[str]
    """Optional version of the prompt template."""


class ResponsesAPIOptionalRequestParams(TypedDict, total=False):
    """TypedDict for Optional parameters supported by the responses API."""

    include: Optional[List[ResponseIncludable]]
    instructions: Optional[str]
    max_output_tokens: Optional[int]
    metadata: Optional[Dict[str, Any]]
    parallel_tool_calls: Optional[bool]
    previous_response_id: Optional[str]
    reasoning: Optional[Reasoning]
    store: Optional[bool]
    background: Optional[bool]
    stream: Optional[bool]
    temperature: Optional[float]
    text: Optional["ResponseText"]
    tool_choice: Optional[ToolChoice]
    tools: Optional[List[ALL_RESPONSES_API_TOOL_PARAMS]]
    top_p: Optional[float]
    truncation: Optional[Literal["auto", "disabled"]]
    user: Optional[str]
    service_tier: Optional[str]
    safety_identifier: Optional[str]
    prompt: Optional[PromptObject]


class ResponsesAPIRequestParams(ResponsesAPIOptionalRequestParams, total=False):
    """TypedDict for request parameters supported by the responses API."""

    input: Union[str, ResponseInputParam]
    model: str


class OutputTokensDetails(BaseLiteLLMOpenAIResponseObject):
    reasoning_tokens: Optional[int] = None

    text_tokens: Optional[int] = None

    model_config = {"extra": "allow"}


class InputTokensDetails(BaseLiteLLMOpenAIResponseObject):
    audio_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    text_tokens: Optional[int] = None

    model_config = {"extra": "allow"}


class ResponseAPIUsage(BaseLiteLLMOpenAIResponseObject):
    input_tokens: int
    """The number of input tokens."""

    input_tokens_details: Optional[InputTokensDetails] = None
    """A detailed breakdown of the input tokens."""

    output_tokens: int
    """The number of output tokens."""

    output_tokens_details: Optional[OutputTokensDetails] = None
    """A detailed breakdown of the output tokens."""

    total_tokens: int
    """The total number of tokens used."""

    model_config = {"extra": "allow"}


class ResponsesAPIResponse(BaseLiteLLMOpenAIResponseObject):
    id: str
    created_at: int
    error: Optional[dict]
    incomplete_details: Optional[IncompleteDetails]
    instructions: Optional[str]
    metadata: Optional[Dict]
    model: Optional[str]
    object: Optional[str]
    output: Union[
        List[Union[ResponseOutputItem, Dict]],
        List[Union[GenericResponseOutputItem, OutputFunctionToolCall]],
    ]
    parallel_tool_calls: bool
    temperature: Optional[float]
    tool_choice: ToolChoice
    tools: Union[List[Tool], List[ResponseFunctionToolCall]]
    top_p: Optional[float]
    max_output_tokens: Optional[int]
    previous_response_id: Optional[str]
    reasoning: Optional[Reasoning]
    status: Optional[str]
    text: Optional[Union["ResponseText", Dict[str, Any]]]
    truncation: Optional[Literal["auto", "disabled"]]
    usage: Optional[ResponseAPIUsage]
    user: Optional[str]
    store: Optional[bool] = None
    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class ResponsesAPIStreamEvents(str, Enum):
    """
    Enum representing all supported OpenAI stream event types for the Responses API.

    Inherits from str to allow direct string comparison and usage as dictionary keys.
    """

    # Response lifecycle events
    RESPONSE_CREATED = "response.created"
    RESPONSE_IN_PROGRESS = "response.in_progress"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_FAILED = "response.failed"
    RESPONSE_INCOMPLETE = "response.incomplete"

    # Reasoning summary events
    RESPONSE_PART_ADDED = "response.reasoning_summary_part.added"
    REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"

    # Output item events
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"

    # Content part events
    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"

    # Output text events
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_TEXT_ANNOTATION_ADDED = "response.output_text.annotation.added"
    OUTPUT_TEXT_DONE = "response.output_text.done"

    # Refusal events
    REFUSAL_DELTA = "response.refusal.delta"
    REFUSAL_DONE = "response.refusal.done"

    # Function call events
    FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"

    # File search events
    FILE_SEARCH_CALL_IN_PROGRESS = "response.file_search_call.in_progress"
    FILE_SEARCH_CALL_SEARCHING = "response.file_search_call.searching"
    FILE_SEARCH_CALL_COMPLETED = "response.file_search_call.completed"

    # Web search events
    WEB_SEARCH_CALL_IN_PROGRESS = "response.web_search_call.in_progress"
    WEB_SEARCH_CALL_SEARCHING = "response.web_search_call.searching"
    WEB_SEARCH_CALL_COMPLETED = "response.web_search_call.completed"

    # Error event
    ERROR = "error"


class ResponseCreatedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.RESPONSE_CREATED]
    response: ResponsesAPIResponse


class ResponseInProgressEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS]
    response: ResponsesAPIResponse


class ResponseCompletedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.RESPONSE_COMPLETED]
    response: ResponsesAPIResponse
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class ResponseFailedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.RESPONSE_FAILED]
    response: ResponsesAPIResponse


class ResponseIncompleteEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE]
    response: ResponsesAPIResponse


class ResponsePartAddedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.RESPONSE_PART_ADDED]
    item_id: str
    output_index: int
    part: dict


class ReasoningSummaryTextDeltaEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA]
    item_id: str
    output_index: int
    delta: str


class OutputItemAddedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED]
    output_index: int
    item: Optional[dict]


class OutputItemDoneEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE]
    output_index: int
    item: dict


class ContentPartAddedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.CONTENT_PART_ADDED]
    item_id: str
    output_index: int
    content_index: int
    part: dict


class ContentPartDoneEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.CONTENT_PART_DONE]
    item_id: str
    output_index: int
    content_index: int
    part: dict


class OutputTextDeltaEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class OutputTextAnnotationAddedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED]
    item_id: str
    output_index: int
    content_index: int
    annotation_index: int
    annotation: dict


class OutputTextDoneEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE]
    item_id: str
    output_index: int
    content_index: int
    text: str


class RefusalDeltaEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.REFUSAL_DELTA]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class RefusalDoneEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.REFUSAL_DONE]
    item_id: str
    output_index: int
    content_index: int
    refusal: str


class FunctionCallArgumentsDeltaEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA]
    item_id: str
    output_index: int
    delta: str


class FunctionCallArgumentsDoneEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE]
    item_id: str
    output_index: int
    arguments: str


class FileSearchCallInProgressEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.FILE_SEARCH_CALL_IN_PROGRESS]
    output_index: int
    item_id: str


class FileSearchCallSearchingEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.FILE_SEARCH_CALL_SEARCHING]
    output_index: int
    item_id: str


class FileSearchCallCompletedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.FILE_SEARCH_CALL_COMPLETED]
    output_index: int
    item_id: str


class WebSearchCallInProgressEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.WEB_SEARCH_CALL_IN_PROGRESS]
    output_index: int
    item_id: str


class WebSearchCallSearchingEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.WEB_SEARCH_CALL_SEARCHING]
    output_index: int
    item_id: str


class WebSearchCallCompletedEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.WEB_SEARCH_CALL_COMPLETED]
    output_index: int
    item_id: str


class ErrorEvent(BaseLiteLLMOpenAIResponseObject):
    type: Literal[ResponsesAPIStreamEvents.ERROR]
    code: Optional[str]
    message: str
    param: Optional[str]


class GenericEvent(BaseLiteLLMOpenAIResponseObject):
    type: str

    model_config = ConfigDict(extra="allow", protected_namespaces=())


# Union type for all possible streaming responses
ResponsesAPIStreamingResponse = Annotated[
    Union[
        ResponseCreatedEvent,
        ResponseInProgressEvent,
        ResponseCompletedEvent,
        ResponseFailedEvent,
        ResponseIncompleteEvent,
        ResponsePartAddedEvent,
        ReasoningSummaryTextDeltaEvent,
        OutputItemAddedEvent,
        OutputItemDoneEvent,
        ContentPartAddedEvent,
        ContentPartDoneEvent,
        OutputTextDeltaEvent,
        OutputTextAnnotationAddedEvent,
        OutputTextDoneEvent,
        RefusalDeltaEvent,
        RefusalDoneEvent,
        FunctionCallArgumentsDeltaEvent,
        FunctionCallArgumentsDoneEvent,
        FileSearchCallInProgressEvent,
        FileSearchCallSearchingEvent,
        FileSearchCallCompletedEvent,
        WebSearchCallInProgressEvent,
        WebSearchCallSearchingEvent,
        WebSearchCallCompletedEvent,
        ErrorEvent,
        GenericEvent,
    ],
    Discriminator("type"),
]


REASONING_EFFORT = Literal["minimal", "low", "medium", "high"]


class OpenAIRealtimeStreamSession(TypedDict, total=False):
    id: Required[str]
    """
    Unique identifier for the session that looks like sess_1234567890abcdef.
    """

    input_audio_format: str
    """
    The format of input audio. Options are pcm16, g711_ulaw, or g711_alaw. For pcm16, input audio must be 16-bit PCM at a 24kHz sample rate, single channel (mono), and little-endian byte order.
    """

    input_audio_noise_reduction: object
    """
    Configuration for input audio noise reduction. This can be set to null to turn off. Noise reduction filters audio added to the input audio buffer before it is sent to VAD and the model. Filtering the audio can improve VAD and turn detection accuracy (reducing false positives) and model performance by improving perception of the input audio.
    """

    input_audio_transcription: object
    """
    Configuration for input audio transcription, defaults to off and can be set to null to turn off once on. Input audio transcription is not native to the model, since the model consumes audio directly. Transcription runs asynchronously through the /audio/transcriptions endpoint and should be treated as guidance of input audio content rather than precisely what the model heard. The client can optionally set the language and prompt for transcription, these offer additional guidance to the transcription service.
    """

    instructions: str
    """
    The default system instructions (i.e. system message) prepended to model calls. This field allows the client to guide the model on desired responses. The model can be instructed on response content and format, (e.g. "be extremely succinct", "act friendly", "here are examples of good responses") and on audio behavior (e.g. "talk quickly", "inject emotion into your voice", "laugh frequently"). The instructions are not guaranteed to be followed by the model, but they provide guidance to the model on the desired behavior.
    """

    max_response_output_tokens: Union[int, Literal["inf"]]
    """
    Maximum number of output tokens for a single assistant response, inclusive of tool calls. Provide an integer between 1 and 4096 to limit output tokens, or inf for the maximum available tokens for a given model. Defaults to inf.
    """

    modalities: List[str]
    """
    The set of modalities the model can respond with. To disable audio, set this to ["text"].
    """

    model: str
    """
    The Realtime model used for this session.
    """

    output_audio_format: str
    """
    The format of output audio. Options are pcm16, g711_ulaw, or g711_alaw. For pcm16, output audio is sampled at a rate of 24kHz.
    """

    temperature: float
    """
    Sampling temperature for the model, limited to [0.6, 1.2]. For audio models a temperature of 0.8 is highly recommended for best performance.
    """

    tool_choice: str
    """
    How the model chooses tools. Options are auto, none, required, or specify a function.
    """

    tools: list
    """
    Tools (functions) available to the model.
    """

    turn_detection: object
    """

    Configuration for turn detection, ether Server VAD or Semantic VAD. This can be set to null to turn off, in which case the client must manually trigger model response. Server VAD means that the model will detect the start and end of speech based on audio volume and respond at the end of user speech. Semantic VAD is more advanced and uses a turn detection model (in conjuction with VAD) to semantically estimate whether the user has finished speaking, then dynamically sets a timeout based on this probability. For example, if user audio trails off with "uhhm", the model will score a low probability of turn end and wait longer for the user to continue speaking. This can be useful for more natural conversations, but may have a higher latency.
    """

    voice: str
    """
    The voice the model uses to respond.
    """


class OpenAIRealtimeStreamSessionEvents(TypedDict):
    event_id: str
    session: OpenAIRealtimeStreamSession
    type: Union[Literal["session.created"], Literal["session.updated"]]


class OpenAIRealtimeStreamResponseOutputItemContent(TypedDict, total=False):
    audio: str
    """Base64-encoded audio bytes, used for 'input_audio' content types"""
    id: str
    """The ID of the previous conversation item for reference"""
    text: str
    """The text content, used for 'input_text' and 'text' content types"""
    transcript: str
    """The transcript content, used for 'input_audio' content types"""
    type: Literal["input_audio", "input_text", "text", "item_reference", "audio"]
    """The type of content"""


class OpenAIRealtimeStreamResponseOutputItem(TypedDict, total=False):
    arguments: str
    """For function call items"""

    call_id: str
    """The ID of the function call"""

    id: str
    """The ID of the previous conversation item for reference"""

    content: List[OpenAIRealtimeStreamResponseOutputItemContent]

    name: str
    """The name of the function call"""

    object: Literal["realtime.item"]
    """The object type"""

    role: Literal["assistant", "user", "system"]
    """The role of the item, only used for 'message' items"""

    status: Literal["completed", "incomplete", "in_progress"]
    """The status of the item"""

    output: str
    """The output of the function call"""

    type: Literal["function_call", "message", "function_call_output"]
    """The type of item"""


class OpenAIRealtimeStreamResponseOutputItemAdded(TypedDict):
    type: Literal["response.output_item.added"]
    response_id: str
    output_index: int
    item: OpenAIRealtimeStreamResponseOutputItem


class OpenAIRealtimeStreamResponseBaseObject(TypedDict):
    event_id: str
    response: dict
    type: str


class OpenAIRealtimeConversationObject(TypedDict, total=False):
    id: str
    object: Required[Literal["realtime.conversation"]]


class OpenAIRealtimeConversationCreated(TypedDict, total=False):
    type: Required[Literal["conversation.created"]]
    conversation: OpenAIRealtimeConversationObject
    event_id: str


class OpenAIRealtimeConversationItemCreated(TypedDict, total=False):
    type: Required[Literal["conversation.item.created"]]
    item: OpenAIRealtimeStreamResponseOutputItem
    event_id: str
    previous_item_id: str


class OpenAIRealtimeResponseContentPart(TypedDict, total=False):
    audio: str
    """Base64-encoded audio bytes, if type is 'audio'"""

    text: str
    """The text content, if type is 'text'"""

    transcript: str
    """The transcript content, if type is 'audio'"""

    type: Literal["audio", "text"]
    """The type of content"""


class OpenAIRealtimeResponseContentPartAdded(TypedDict):
    type: Literal["response.content_part.added"]
    content_index: int
    event_id: str
    item_id: str
    output_index: int
    part: OpenAIRealtimeResponseContentPart
    response_id: str


class OpenAIRealtimeResponseDelta(TypedDict):
    content_index: int
    delta: str
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    type: Union[Literal["response.text.delta"], Literal["response.audio.delta"]]


class OpenAIRealtimeResponseTextDone(TypedDict):
    content_index: int
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    text: str
    type: Literal["response.text.done"]


class OpenAIRealtimeResponseAudioDone(TypedDict):
    content_index: int
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    type: Literal["response.audio.done"]


class OpenAIRealtimeContentPartDone(TypedDict):
    content_index: int
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    part: OpenAIRealtimeResponseContentPart
    type: Literal["response.content_part.done"]


class OpenAIRealtimeOutputItemDone(TypedDict):
    event_id: str
    item: OpenAIRealtimeStreamResponseOutputItem
    output_index: int
    response_id: str
    type: Literal["response.output_item.done"]


class OpenAIRealtimeResponseDoneObject(TypedDict, total=False):
    conversation_id: str
    id: str
    max_output_tokens: int
    metadata: dict
    modalities: list
    object: Literal["realtime.response"]
    output: List[OpenAIRealtimeStreamResponseOutputItem]
    output_audio_format: str
    status: Literal["completed", "cancelled", "failed", "incomplete"]
    status_details: dict
    temperature: float
    usage: dict  # ResponseAPIUsage
    voice: str


class OpenAIRealtimeDoneEvent(TypedDict):
    event_id: str
    response: OpenAIRealtimeResponseDoneObject
    type: Literal["response.done"]


class OpenAIRealtimeEventTypes(Enum):
    SESSION_CREATED = "session.created"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"


OpenAIRealtimeEvents = Union[
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamSessionEvents,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeConversationItemCreated,
    OpenAIRealtimeConversationCreated,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeResponseTextDone,
    OpenAIRealtimeResponseAudioDone,
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeDoneEvent,
]

OpenAIRealtimeStreamList = List[OpenAIRealtimeEvents]


class ImageGenerationRequestQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"
    STANDARD = "standard"
    HD = "hd"


class OpenAIModerationResult(BaseLiteLLMOpenAIResponseObject):
    categories: Optional[Dict]
    category_applied_input_types: Optional[Dict]
    category_scores: Optional[Dict]
    flagged: Optional[bool]


class OpenAIModerationResponse(BaseLiteLLMOpenAIResponseObject):
    """
    Response from the OpenAI Moderation API.
    """

    id: str
    """The unique identifier for the moderation request."""

    model: str
    """The model used to generate the moderation results."""

    results: List[OpenAIModerationResult]
    """A list of moderation objects."""

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class OpenAIChatCompletionLogprobsContentTopLogprobs(TypedDict, total=False):
    bytes: List
    logprob: Required[float]
    token: Required[str]


class OpenAIChatCompletionLogprobsContent(TypedDict, total=False):
    bytes: List
    logprob: Required[float]
    token: Required[str]
    top_logprobs: List[OpenAIChatCompletionLogprobsContentTopLogprobs]


class OpenAIChatCompletionLogprobs(TypedDict, total=False):
    content: List[OpenAIChatCompletionLogprobsContent]
    refusal: List[OpenAIChatCompletionLogprobsContent]


class OpenAIChatCompletionChoices(TypedDict, total=False):
    finish_reason: Required[str]
    index: Required[int]
    logprobs: Optional[OpenAIChatCompletionLogprobs]
    message: Required[ChatCompletionResponseMessage]


class OpenAIChatCompletionResponse(TypedDict, total=False):
    id: Required[str]
    object: Required[str]
    created: Required[int]
    model: Required[str]
    choices: Required[List[OpenAIChatCompletionChoices]]
    usage: Required[ChatCompletionUsageBlock]
    system_fingerprint: str
    service_tier: str


OpenAIChatCompletionFinishReason = Literal[
    "stop", "content_filter", "function_call", "tool_calls", "length"
]


class OpenAIWebSearchUserLocationApproximate(TypedDict):
    city: str
    country: str
    region: str
    timezone: str


class OpenAIWebSearchUserLocation(TypedDict):
    approximate: OpenAIWebSearchUserLocationApproximate
    type: Literal["approximate"]


class OpenAIWebSearchOptions(TypedDict, total=False):
    search_context_size: Optional[Literal["low", "medium", "high"]]
    user_location: Optional[OpenAIWebSearchUserLocation]


class OpenAIRealtimeTurnDetection(TypedDict, total=False):
    create_response: bool
    eagerness: str
    interrupt_response: bool
    prefix_padding_ms: int
    silence_duration_ms: int
    threshold: int
    type: str


class OpenAIMcpServerTool(TypedDict, total=False):
    type: Required[Literal["mcp"]]
    server_label: Required[str]
    server_url: Required[str]
    require_approval: str
    allowed_tools: Optional[List[str]]
    headers: Optional[Dict[str, str]]
