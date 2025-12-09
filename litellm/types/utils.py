import json
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union

from aiohttp import FormData
from openai._models import BaseModel as OpenAIObject
from openai.types.audio.transcription_create_params import FileTypes  # type: ignore
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.moderation import (
    Categories,
    CategoryAppliedInputTypes,
    CategoryScores,
)
from openai.types.moderation_create_response import Moderation, ModerationCreateResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Callable, Dict, Required, TypedDict, override

import litellm
from litellm._uuid import uuid
from litellm.types.llms.base import (
    BaseLiteLLMOpenAIResponseObject,
    LiteLLMPydanticObjectBase,
)
from litellm.types.mcp import MCPServerCostInfo

from ..litellm_core_utils.core_helpers import map_finish_reason
from .guardrails import GuardrailEventHooks
from .llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from .llms.base import HiddenParams
from .llms.openai import (
    Batch,
    ChatCompletionAnnotation,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    FileSearchTool,
    FineTuningJob,
    ImageURLListItem,
    OpenAIChatCompletionChunk,
    OpenAIFileObject,
    OpenAIRealtimeStreamList,
    ResponsesAPIResponse,
    WebSearchOptions,
)
from .rerank import RerankResponse

if TYPE_CHECKING:
    from .vector_stores import VectorStoreSearchResponse
else:
    VectorStoreSearchResponse = Any


def _generate_id():  # private helper function
    return "chatcmpl-" + str(uuid.uuid4())


class LiteLLMCommonStrings(Enum):
    redacted_by_litellm = "redacted by litellm. 'litellm.turn_off_message_logging=True'"
    llm_provider_not_provided = "Unmapped LLM provider for this endpoint. You passed model={model}, custom_llm_provider={custom_llm_provider}. Check supported provider and route: https://docs.litellm.ai/docs/providers"


SupportedCacheControls = ["ttl", "s-maxage", "no-cache", "no-store"]


class CostPerToken(TypedDict):
    input_cost_per_token: float
    output_cost_per_token: float


class ProviderField(TypedDict):
    field_name: str
    field_type: Literal["string"]
    field_description: str
    field_value: str


class ProviderSpecificModelInfo(TypedDict, total=False):
    supports_system_messages: Optional[bool]
    supports_response_schema: Optional[bool]
    supports_vision: Optional[bool]
    supports_function_calling: Optional[bool]
    supports_tool_choice: Optional[bool]
    supports_assistant_prefill: Optional[bool]
    supports_prompt_caching: Optional[bool]
    supports_computer_use: Optional[bool]
    supports_audio_input: Optional[bool]
    supports_embedding_image_input: Optional[bool]
    supports_audio_output: Optional[bool]
    supports_pdf_input: Optional[bool]
    supports_native_streaming: Optional[bool]
    supports_parallel_function_calling: Optional[bool]
    supports_web_search: Optional[bool]
    supports_reasoning: Optional[bool]
    supports_url_context: Optional[bool]


class SearchContextCostPerQuery(TypedDict, total=False):
    search_context_size_low: float
    search_context_size_medium: float
    search_context_size_high: float


class ModelInfoBase(ProviderSpecificModelInfo, total=False):
    key: Required[str]  # the key in litellm.model_cost which is returned

    max_tokens: Required[Optional[int]]
    max_input_tokens: Required[Optional[int]]
    max_output_tokens: Required[Optional[int]]
    input_cost_per_token: Required[float]
    input_cost_per_token_flex: Optional[float]  # OpenAI flex service tier pricing
    input_cost_per_token_priority: Optional[
        float
    ]  # OpenAI priority service tier pricing
    cache_creation_input_token_cost: Optional[float]
    cache_creation_input_token_cost_above_200k_tokens: Optional[float]
    cache_creation_input_token_cost_above_1hr: Optional[float]
    cache_read_input_token_cost: Optional[float]
    cache_read_input_token_cost_flex: Optional[
        float
    ]  # OpenAI flex service tier pricing
    cache_read_input_token_cost_priority: Optional[
        float
    ]  # OpenAI priority service tier pricing
    cache_read_input_token_cost_above_200k_tokens: Optional[float]
    input_cost_per_character: Optional[float]  # only for vertex ai models
    input_cost_per_audio_token: Optional[float]
    input_cost_per_token_above_128k_tokens: Optional[float]  # only for vertex ai models
    input_cost_per_token_above_200k_tokens: Optional[
        float
    ]  # only for vertex ai gemini-2.5-pro models
    input_cost_per_character_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    input_cost_per_query: Optional[float]  # only for rerank models
    input_cost_per_image: Optional[float]  # only for vertex ai models
    input_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
    input_cost_per_video_per_second: Optional[float]  # only for vertex ai models
    input_cost_per_second: Optional[float]  # for OpenAI Speech models
    input_cost_per_token_batches: Optional[float]
    output_cost_per_token_batches: Optional[float]
    output_cost_per_token: Required[float]
    output_cost_per_token_flex: Optional[float]  # OpenAI flex service tier pricing
    output_cost_per_token_priority: Optional[
        float
    ]  # OpenAI priority service tier pricing
    output_cost_per_character: Optional[float]  # only for vertex ai models
    output_cost_per_audio_token: Optional[float]
    output_cost_per_token_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    output_cost_per_token_above_200k_tokens: Optional[
        float
    ]  # only for vertex ai gemini-2.5-pro models
    output_cost_per_character_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    output_cost_per_image: Optional[float]
    output_cost_per_image_token: Optional[float]
    output_vector_size: Optional[int]
    output_cost_per_reasoning_token: Optional[float]
    output_cost_per_video_per_second: Optional[float]  # only for vertex ai models
    output_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
    output_cost_per_second: Optional[float]  # for OpenAI Speech models
    ocr_cost_per_page: Optional[float]  # for OCR models
    annotation_cost_per_page: Optional[float]  # for OCR models
    search_context_cost_per_query: Optional[
        SearchContextCostPerQuery
    ]  # Cost for using web search tool
    citation_cost_per_token: Optional[float]  # Cost per citation token for Perplexity
    tiered_pricing: Optional[
        List[Dict[str, Any]]
    ]  # Tiered pricing structure for models like Dashscope
    litellm_provider: Required[str]
    mode: Required[
        Literal[
            "completion",
            "embedding",
            "image_generation",
            "chat",
            "audio_transcription",
            "responses",
        ]
    ]
    tpm: Optional[int]
    rpm: Optional[int]


class ModelInfo(ModelInfoBase, total=False):
    """
    Model info for a given model, this is information found in litellm.model_prices_and_context_window.json
    """

    supported_openai_params: Required[Optional[List[str]]]


class GenericStreamingChunk(TypedDict, total=False):
    text: Required[str]
    tool_use: Optional[ChatCompletionToolCallChunk]
    is_finished: Required[bool]
    finish_reason: Required[str]
    usage: Required[Optional[ChatCompletionUsageBlock]]
    index: int

    # use this dict if you want to return any provider specific fields in the response
    provider_specific_fields: Optional[Dict[str, Any]]


from enum import Enum


class CallTypes(str, Enum):
    embedding = "embedding"
    aembedding = "aembedding"
    completion = "completion"
    acompletion = "acompletion"
    atext_completion = "atext_completion"
    text_completion = "text_completion"
    image_generation = "image_generation"
    aimage_generation = "aimage_generation"
    image_edit = "image_edit"
    aimage_edit = "aimage_edit"
    moderation = "moderation"
    amoderation = "amoderation"
    atranscription = "atranscription"
    transcription = "transcription"
    aspeech = "aspeech"
    speech = "speech"
    rerank = "rerank"
    arerank = "arerank"
    search = "search"
    asearch = "asearch"
    arealtime = "_arealtime"
    create_batch = "create_batch"
    acreate_batch = "acreate_batch"
    aretrieve_batch = "aretrieve_batch"
    retrieve_batch = "retrieve_batch"
    pass_through = "pass_through_endpoint"
    anthropic_messages = "anthropic_messages"
    get_assistants = "get_assistants"
    aget_assistants = "aget_assistants"
    create_assistants = "create_assistants"
    acreate_assistants = "acreate_assistants"
    delete_assistant = "delete_assistant"
    adelete_assistant = "adelete_assistant"
    acreate_thread = "acreate_thread"
    create_thread = "create_thread"
    aget_thread = "aget_thread"
    get_thread = "get_thread"
    a_add_message = "a_add_message"
    add_message = "add_message"
    aget_messages = "aget_messages"
    get_messages = "get_messages"
    arun_thread = "arun_thread"
    run_thread = "run_thread"
    arun_thread_stream = "arun_thread_stream"
    run_thread_stream = "run_thread_stream"
    afile_retrieve = "afile_retrieve"
    file_retrieve = "file_retrieve"
    afile_delete = "afile_delete"
    file_delete = "file_delete"
    afile_list = "afile_list"
    file_list = "file_list"
    acreate_file = "acreate_file"
    create_file = "create_file"
    afile_content = "afile_content"
    file_content = "file_content"
    create_fine_tuning_job = "create_fine_tuning_job"
    acreate_fine_tuning_job = "acreate_fine_tuning_job"

    #########################################################
    # Video Generation Call Types
    #########################################################
    create_video = "create_video"
    acreate_video = "acreate_video"
    avideo_retrieve = "avideo_retrieve"
    video_retrieve = "video_retrieve"
    avideo_content = "avideo_content"
    video_content = "video_content"
    video_remix = "video_remix"
    avideo_remix = "avideo_remix"
    video_list = "video_list"
    avideo_list = "avideo_list"
    video_retrieve_job = "video_retrieve_job"
    avideo_retrieve_job = "avideo_retrieve_job"
    video_delete = "video_delete"
    avideo_delete = "avideo_delete"
    vector_store_file_create = "vector_store_file_create"
    avector_store_file_create = "avector_store_file_create"
    vector_store_file_list = "vector_store_file_list"
    avector_store_file_list = "avector_store_file_list"
    vector_store_file_retrieve = "vector_store_file_retrieve"
    avector_store_file_retrieve = "avector_store_file_retrieve"
    vector_store_file_content = "vector_store_file_content"
    avector_store_file_content = "avector_store_file_content"
    vector_store_file_update = "vector_store_file_update"
    avector_store_file_update = "avector_store_file_update"
    vector_store_file_delete = "vector_store_file_delete"
    avector_store_file_delete = "avector_store_file_delete"
    vector_store_create = "vector_store_create"
    avector_store_create = "avector_store_create"
    vector_store_search = "vector_store_search"
    avector_store_search = "avector_store_search"

    #########################################################
    # Container Call Types
    #########################################################
    create_container = "create_container"
    acreate_container = "acreate_container"
    list_containers = "list_containers"
    alist_containers = "alist_containers"
    retrieve_container = "retrieve_container"
    aretrieve_container = "aretrieve_container"
    delete_container = "delete_container"
    adelete_container = "adelete_container"

    acancel_fine_tuning_job = "acancel_fine_tuning_job"
    cancel_fine_tuning_job = "cancel_fine_tuning_job"
    alist_fine_tuning_jobs = "alist_fine_tuning_jobs"
    list_fine_tuning_jobs = "list_fine_tuning_jobs"
    aretrieve_fine_tuning_job = "aretrieve_fine_tuning_job"
    retrieve_fine_tuning_job = "retrieve_fine_tuning_job"
    responses = "responses"
    aresponses = "aresponses"
    alist_input_items = "alist_input_items"
    llm_passthrough_route = "llm_passthrough_route"
    allm_passthrough_route = "allm_passthrough_route"

    #########################################################
    # Google GenAI Native Call Types
    #########################################################
    generate_content = "generate_content"
    agenerate_content = "agenerate_content"
    generate_content_stream = "generate_content_stream"
    agenerate_content_stream = "agenerate_content_stream"

    #########################################################
    # MCP Call Types
    #########################################################
    call_mcp_tool = "call_mcp_tool"


CallTypesLiteral = Literal[
    "embedding",
    "aembedding",
    "completion",
    "acompletion",
    "atext_completion",
    "text_completion",
    "image_generation",
    "aimage_generation",
    "image_edit",
    "aimage_edit",
    "moderation",
    "amoderation",
    "atranscription",
    "transcription",
    "aspeech",
    "speech",
    "rerank",
    "arerank",
    "search",
    "asearch",
    "_arealtime",
    "create_batch",
    "acreate_batch",
    "pass_through_endpoint",
    "anthropic_messages",
    "aretrieve_batch",
    "retrieve_batch",
    "generate_content",
    "agenerate_content",
    "generate_content_stream",
    "agenerate_content_stream",
    "ocr",
    "aocr",
    "vector_store_create",
    "avector_store_create",
    "vector_store_search",
    "avector_store_search",
    "vector_store_file_create",
    "avector_store_file_create",
    "vector_store_file_list",
    "avector_store_file_list",
    "vector_store_file_retrieve",
    "avector_store_file_retrieve",
    "vector_store_file_content",
    "avector_store_file_content",
    "vector_store_file_update",
    "avector_store_file_update",
    "vector_store_file_delete",
    "avector_store_file_delete",
    "call_mcp_tool",
    "aresponses",
    "responses",
]


class PassthroughCallTypes(Enum):
    passthrough_image_generation = "passthrough-image-generation"


class TopLogprob(OpenAIObject):
    token: str
    """The token."""

    bytes: Optional[List[int]] = None
    """A list of integers representing the UTF-8 bytes representation of the token.

    Useful in instances where characters are represented by multiple tokens and
    their byte representations must be combined to generate the correct text
    representation. Can be `null` if there is no bytes representation for the token.
    """

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """


class ChatCompletionTokenLogprob(OpenAIObject):
    token: str
    """The token."""

    bytes: Optional[List[int]] = None
    """A list of integers representing the UTF-8 bytes representation of the token.

    Useful in instances where characters are represented by multiple tokens and
    their byte representations must be combined to generate the correct text
    representation. Can be `null` if there is no bytes representation for the token.
    """

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """

    top_logprobs: List[TopLogprob]
    """List of the most likely tokens and their log probability, at this token
    position.

    In rare cases, there may be fewer than the number of requested `top_logprobs`
    returned.
    """

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)


class ChoiceLogprobs(OpenAIObject):
    content: Optional[List[ChatCompletionTokenLogprob]] = None
    """A list of message content tokens with log probability information."""

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)


class FunctionCall(OpenAIObject):
    arguments: str
    name: Optional[str] = None


class Function(OpenAIObject):
    arguments: str
    name: Optional[
        str
    ]  # can be None - openai e.g.: ChoiceDeltaToolCallFunction(arguments='{"', name=None), type=None)

    def __init__(
        self,
        arguments: Optional[Union[Dict, str]] = None,
        name: Optional[str] = None,
        **params,
    ):
        if arguments is None:
            if params.get("parameters", None) is not None and isinstance(
                params["parameters"], dict
            ):
                arguments = json.dumps(params["parameters"])
                params.pop("parameters")
            else:
                arguments = ""
        elif isinstance(arguments, Dict):
            arguments = json.dumps(arguments)
        else:
            arguments = arguments

        name = name

        # Build a dictionary with the structure your BaseModel expects
        data = {"arguments": arguments, "name": name}

        super(Function, self).__init__(**data)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class ChatCompletionDeltaToolCall(OpenAIObject):
    id: Optional[str] = None
    function: Function
    type: Optional[str] = None
    index: int

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class ChatCompletionMessageToolCall(OpenAIObject):
    def __init__(
        self,
        function: Union[Dict, Function],
        id: Optional[str] = None,
        type: Optional[str] = None,
        **params,
    ):
        super(ChatCompletionMessageToolCall, self).__init__(**params)
        if isinstance(function, Dict):
            self.function = Function(**function)
        else:
            self.function = function

        if id is not None:
            self.id = id
        else:
            self.id = f"{uuid.uuid4()}"

        if type is not None:
            self.type = type
        else:
            self.type = "function"

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


from openai.types.chat.chat_completion_audio import ChatCompletionAudio


class ChatCompletionAudioResponse(ChatCompletionAudio):
    def __init__(
        self,
        data: str,
        expires_at: int,
        transcript: str,
        id: Optional[str] = None,
        **params,
    ):
        if id is not None:
            id = id
        else:
            id = f"{uuid.uuid4()}"
        super(ChatCompletionAudioResponse, self).__init__(
            data=data, expires_at=expires_at, transcript=transcript, id=id, **params
        )

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


"""
Reference:
ChatCompletionMessage(content='This is a test', role='assistant', function_call=None, tool_calls=None))
"""


def add_provider_specific_fields(
    object: BaseModel, provider_specific_fields: Optional[Dict[str, Any]]
):
    if not provider_specific_fields:  # set if provider_specific_fields is not empty
        return
    setattr(object, "provider_specific_fields", provider_specific_fields)


class Message(OpenAIObject):
    content: Optional[str]
    role: Literal["assistant", "user", "system", "tool", "function"]
    tool_calls: Optional[List[ChatCompletionMessageToolCall]]
    function_call: Optional[FunctionCall]
    audio: Optional[ChatCompletionAudioResponse] = None
    images: Optional[List[ImageURLListItem]] = None
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ] = None
    provider_specific_fields: Optional[Dict[str, Any]] = Field(default=None)
    annotations: Optional[List[ChatCompletionAnnotation]] = None

    def __init__(
        self,
        content: Optional[str] = None,
        role: Literal["assistant", "user", "system", "tool", "function"] = "assistant",
        function_call=None,
        tool_calls: Optional[list] = None,
        audio: Optional[ChatCompletionAudioResponse] = None,
        images: Optional[List[ImageURLListItem]] = None,
        provider_specific_fields: Optional[Dict[str, Any]] = None,
        reasoning_content: Optional[str] = None,
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ] = None,
        annotations: Optional[List[ChatCompletionAnnotation]] = None,
        **params,
    ):
        init_values: Dict[str, Any] = {
            "content": content,
            "role": role or "assistant",  # handle null input
            "function_call": (
                FunctionCall(**function_call) if function_call is not None else None
            ),
            "tool_calls": (
                [
                    (
                        ChatCompletionMessageToolCall(**tool_call)
                        if isinstance(tool_call, dict)
                        else tool_call
                    )
                    for tool_call in tool_calls
                ]
                if tool_calls is not None and len(tool_calls) > 0
                else None
            ),
        }

        if audio is not None:
            init_values["audio"] = audio

        if images is not None:
            init_values["images"] = images

        if thinking_blocks is not None:
            init_values["thinking_blocks"] = thinking_blocks

        if annotations is not None:
            init_values["annotations"] = annotations

        if reasoning_content is not None:
            init_values["reasoning_content"] = reasoning_content

        super(Message, self).__init__(
            **init_values,  # type: ignore
            **params,
        )

        if audio is None:
            # delete audio from self
            # OpenAI compatible APIs like mistral API will raise an error if audio is passed in
            if hasattr(self, "audio"):
                del self.audio

        if images is None:
            if hasattr(self, "images"):
                del self.images

        if annotations is None:
            # ensure default response matches OpenAI spec
            # Some OpenAI compatible APIs raise an error if annotations are passed in
            if hasattr(self, "annotations"):
                del self.annotations

        if reasoning_content is None:
            # ensure default response matches OpenAI spec
            if hasattr(self, "reasoning_content"):
                del self.reasoning_content

        if thinking_blocks is None:
            # ensure default response matches OpenAI spec
            if hasattr(self, "thinking_blocks"):
                del self.thinking_blocks

        add_provider_specific_fields(self, provider_specific_fields)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class Delta(OpenAIObject):
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ] = None
    provider_specific_fields: Optional[Dict[str, Any]] = Field(default=None)

    def __init__(
        self,
        content=None,
        role=None,
        function_call=None,
        tool_calls=None,
        audio: Optional[ChatCompletionAudioResponse] = None,
        images: Optional[List[ImageURLListItem]] = None,
        reasoning_content: Optional[str] = None,
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ] = None,
        annotations: Optional[List[ChatCompletionAnnotation]] = None,
        **params,
    ):
        super(Delta, self).__init__(**params)
        add_provider_specific_fields(self, params.get("provider_specific_fields", {}))
        self.content = content
        self.role = role
        # Set default values and correct types
        self.function_call: Optional[Union[FunctionCall, Any]] = None
        self.tool_calls: Optional[List[Union[ChatCompletionDeltaToolCall, Any]]] = None
        self.audio: Optional[ChatCompletionAudioResponse] = None
        self.images: Optional[List[ImageURLListItem]] = None
        self.annotations: Optional[List[ChatCompletionAnnotation]] = None

        if reasoning_content is not None:
            self.reasoning_content = reasoning_content
        else:
            # ensure default response matches OpenAI spec
            del self.reasoning_content

        if thinking_blocks is not None:
            self.thinking_blocks = thinking_blocks
        else:
            # ensure default response matches OpenAI spec
            del self.thinking_blocks

        # Add annotations to the delta, ensure they are only on Delta if they exist (Match OpenAI spec)
        if annotations is not None:
            self.annotations = annotations
        else:
            del self.annotations

        if images is not None and len(images) > 0:
            self.images = images
        else:
            del self.images

        if function_call is not None and isinstance(function_call, dict):
            self.function_call = FunctionCall(**function_call)
        else:
            self.function_call = function_call
        if tool_calls is not None and isinstance(tool_calls, list):
            self.tool_calls = []
            current_index = 0
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    if tool_call.get("index", None) is None:
                        tool_call["index"] = current_index
                        current_index += 1
                    if tool_call.get("type", None) is None:
                        tool_call["type"] = "function"
                    self.tool_calls.append(ChatCompletionDeltaToolCall(**tool_call))
                elif isinstance(tool_call, ChatCompletionDeltaToolCall):
                    self.tool_calls.append(tool_call)
        else:
            self.tool_calls = tool_calls

        self.audio = audio

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Choices(OpenAIObject):
    finish_reason: str
    index: int
    message: Message
    logprobs: Optional[Union[ChoiceLogprobs, Any]] = None

    provider_specific_fields: Optional[Dict[str, Any]] = Field(default=None)

    def __init__(
        self,
        finish_reason=None,
        index=0,
        message: Optional[Union[Message, dict]] = None,
        logprobs: Optional[Union[ChoiceLogprobs, dict, Any]] = None,
        enhancements=None,
        provider_specific_fields: Optional[Dict[str, Any]] = None,
        **params,
    ):
        if finish_reason is not None:
            params["finish_reason"] = map_finish_reason(finish_reason)
        else:
            params["finish_reason"] = "stop"
        if index is not None:
            params["index"] = index
        else:
            params["index"] = 0
        if message is None:
            params["message"] = Message()
        else:
            if isinstance(message, Message):
                params["message"] = message
            elif isinstance(message, dict):
                params["message"] = Message(**message)
        if logprobs is not None:
            if isinstance(logprobs, dict):
                params["logprobs"] = ChoiceLogprobs(**logprobs)
            else:
                params["logprobs"] = logprobs
        else:
            params["logprobs"] = None
        super(Choices, self).__init__(**params)

        if enhancements is not None:
            self.enhancements = enhancements

        self.provider_specific_fields = provider_specific_fields

        if self.logprobs is None:
            del self.logprobs
        if self.provider_specific_fields is None:
            del self.provider_specific_fields

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class CompletionTokensDetailsWrapper(
    CompletionTokensDetails
):  # wrapper for older openai versions
    text_tokens: Optional[int] = None
    """Text tokens generated by the model."""

    image_tokens: Optional[int] = None
    """Image tokens generated by the model."""


class CacheCreationTokenDetails(BaseModel):
    ephemeral_5m_input_tokens: Optional[int] = None
    ephemeral_1h_input_tokens: Optional[int] = None


class PromptTokensDetailsWrapper(
    PromptTokensDetails
):  # wrapper for older openai versions
    text_tokens: Optional[int] = None
    """Text tokens sent to the model."""

    image_tokens: Optional[int] = None
    """Image tokens sent to the model."""

    web_search_requests: Optional[int] = None
    """Number of web search requests made by the tool call. Used for Anthropic to calculate web search cost."""

    character_count: Optional[int] = None
    """Character count sent to the model. Used for Vertex AI multimodal embeddings."""

    image_count: Optional[int] = None
    """Number of images sent to the model. Used for Vertex AI multimodal embeddings."""

    video_length_seconds: Optional[float] = None
    """Length of videos sent to the model. Used for Vertex AI multimodal embeddings."""

    cache_creation_tokens: Optional[int] = None
    """Number of cache creation tokens sent to the model. Used for Anthropic prompt caching."""

    cache_creation_token_details: Optional[CacheCreationTokenDetails] = None
    """Details of cache creation tokens sent to the model. Used for tracking 5m/1h cache creation tokens for Anthropic prompt caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.character_count is None:
            del self.character_count
        if self.image_count is None:
            del self.image_count
        if self.video_length_seconds is None:
            del self.video_length_seconds
        if self.web_search_requests is None:
            del self.web_search_requests
        if self.cache_creation_tokens is None:
            del self.cache_creation_tokens
        if self.cache_creation_token_details is None:
            del self.cache_creation_token_details


class ServerToolUse(BaseModel):
    web_search_requests: Optional[int]


class Usage(CompletionUsage):
    _cache_creation_input_tokens: int = PrivateAttr(
        0
    )  # hidden param for prompt caching. Might change, once openai introduces their equivalent.
    _cache_read_input_tokens: int = PrivateAttr(
        0
    )  # hidden param for prompt caching. Might change, once openai introduces their equivalent.

    server_tool_use: Optional[ServerToolUse] = None
    cost: Optional[float] = None

    completion_tokens_details: Optional[CompletionTokensDetailsWrapper] = None
    """Breakdown of tokens used in a completion."""

    prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
    """Breakdown of tokens used in the prompt."""

    def __init__(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        prompt_tokens_details: Optional[
            Union[PromptTokensDetailsWrapper, PromptTokensDetails, dict]
        ] = None,
        completion_tokens_details: Optional[
            Union[CompletionTokensDetailsWrapper, dict]
        ] = None,
        server_tool_use: Optional[ServerToolUse] = None,
        cost: Optional[float] = None,
        **params,
    ):
        # handle reasoning_tokens
        _completion_tokens_details: Optional[CompletionTokensDetailsWrapper] = None

        # First, handle existing completion_tokens_details
        if completion_tokens_details:
            if isinstance(completion_tokens_details, dict):
                _completion_tokens_details = CompletionTokensDetailsWrapper(
                    **completion_tokens_details
                )
            elif isinstance(completion_tokens_details, CompletionTokensDetails):
                _completion_tokens_details = completion_tokens_details

        # Handle reasoning_tokens and auto-calculate text_tokens if needed
        if reasoning_tokens:
            # Ensure we have a details object to work with
            if _completion_tokens_details is None:
                _completion_tokens_details = CompletionTokensDetailsWrapper()

            # Set reasoning_tokens if not already set by provider
            if _completion_tokens_details.reasoning_tokens is None:
                _completion_tokens_details.reasoning_tokens = reasoning_tokens

            # Auto-calculate text_tokens only if provider didn't set it explicitly
            # Formula: text_tokens = completion_tokens - reasoning_tokens - image_tokens - audio_tokens
            if _completion_tokens_details.text_tokens is None and completion_tokens is not None:
                calculated_text_tokens = completion_tokens - reasoning_tokens

                # Subtract other modality tokens if present
                if _completion_tokens_details.image_tokens:
                    calculated_text_tokens -= _completion_tokens_details.image_tokens
                if _completion_tokens_details.audio_tokens:
                    calculated_text_tokens -= _completion_tokens_details.audio_tokens

                # Prevent negative token counts from inconsistent data
                _completion_tokens_details.text_tokens = max(0, calculated_text_tokens)

        # handle prompt_tokens_details
        _prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None

        # guarantee prompt_token_details is always a PromptTokensDetailsWrapper
        if prompt_tokens_details:
            if isinstance(prompt_tokens_details, dict):
                _prompt_tokens_details = PromptTokensDetailsWrapper(
                    **prompt_tokens_details
                )
            elif isinstance(prompt_tokens_details, PromptTokensDetails):
                _prompt_tokens_details = PromptTokensDetailsWrapper(
                    **prompt_tokens_details.model_dump()
                )
            elif isinstance(prompt_tokens_details, PromptTokensDetailsWrapper):
                _prompt_tokens_details = prompt_tokens_details

        ## DEEPSEEK MAPPING ##
        if "prompt_cache_hit_tokens" in params and isinstance(
            params["prompt_cache_hit_tokens"], int
        ):
            if _prompt_tokens_details is None:
                _prompt_tokens_details = PromptTokensDetailsWrapper(
                    cached_tokens=params["prompt_cache_hit_tokens"]
                )
            else:
                _prompt_tokens_details.cached_tokens = params["prompt_cache_hit_tokens"]

        ## ANTHROPIC MAPPING ##
        if "cache_read_input_tokens" in params and isinstance(
            params["cache_read_input_tokens"], int
        ):
            if _prompt_tokens_details is None:
                _prompt_tokens_details = PromptTokensDetailsWrapper(
                    cached_tokens=params["cache_read_input_tokens"]
                )
            else:
                _prompt_tokens_details.cached_tokens = params["cache_read_input_tokens"]

        if "cache_creation_input_tokens" in params and isinstance(
            params["cache_creation_input_tokens"], int
        ):
            if _prompt_tokens_details is None:
                _prompt_tokens_details = PromptTokensDetailsWrapper(
                    cache_creation_tokens=params["cache_creation_input_tokens"]
                )
            else:
                _prompt_tokens_details.cache_creation_tokens = params[
                    "cache_creation_input_tokens"
                ]

        super().__init__(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
            completion_tokens_details=_completion_tokens_details or None,
            prompt_tokens_details=_prompt_tokens_details or None,
        )

        if server_tool_use is not None:
            self.server_tool_use = server_tool_use
        else:  # maintain openai compatibility in usage object if possible
            del self.server_tool_use

        if cost is not None:
            self.cost = cost
        else:
            del self.cost

        ## ANTHROPIC MAPPING ##
        if "cache_creation_input_tokens" in params and isinstance(
            params["cache_creation_input_tokens"], int
        ):
            self._cache_creation_input_tokens = params["cache_creation_input_tokens"]

        if "cache_read_input_tokens" in params and isinstance(
            params["cache_read_input_tokens"], int
        ):
            self._cache_read_input_tokens = params["cache_read_input_tokens"]

        ## DEEPSEEK MAPPING ##
        if "prompt_cache_hit_tokens" in params and isinstance(
            params["prompt_cache_hit_tokens"], int
        ):
            self._cache_read_input_tokens = params["prompt_cache_hit_tokens"]

        for k, v in params.items():
            setattr(self, k, v)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class StreamingChoices(OpenAIObject):
    def __init__(
        self,
        finish_reason=None,
        index=0,
        delta: Optional[Delta] = None,
        logprobs=None,
        enhancements=None,
        **params,
    ):
        # Fix Perplexity return both delta and message cause OpenWebUI repect text
        # https://github.com/BerriAI/litellm/issues/8455
        params.pop("message", None)
        super(StreamingChoices, self).__init__(**params)
        if finish_reason:
            self.finish_reason = map_finish_reason(finish_reason)
        else:
            self.finish_reason = None
        self.index = index
        if delta is not None:
            if isinstance(delta, Delta):
                self.delta = delta
            elif isinstance(delta, dict):
                self.delta = Delta(**delta)
        else:
            self.delta = Delta()
        if enhancements is not None:
            self.enhancements = enhancements

        if logprobs is not None and isinstance(logprobs, dict):
            self.logprobs = ChoiceLogprobs(**logprobs)
        else:
            self.logprobs = logprobs  # type: ignore

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class StreamingChatCompletionChunk(OpenAIChatCompletionChunk):
    def __init__(self, **kwargs):
        new_choices = []
        for choice in kwargs["choices"]:
            new_choice = StreamingChoices(**choice).model_dump()
            new_choices.append(new_choice)
        kwargs["choices"] = new_choices

        super().__init__(**kwargs)


class ModelResponseBase(OpenAIObject):
    id: str
    """A unique identifier for the completion."""

    created: int
    """The Unix timestamp (in seconds) of when the completion was created."""

    model: Optional[str] = None
    """The model used for completion."""

    object: str
    """The object type, which is always "text_completion" """

    system_fingerprint: Optional[str] = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    _hidden_params: dict = {}

    _response_headers: Optional[dict] = None


class ModelResponseStream(ModelResponseBase):
    choices: List[StreamingChoices]
    provider_specific_fields: Optional[Dict[str, Any]] = Field(default=None)

    def __init__(
        self,
        choices: Optional[
            Union[List[StreamingChoices], Union[StreamingChoices, dict, BaseModel]]
        ] = None,
        id: Optional[str] = None,
        created: Optional[int] = None,
        provider_specific_fields: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if choices is not None and isinstance(choices, list):
            new_choices = []
            for choice in choices:
                _new_choice = None
                if isinstance(choice, StreamingChoices):
                    _new_choice = choice
                elif isinstance(choice, dict):
                    _new_choice = StreamingChoices(**choice)
                elif isinstance(choice, BaseModel):
                    _new_choice = StreamingChoices(**choice.model_dump())
                new_choices.append(_new_choice)
            kwargs["choices"] = new_choices
        else:
            kwargs["choices"] = [StreamingChoices()]

        if id is None:
            id = _generate_id()
        else:
            id = id
        if created is None:
            created = int(time.time())
        else:
            created = created

        if (
            "usage" in kwargs
            and kwargs["usage"] is not None
            and isinstance(kwargs["usage"], dict)
        ):
            kwargs["usage"] = Usage(**kwargs["usage"])

        kwargs["id"] = id
        kwargs["created"] = created
        kwargs["object"] = "chat.completion.chunk"
        kwargs["provider_specific_fields"] = provider_specific_fields

        super().__init__(**kwargs)

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


class ModelResponse(ModelResponseBase):
    choices: List[Union[Choices, StreamingChoices]]
    """The list of completion choices the model generated for the input prompt."""

    def __init__(  # noqa: PLR0915
        self,
        id=None,
        choices=None,
        created=None,
        model=None,
        object=None,
        system_fingerprint=None,
        usage=None,
        stream=None,
        stream_options=None,
        response_ms=None,
        hidden_params=None,
        _response_headers=None,
        **params,
    ) -> None:
        if stream is not None and stream is True:
            object = "chat.completion.chunk"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    _new_choice = None
                    if isinstance(choice, StreamingChoices):
                        _new_choice = choice
                    elif isinstance(choice, dict):
                        _new_choice = StreamingChoices(**choice)
                    elif isinstance(choice, BaseModel):
                        _new_choice = StreamingChoices(**choice.model_dump())
                    new_choices.append(_new_choice)
                choices = new_choices
            else:
                choices = [StreamingChoices()]
        else:
            object = "chat.completion"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    if isinstance(choice, Choices):
                        _new_choice = choice  # type: ignore
                    elif isinstance(choice, dict):
                        _new_choice = Choices(**choice)  # type: ignore
                    else:
                        _new_choice = choice
                    new_choices.append(_new_choice)
                choices = new_choices
            else:
                choices = [Choices()]
        if id is None:
            id = _generate_id()
        else:
            id = id
        if created is None:
            created = int(time.time())
        else:
            created = created
        model = model
        if usage is not None:
            if isinstance(usage, dict):
                usage = Usage(**usage)
            else:
                usage = usage
        elif stream is None or stream is False:
            usage = Usage()
        if hidden_params:
            self._hidden_params = hidden_params

        if _response_headers:
            self._response_headers = _response_headers

        init_values = {
            "id": id,
            "choices": choices,
            "created": created,
            "model": model,
            "object": object,
            "system_fingerprint": system_fingerprint,
        }

        if usage is not None:
            init_values["usage"] = usage

        super().__init__(
            **init_values,
            **params,
        )

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


class Embedding(OpenAIObject):
    embedding: Union[list, str] = []
    index: int
    object: Literal["embedding"]

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class EmbeddingResponse(OpenAIObject):
    model: Optional[str] = None
    """The model used for embedding."""

    data: List
    """The actual embedding value"""

    object: Literal["list"]
    """The object type, which is always "list" """

    usage: Optional[Usage] = None
    """Usage statistics for the embedding request."""

    _hidden_params: dict = {}
    _response_headers: Optional[Dict] = None
    _response_ms: Optional[float] = None

    def __init__(
        self,
        model: Optional[str] = None,
        usage: Optional[Usage] = None,
        response_ms=None,
        data: Optional[Union[List, List[Embedding]]] = None,
        hidden_params=None,
        _response_headers=None,
        **params,
    ):
        object = "list"
        if response_ms:
            _response_ms = response_ms
        else:
            _response_ms = None
        if data:
            data = data
        else:
            data = []

        if usage:
            usage = usage
        else:
            usage = Usage()

        if _response_headers:
            self._response_headers = _response_headers

        model = model
        super().__init__(model=model, object=object, data=data, usage=usage)  # type: ignore

        if hidden_params:
            self._hidden_params = hidden_params

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class Logprobs(OpenAIObject):
    text_offset: Optional[List[int]]
    token_logprobs: Optional[List[Union[float, None]]]
    tokens: Optional[List[str]]
    top_logprobs: Optional[List[Union[Dict[str, float], None]]]


class TextChoices(OpenAIObject):
    def __init__(self, finish_reason=None, index=0, text=None, logprobs=None, **params):
        super(TextChoices, self).__init__(**params)
        if finish_reason:
            self.finish_reason = map_finish_reason(finish_reason)
        else:
            self.finish_reason = None
        self.index = index
        if text is not None:
            self.text = text
        else:
            self.text = None
        if logprobs is None:
            self.logprobs = None
        else:
            if isinstance(logprobs, dict):
                self.logprobs = Logprobs(**logprobs)
            else:
                self.logprobs = logprobs

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class TextCompletionResponse(OpenAIObject):
    """
    {
        "id": response["id"],
        "object": "text_completion",
        "created": response["created"],
        "model": response["model"],
        "choices": [
        {
            "text": response["choices"][0]["message"]["content"],
            "index": response["choices"][0]["index"],
            "logprobs": transformed_logprobs,
            "finish_reason": response["choices"][0]["finish_reason"]
        }
        ],
        "usage": response["usage"]
    }
    """

    id: str
    object: str
    created: int
    model: Optional[str]
    choices: List[TextChoices]
    usage: Optional[Usage]
    _response_ms: Optional[int] = None
    _hidden_params: HiddenParams

    def __init__(
        self,
        id=None,
        choices=None,
        created=None,
        model=None,
        usage=None,
        stream=False,
        response_ms=None,
        object=None,
        **params,
    ):
        if stream:
            object = "text_completion.chunk"
            choices = [TextChoices()]
        else:
            object = "text_completion"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    _new_choice = None
                    if isinstance(choice, TextChoices):
                        _new_choice = choice
                    elif isinstance(choice, dict):
                        _new_choice = TextChoices(**choice)
                    new_choices.append(_new_choice)
                choices = new_choices
            else:
                choices = [TextChoices()]
        if object is not None:
            object = object
        if id is None:
            id = _generate_id()
        else:
            id = id
        if created is None:
            created = int(time.time())
        else:
            created = created

        model = model
        if usage:
            usage = usage
        else:
            usage = Usage()

        super(TextCompletionResponse, self).__init__(
            id=id,  # type: ignore
            object=object,  # type: ignore
            created=created,  # type: ignore
            model=model,  # type: ignore
            choices=choices,  # type: ignore
            usage=usage,  # type: ignore
            **params,
        )

        if response_ms:
            self._response_ms = response_ms
        else:
            self._response_ms = None
        self._hidden_params = HiddenParams()

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


from openai.types.images_response import Image as OpenAIImage


class ImageObject(OpenAIImage):
    """
    Represents the url or the content of an image generated by the OpenAI API.

    Attributes:
    b64_json: The base64-encoded JSON of the generated image, if response_format is b64_json.
    url: The URL of the generated image, if response_format is url (default).
    revised_prompt: The prompt that was used to generate the image, if there was any revision to the prompt.

    https://platform.openai.com/docs/api-reference/images/object
    """

    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

    def __init__(self, b64_json=None, url=None, revised_prompt=None, **kwargs):
        super().__init__(b64_json=b64_json, url=url, revised_prompt=revised_prompt)  # type: ignore

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class ImageUsageInputTokensDetails(BaseLiteLLMOpenAIResponseObject):
    image_tokens: int
    """The number of image tokens in the input prompt."""

    text_tokens: int
    """The number of text tokens in the input prompt."""


class ImageUsage(BaseLiteLLMOpenAIResponseObject):
    input_tokens: int
    """The number of tokens (images and text) in the input prompt."""

    input_tokens_details: ImageUsageInputTokensDetails
    """The input tokens detailed information for the image generation."""

    output_tokens: int
    """The number of image tokens in the output image."""

    total_tokens: int
    """The total number of tokens (images and text) used for the image generation."""


from openai.types.images_response import ImagesResponse as OpenAIImageResponse


class ImageResponse(OpenAIImageResponse, BaseLiteLLMOpenAIResponseObject):
    _hidden_params: dict = {}

    usage: Optional[ImageUsage] = None  # type: ignore
    """
    Users might use litellm with older python versions, we don't want this to break for them.
    Happens when their OpenAIImageResponse has the old OpenAI usage class.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def __init__(
        self,
        created: Optional[int] = None,
        data: Optional[List[ImageObject]] = None,
        response_ms=None,
        usage: Optional[ImageUsage] = None,
        hidden_params: Optional[dict] = None,
        **kwargs,
    ):
        if response_ms:
            _response_ms = response_ms
        else:
            _response_ms = None
        if data:
            data = data
        else:
            data = []

        if created:
            created = created
        else:
            created = int(time.time())

        _data: List[OpenAIImage] = []
        for d in data:
            if isinstance(d, dict):
                _data.append(ImageObject(**d))
            elif isinstance(d, BaseModel):
                _data.append(ImageObject(**d.model_dump()))

        _usage = usage or ImageUsage(
            input_tokens=0,
            input_tokens_details=ImageUsageInputTokensDetails(
                image_tokens=0,
                text_tokens=0,
            ),
            output_tokens=0,
            total_tokens=0,
        )
        super().__init__(created=created, data=_data, usage=_usage)  # type: ignore

        self.quality = kwargs.get("quality", None)
        self.output_format = kwargs.get("output_format", None)
        self.size = kwargs.get("size", None)
        self._hidden_params = hidden_params or {}

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class TranscriptionUsageDurationObject(BaseModel):
    type: Literal["duration"]
    seconds: int


class TranscriptionUsageInputTokenDetailsObject(BaseModel):
    audio_tokens: int
    text_tokens: int


class TranscriptionUsageTokensObject(BaseModel):
    type: Literal["tokens"]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: TranscriptionUsageInputTokenDetailsObject


class TranscriptionResponse(OpenAIObject):
    text: Optional[str] = None
    usage: Optional[
        Union[TranscriptionUsageDurationObject, TranscriptionUsageTokensObject]
    ] = None

    _hidden_params: dict = {}
    _response_headers: Optional[dict] = None

    def __init__(self, text=None):
        super().__init__(text=text)  # type: ignore

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class GenericImageParsingChunk(TypedDict):
    type: str
    media_type: str
    data: str


class ResponseFormatChunk(TypedDict, total=False):
    type: Required[Literal["json_object", "text"]]
    response_schema: dict


class LoggedLiteLLMParams(TypedDict, total=False):
    force_timeout: Optional[float]
    custom_llm_provider: Optional[str]
    api_base: Optional[str]
    litellm_call_id: Optional[str]
    model_alias_map: Optional[dict]
    metadata: Optional[dict]
    model_info: Optional[dict]
    proxy_server_request: Optional[dict]
    acompletion: Optional[bool]
    preset_cache_key: Optional[str]
    no_log: Optional[bool]
    input_cost_per_second: Optional[float]
    input_cost_per_token: Optional[float]
    output_cost_per_token: Optional[float]
    output_cost_per_second: Optional[float]
    cooldown_time: Optional[float]


class AdapterCompletionStreamWrapper:
    def __init__(self, completion_stream):
        self.completion_stream = completion_stream

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        try:
            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                return chunk
            raise StopIteration
        except StopIteration:
            raise StopIteration
        except Exception as e:
            print(f"AdapterCompletionStreamWrapper - {e}")  # noqa

    async def __anext__(self):
        try:
            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                return chunk
            raise StopIteration
        except StopIteration:
            raise StopAsyncIteration


class StandardLoggingUserAPIKeyMetadata(TypedDict):
    user_api_key_hash: Optional[str]  # hash of the litellm virtual key used
    user_api_key_alias: Optional[str]
    user_api_key_spend: Optional[float]
    user_api_key_max_budget: Optional[float]
    user_api_key_budget_reset_at: Optional[str]
    user_api_key_org_id: Optional[str]
    user_api_key_team_id: Optional[str]
    user_api_key_user_id: Optional[str]
    user_api_key_user_email: Optional[str]
    user_api_key_team_alias: Optional[str]
    user_api_key_end_user_id: Optional[str]
    user_api_key_request_route: Optional[str]
    user_api_key_auth_metadata: Optional[Dict[str, str]]


class StandardLoggingMCPToolCall(TypedDict, total=False):
    name: str
    """
    Name of the tool to call
    """
    arguments: dict
    """
    Arguments to pass to the tool
    """
    result: dict
    """
    Result of the tool call
    """

    mcp_server_name: Optional[str]
    """
    Name of the MCP server that the tool call was made to
    """

    mcp_server_logo_url: Optional[str]
    """
    Optional logo URL of the MCP server that the tool call was made to

    (this is to render the logo on the logs page on litellm ui)
    """

    namespaced_tool_name: Optional[str]
    """
    Namespaced tool name of the MCP tool that the tool call was made to

    Includes the server name prefix if it exists - eg. `deepwiki-mcp/get_page_content`
    """

    mcp_server_cost_info: Optional[MCPServerCostInfo]
    """
    Cost per query for the MCP server tool call
    """


class StandardLoggingVectorStoreRequest(TypedDict, total=False):
    """
    Logging information for a vector store request/payload
    """

    vector_store_id: Optional[str]
    """
    ID of the vector store
    """

    custom_llm_provider: Optional[str]
    """
    Custom LLM provider the vector store is associated with eg. bedrock, openai, anthropic, etc.
    """

    query: Optional[str]
    """
    Query to the vector store
    """

    vector_store_search_response: Optional[VectorStoreSearchResponse]
    """
    OpenAI format vector store search response
    """

    start_time: Optional[float]
    """
    Start time of the vector store request
    """

    end_time: Optional[float]
    """
    End time of the vector store request
    """


class StandardBuiltInToolsParams(TypedDict, total=False):
    """
    Standard built-in OpenAItools parameters

    This is used to calculate the cost of built-in tools, insert any standard built-in tools parameters here

    OpenAI charges users based on the `web_search_options` parameter
    """

    web_search_options: Optional[WebSearchOptions]
    file_search: Optional[FileSearchTool]


class StandardLoggingPromptManagementMetadata(TypedDict):
    prompt_id: str
    prompt_variables: Optional[dict]
    prompt_integration: str


class StandardLoggingMetadata(StandardLoggingUserAPIKeyMetadata):
    """
    Specific metadata k,v pairs logged to integration for easier cost tracking and prompt management
    """

    spend_logs_metadata: Optional[
        dict
    ]  # special param to log k,v pairs to spendlogs for a call
    requester_ip_address: Optional[str]
    requester_metadata: Optional[dict]
    requester_custom_headers: Optional[
        Dict[str, str]
    ]  # Log any custom (`x-`) headers sent by the client to the proxy.
    prompt_management_metadata: Optional[StandardLoggingPromptManagementMetadata]
    mcp_tool_call_metadata: Optional[StandardLoggingMCPToolCall]
    vector_store_request_metadata: Optional[List[StandardLoggingVectorStoreRequest]]
    applied_guardrails: Optional[List[str]]
    usage_object: Optional[dict]
    cold_storage_object_key: Optional[
        str
    ]  # S3/GCS object key for cold storage retrieval


class StandardLoggingAdditionalHeaders(TypedDict, total=False):
    x_ratelimit_limit_requests: int
    x_ratelimit_limit_tokens: int
    x_ratelimit_remaining_requests: int
    x_ratelimit_remaining_tokens: int


class StandardLoggingHiddenParams(TypedDict):
    model_id: Optional[
        str
    ]  # id of the model in the router, separates multiple models with the same name but different credentials
    cache_key: Optional[str]
    api_base: Optional[str]
    response_cost: Optional[str]
    litellm_overhead_time_ms: Optional[float]
    additional_headers: Optional[StandardLoggingAdditionalHeaders]
    batch_models: Optional[List[str]]
    litellm_model_name: Optional[str]  # the model name sent to the provider by litellm
    usage_object: Optional[dict]


class StandardLoggingModelInformation(TypedDict):
    model_map_key: str
    model_map_value: Optional[ModelInfo]


class StandardLoggingModelCostFailureDebugInformation(TypedDict, total=False):
    """
    Debug information, if cost tracking fails.

    Avoid logging sensitive information like response or optional params
    """

    error_str: Required[str]
    traceback_str: Required[str]
    model: str
    cache_hit: Optional[bool]
    custom_llm_provider: Optional[str]
    base_model: Optional[str]
    call_type: str
    custom_pricing: Optional[bool]


class StandardLoggingPayloadErrorInformation(TypedDict, total=False):
    error_code: Optional[str]
    error_class: Optional[str]
    llm_provider: Optional[str]
    traceback: Optional[str]
    error_message: Optional[str]


class GuardrailMode(TypedDict, total=False):
    tags: Optional[Dict[str, str]]
    default: Optional[str]


GuardrailStatus = Literal[
    "success", "guardrail_intervened", "guardrail_failed_to_respond", "not_run"
]


class StandardLoggingGuardrailInformation(TypedDict, total=False):
    guardrail_name: Optional[str]
    guardrail_provider: Optional[str]
    guardrail_mode: Optional[
        Union[GuardrailEventHooks, List[GuardrailEventHooks], GuardrailMode]
    ]
    guardrail_request: Optional[dict]
    guardrail_response: Optional[Union[dict, str, List[dict]]]
    guardrail_status: GuardrailStatus
    start_time: Optional[float]
    end_time: Optional[float]
    duration: Optional[float]
    """
    Duration of the guardrail in seconds
    """

    masked_entity_count: Optional[Dict[str, int]]
    """
    Count of masked entities
    {
        "CREDIT_CARD": 2,
        "PHONE": 1
    }
    """


StandardLoggingPayloadStatus = Literal["success", "failure"]


class CachingDetails(TypedDict):
    """
    Track all caching related metrics, fields for a given request
    """

    cache_hit: Optional[bool]
    """
    Whether the request hit the cache
    """
    cache_duration_ms: Optional[float]
    """
    Duration for reading from cache
    """


class CostBreakdown(TypedDict, total=False):
    """
    Detailed cost breakdown for a request
    """

    input_cost: float  # Cost of input/prompt tokens
    output_cost: (
        float  # Cost of output/completion tokens (includes reasoning if applicable)
    )
    total_cost: float  # Total cost (input + output + tool usage)
    tool_usage_cost: float  # Cost of usage of built-in tools
    original_cost: float  # Cost before discount (optional)
    discount_percent: float  # Discount percentage applied (e.g., 0.05 = 5%) (optional)
    discount_amount: float  # Discount amount in USD (optional)


class StandardLoggingPayloadStatusFields(TypedDict, total=False):
    """Status fields for easy filtering and analytics"""

    llm_api_status: StandardLoggingPayloadStatus
    """Status of the LLM API call - 'success' if completed, 'failure' if errored"""
    guardrail_status: GuardrailStatus
    """
    Status of guardrail execution:
    - 'success': Guardrail ran and allowed content through
    - 'guardrail_intervened': Guardrail blocked or modified content
    - 'guardrail_failed_to_respond': Guardrail had technical failure
    - 'not_run': No guardrail was run
    """


class StandardLoggingPayload(TypedDict):
    id: str
    trace_id: str  # Trace multiple LLM calls belonging to same overall request (e.g. fallbacks/retries)
    call_type: str
    stream: Optional[bool]
    response_cost: float
    cost_breakdown: Optional[CostBreakdown]  # Detailed cost breakdown
    response_cost_failure_debug_info: Optional[
        StandardLoggingModelCostFailureDebugInformation
    ]
    status: StandardLoggingPayloadStatus
    status_fields: StandardLoggingPayloadStatusFields
    custom_llm_provider: Optional[str]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    startTime: float  # Note: making this camelCase was a mistake, everything should be snake case
    endTime: float
    completionStartTime: float
    response_time: float
    model_map_information: StandardLoggingModelInformation
    model: str
    model_id: Optional[str]
    model_group: Optional[str]
    api_base: str
    metadata: StandardLoggingMetadata
    cache_hit: Optional[bool]
    cache_key: Optional[str]
    saved_cache_cost: float
    request_tags: list
    end_user: Optional[str]
    requester_ip_address: Optional[str]
    messages: Optional[Union[str, list, dict]]
    response: Optional[Union[str, list, dict]]
    error_str: Optional[str]
    error_information: Optional[StandardLoggingPayloadErrorInformation]
    model_parameters: dict
    hidden_params: StandardLoggingHiddenParams
    guardrail_information: Optional[List[StandardLoggingGuardrailInformation]]
    standard_built_in_tools_params: Optional[StandardBuiltInToolsParams]


from typing import AsyncIterator, Iterator


class CustomStreamingDecoder:
    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[
        Optional[Union[GenericStreamingChunk, StreamingChatCompletionChunk]]
    ]:
        raise NotImplementedError

    def iter_bytes(
        self, iterator: Iterator[bytes]
    ) -> Iterator[Optional[Union[GenericStreamingChunk, StreamingChatCompletionChunk]]]:
        raise NotImplementedError


class StandardPassThroughResponseObject(TypedDict):
    response: Union[str, dict]


OPENAI_RESPONSE_HEADERS = [
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
]


class StandardCallbackDynamicParams(TypedDict, total=False):
    # Langfuse dynamic params
    langfuse_public_key: Optional[str]
    langfuse_secret: Optional[str]
    langfuse_secret_key: Optional[str]
    langfuse_host: Optional[str]

    # Langfuse prompt version
    langfuse_prompt_version: Optional[int]

    # GCS dynamic params
    gcs_bucket_name: Optional[str]
    gcs_path_service_account: Optional[str]

    # Langsmith dynamic params
    langsmith_api_key: Optional[str]
    langsmith_project: Optional[str]
    langsmith_base_url: Optional[str]
    langsmith_sampling_rate: Optional[float]

    # Humanloop dynamic params
    humanloop_api_key: Optional[str]

    # Arize dynamic params
    arize_api_key: Optional[str]
    arize_space_key: Optional[str]
    arize_space_id: Optional[str]

    # PostHog dynamic params
    posthog_api_key: Optional[str]
    posthog_api_url: Optional[str]

    # Logging settings
    turn_off_message_logging: Optional[bool]  # when true will not log messages
    litellm_disabled_callbacks: Optional[List[str]]


class CustomPricingLiteLLMParams(BaseModel):
    ## CUSTOM PRICING ##
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    input_cost_per_second: Optional[float] = None
    output_cost_per_second: Optional[float] = None
    input_cost_per_pixel: Optional[float] = None
    output_cost_per_pixel: Optional[float] = None

    # Include all ModelInfoBase fields as optional
    # This allows any model_info parameter to be set in litellm_params
    input_cost_per_token_flex: Optional[float] = None
    input_cost_per_token_priority: Optional[float] = None
    cache_creation_input_token_cost: Optional[float] = None
    cache_creation_input_token_cost_above_1hr: Optional[float] = None
    cache_creation_input_token_cost_above_200k_tokens: Optional[float] = None
    cache_creation_input_audio_token_cost: Optional[float] = None
    cache_read_input_token_cost: Optional[float] = None
    cache_read_input_token_cost_flex: Optional[float] = None
    cache_read_input_token_cost_priority: Optional[float] = None
    cache_read_input_token_cost_above_200k_tokens: Optional[float] = None
    cache_read_input_audio_token_cost: Optional[float] = None
    input_cost_per_character: Optional[float] = None
    input_cost_per_character_above_128k_tokens: Optional[float] = None
    input_cost_per_audio_token: Optional[float] = None
    input_cost_per_token_cache_hit: Optional[float] = None
    input_cost_per_token_above_128k_tokens: Optional[float] = None
    input_cost_per_token_above_200k_tokens: Optional[float] = None
    input_cost_per_query: Optional[float] = None
    input_cost_per_image: Optional[float] = None
    input_cost_per_image_above_128k_tokens: Optional[float] = None
    input_cost_per_audio_per_second: Optional[float] = None
    input_cost_per_audio_per_second_above_128k_tokens: Optional[float] = None
    input_cost_per_video_per_second: Optional[float] = None
    input_cost_per_video_per_second_above_128k_tokens: Optional[float] = None
    input_cost_per_video_per_second_above_15s_interval: Optional[float] = None
    input_cost_per_video_per_second_above_8s_interval: Optional[float] = None
    input_cost_per_token_batches: Optional[float] = None
    output_cost_per_token_batches: Optional[float] = None
    output_cost_per_token_flex: Optional[float] = None
    output_cost_per_token_priority: Optional[float] = None
    output_cost_per_character: Optional[float] = None
    output_cost_per_audio_token: Optional[float] = None
    output_cost_per_token_above_128k_tokens: Optional[float] = None
    output_cost_per_token_above_200k_tokens: Optional[float] = None
    output_cost_per_character_above_128k_tokens: Optional[float] = None
    output_cost_per_image: Optional[float] = None
    output_cost_per_image_token: Optional[float] = None
    output_cost_per_reasoning_token: Optional[float] = None
    output_cost_per_video_per_second: Optional[float] = None
    output_cost_per_audio_per_second: Optional[float] = None
    search_context_cost_per_query: Optional[Dict[str, Any]] = None
    citation_cost_per_token: Optional[float] = None
    tiered_pricing: Optional[List[Dict[str, Any]]] = None


all_litellm_params = (
    [
        "metadata",
        "litellm_metadata",
        "litellm_trace_id",
        "litellm_request_debug",
        "guardrails",
        "tags",
        "acompletion",
        "aimg_generation",
        "atext_completion",
        "text_completion",
        "caching",
        "mock_response",
        "mock_timeout",
        "disable_add_transform_inline_image_block",
        "litellm_proxy_rate_limit_response",
        "api_key",
        "api_version",
        "prompt_id",
        "provider_specific_header",
        "prompt_variables",
        "prompt_version",
        "api_base",
        "force_timeout",
        "logger_fn",
        "verbose",
        "custom_llm_provider",
        "model_file_id_mapping",
        "litellm_logging_obj",
        "litellm_call_id",
        "use_client",
        "id",
        "fallbacks",
        "azure",
        "headers",
        "model_list",
        "num_retries",
        "context_window_fallback_dict",
        "retry_policy",
        "retry_strategy",
        "roles",
        "final_prompt_value",
        "bos_token",
        "eos_token",
        "request_timeout",
        "complete_response",
        "self",
        "client",
        "rpm",
        "tpm",
        "max_parallel_requests",
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_second",
        "output_cost_per_second",
        "hf_model_name",
        "model_info",
        "proxy_server_request",
        "secret_fields",
        "preset_cache_key",
        "caching_groups",
        "ttl",
        "cache",
        "no-log",
        "base_model",
        "stream_timeout",
        "supports_system_message",
        "region_name",
        "allowed_model_region",
        "model_config",
        "fastest_response",
        "cooldown_time",
        "cache_key",
        "max_retries",
        "azure_ad_token_provider",
        "tenant_id",
        "client_id",
        "azure_username",
        "azure_password",
        "azure_scope",
        "client_secret",
        "user_continue_message",
        "configurable_clientside_auth_params",
        "weight",
        "ensure_alternating_roles",
        "assistant_continue_message",
        "user_continue_message",
        "fallback_depth",
        "max_fallbacks",
        "max_budget",
        "budget_duration",
        "use_in_pass_through",
        "merge_reasoning_content_in_choices",
        "litellm_credential_name",
        "allowed_openai_params",
        "litellm_session_id",
        "use_litellm_proxy",
        "prompt_label",
        "shared_session",
        "search_tool_name",
    ]
    + list(StandardCallbackDynamicParams.__annotations__.keys())
    + list(CustomPricingLiteLLMParams.model_fields.keys())
)


class KeyGenerationConfig(TypedDict, total=False):
    required_params: List[
        str
    ]  # specify params that must be present in the key generation request


class TeamUIKeyGenerationConfig(KeyGenerationConfig):
    allowed_team_member_roles: List[str]


class PersonalUIKeyGenerationConfig(KeyGenerationConfig):
    allowed_user_roles: List[str]


class StandardKeyGenerationConfig(TypedDict, total=False):
    team_key_generation: TeamUIKeyGenerationConfig
    personal_key_generation: PersonalUIKeyGenerationConfig


class BudgetConfig(BaseModel):
    max_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None

    def __init__(self, **data: Any) -> None:
        # Map time_period to budget_duration if present
        if "time_period" in data:
            data["budget_duration"] = data.pop("time_period")

        # Map budget_limit to max_budget if present
        if "budget_limit" in data:
            data["max_budget"] = data.pop("budget_limit")

        super().__init__(**data)


GenericBudgetConfigType = Dict[str, BudgetConfig]


class LlmProviders(str, Enum):
    OPENAI = "openai"
    OPENAI_LIKE = "openai_like"  # embedding only
    JINA_AI = "jina_ai"
    XAI = "xai"
    CUSTOM_OPENAI = "custom_openai"
    TEXT_COMPLETION_OPENAI = "text-completion-openai"
    COHERE = "cohere"
    COHERE_CHAT = "cohere_chat"
    CLARIFAI = "clarifai"
    ANTHROPIC = "anthropic"
    ANTHROPIC_TEXT = "anthropic_text"
    BYTEZ = "bytez"
    REPLICATE = "replicate"
    RUNWAYML = "runwayml"
    HUGGINGFACE = "huggingface"
    TOGETHER_AI = "together_ai"
    OPENROUTER = "openrouter"
    DATAROBOT = "datarobot"
    VERTEX_AI = "vertex_ai"
    VERTEX_AI_BETA = "vertex_ai_beta"
    GEMINI = "gemini"
    AI21 = "ai21"
    BASETEN = "baseten"
    AZURE = "azure"
    AZURE_TEXT = "azure_text"
    AZURE_AI = "azure_ai"
    SAGEMAKER = "sagemaker"
    SAGEMAKER_CHAT = "sagemaker_chat"
    BEDROCK = "bedrock"
    VLLM = "vllm"
    NLP_CLOUD = "nlp_cloud"
    PETALS = "petals"
    OOBABOOGA = "oobabooga"
    OLLAMA = "ollama"
    OLLAMA_CHAT = "ollama_chat"
    DEEPINFRA = "deepinfra"
    PERPLEXITY = "perplexity"
    MISTRAL = "mistral"
    MILVUS = "milvus"
    GROQ = "groq"
    NVIDIA_NIM = "nvidia_nim"
    CEREBRAS = "cerebras"
    AI21_CHAT = "ai21_chat"
    VOLCENGINE = "volcengine"
    CODESTRAL = "codestral"
    TEXT_COMPLETION_CODESTRAL = "text-completion-codestral"
    DASHSCOPE = "dashscope"
    MOONSHOT = "moonshot"
    V0 = "v0"
    MORPH = "morph"
    LAMBDA_AI = "lambda_ai"
    DEEPSEEK = "deepseek"
    SAMBANOVA = "sambanova"
    MARITALK = "maritalk"
    VOYAGE = "voyage"
    CLOUDFLARE = "cloudflare"
    XINFERENCE = "xinference"
    FIREWORKS_AI = "fireworks_ai"
    FRIENDLIAI = "friendliai"
    FEATHERLESS_AI = "featherless_ai"
    WATSONX = "watsonx"
    WATSONX_TEXT = "watsonx_text"
    TRITON = "triton"
    PREDIBASE = "predibase"
    DATABRICKS = "databricks"
    EMPOWER = "empower"
    GITHUB = "github"
    COMPACTIFAI = "compactifai"
    DOCKER_MODEL_RUNNER = "docker_model_runner"
    CUSTOM = "custom"
    LITELLM_PROXY = "litellm_proxy"
    HOSTED_VLLM = "hosted_vllm"
    LLAMAFILE = "llamafile"
    LM_STUDIO = "lm_studio"
    GALADRIEL = "galadriel"
    NEBIUS = "nebius"
    INFINITY = "infinity"
    DEEPGRAM = "deepgram"
    ELEVENLABS = "elevenlabs"
    NOVITA = "novita"
    AIOHTTP_OPENAI = "aiohttp_openai"
    LANGFUSE = "langfuse"
    HUMANLOOP = "humanloop"
    TOPAZ = "topaz"
    ASSEMBLYAI = "assemblyai"
    GITHUB_COPILOT = "github_copilot"
    SNOWFLAKE = "snowflake"
    GRADIENT_AI = "gradient_ai"
    LLAMA = "meta_llama"
    NSCALE = "nscale"
    PG_VECTOR = "pg_vector"
    HYPERBOLIC = "hyperbolic"
    RECRAFT = "recraft"
    FAL_AI = "fal_ai"
    HEROKU = "heroku"
    AIML = "aiml"
    COMETAPI = "cometapi"
    OCI = "oci"
    AUTO_ROUTER = "auto_router"
    VERCEL_AI_GATEWAY = "vercel_ai_gateway"
    DOTPROMPT = "dotprompt"
    WANDB = "wandb"
    OVHCLOUD = "ovhcloud"
    LEMONADE = "lemonade"


# Create a set of all provider values for quick lookup
LlmProvidersSet = {provider.value for provider in LlmProviders}


class SearchProviders(str, Enum):
    """
    Enum for search provider types.
    Separate from LlmProviders for semantic clarity.
    """

    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    PARALLEL_AI = "parallel_ai"
    EXA_AI = "exa_ai"
    GOOGLE_PSE = "google_pse"
    DATAFORSEO = "dataforseo"
    FIRECRAWL = "firecrawl"
    SEARXNG = "searxng"


# Create a set of all search provider values for quick lookup
SearchProvidersSet = {provider.value for provider in SearchProviders}


class LiteLLMLoggingBaseClass:
    """
    Base class for logging pre and post call

    Meant to simplify type checking for logging obj.
    """

    def pre_call(self, input, api_key, model=None, additional_args={}):
        pass

    def post_call(
        self, original_response, input=None, api_key=None, additional_args={}
    ):
        pass


class TokenCountResponse(LiteLLMPydanticObjectBase):
    total_tokens: int
    request_model: str
    model_used: str
    tokenizer_type: str
    original_response: Optional[dict] = None
    """
    Original Response from upstream API call - if an API call was made for token counting
    """


class CustomHuggingfaceTokenizer(TypedDict):
    identifier: str
    revision: str  # usually 'main'
    auth_token: Optional[str]


class LITELLM_IMAGE_VARIATION_PROVIDERS(Enum):
    """
    Try using an enum for endpoints. This should make it easier to track what provider is supported for what endpoint.
    """

    OPENAI = LlmProviders.OPENAI.value
    TOPAZ = LlmProviders.TOPAZ.value


class HttpHandlerRequestFields(TypedDict, total=False):
    data: dict  # request body
    params: dict  # query params
    files: dict  # file uploads
    content: Any  # raw content


class ProviderSpecificHeader(TypedDict):
    custom_llm_provider: str
    extra_headers: dict


class SelectTokenizerResponse(TypedDict):
    type: Literal["openai_tokenizer", "huggingface_tokenizer"]
    tokenizer: Any


class LiteLLMFineTuningJob(FineTuningJob):
    _hidden_params: dict = {}
    seed: Optional[int] = None  # type: ignore

    def __init__(self, **kwargs):
        if "error" in kwargs and kwargs["error"] is not None:
            # check if error is all None - if so, set error to None
            if all(value is None for value in kwargs["error"].values()):
                kwargs["error"] = None
        super().__init__(**kwargs)
        self._hidden_params = kwargs.get("_hidden_params", {})


class LiteLLMBatch(Batch):
    _hidden_params: dict = {}
    usage: Optional[Usage] = None  # type: ignore[assignment]

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


class LiteLLMRealtimeStreamLoggingObject(LiteLLMPydanticObjectBase):
    results: OpenAIRealtimeStreamList
    usage: Usage
    _hidden_params: dict = {}

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


class RawRequestTypedDict(TypedDict, total=False):
    raw_request_api_base: Optional[str]
    raw_request_body: Optional[dict]
    raw_request_headers: Optional[dict]
    error: Optional[str]


class CredentialBase(BaseModel):
    credential_name: str
    credential_info: dict


class CredentialItem(CredentialBase):
    credential_values: dict


class CreateCredentialItem(CredentialBase):
    credential_values: Optional[dict] = None
    model_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def check_credential_params(cls, values):
        if not values.get("credential_values") and not values.get("model_id"):
            raise ValueError("Either credential_values or model_id must be set")
        return values


class ExtractedFileData(TypedDict):
    """
    TypedDict for storing processed file data

    Attributes:
        filename: Name of the file if provided
        content: The file content in bytes
        content_type: MIME type of the file
        headers: Any additional headers for the file
    """

    filename: Optional[str]
    content: bytes
    content_type: Optional[str]
    headers: Mapping[str, str]


class SpecialEnums(Enum):
    LITELM_MANAGED_FILE_ID_PREFIX = "litellm_proxy"
    LITELLM_MANAGED_FILE_COMPLETE_STR = "litellm_proxy:{};unified_id,{};target_model_names,{};llm_output_file_id,{};llm_output_file_model_id,{}"

    LITELLM_MANAGED_RESPONSE_COMPLETE_STR = (
        "litellm:custom_llm_provider:{};model_id:{};response_id:{}"
    )

    LITELLM_MANAGED_BATCH_COMPLETE_STR = "litellm_proxy;model_id:{};llm_batch_id:{}"

    LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR = (
        "litellm_proxy:responses_api:response_id:{};user_id:{};team_id:{}"
    )

    LITELLM_MANAGED_GENERIC_RESPONSE_COMPLETE_STR = "litellm_proxy;model_id:{};generic_response_id:{}"  # generic implementation of 'managed batches' - used for finetuning and any future work.

    LITELLM_MANAGED_VIDEO_COMPLETE_STR = (
        "litellm:custom_llm_provider:{};model_id:{};video_id:{}"
    )


class ServiceTier(Enum):
    """Enum for service tier types used in cost calculations."""

    FLEX = "flex"
    PRIORITY = "priority"


LLMResponseTypes = Union[
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    OpenAIFileObject,
    LiteLLMBatch,
    LiteLLMFineTuningJob,
    AnthropicMessagesResponse,
    ResponsesAPIResponse,
]


class DynamicPromptManagementParamLiteral(str, Enum):
    """
    If any of these params are passed, the user is trying to use dynamic prompt management
    """

    CACHE_CONTROL_INJECTION_POINTS = "cache_control_injection_points"
    KNOWLEDGE_BASES = "knowledge_bases"
    VECTOR_STORE_IDS = "vector_store_ids"

    @classmethod
    def list_all_params(cls):
        return [param.value for param in cls]


class CallbacksByType(TypedDict):
    success: List[str]
    failure: List[str]
    success_and_failure: List[str]


CostResponseTypes = Union[
    ModelResponse,
    TextCompletionResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
]


class PriorityReservationDict(TypedDict, total=False):
    """
    Dictionary format for priority reservation values.

    Used in litellm.priority_reservation to specify how much capacity to reserve
    for each priority level. Supports three formats:
    1. Percentage-based: {"type": "percent", "value": 0.9} -> 90% of capacity
    2. RPM-based: {"type": "rpm", "value": 900} -> 900 requests per minute
    3. TPM-based: {"type": "tpm", "value": 900000} -> 900,000 tokens per minute

    Attributes:
        type: The type of value - "percent", "rpm", or "tpm". Defaults to "percent".
        value: The numeric value. For percent (0.0-1.0), for rpm/tpm (absolute value).
    """

    type: Literal["percent", "rpm", "tpm"]
    value: float


class PriorityReservationSettings(BaseModel):
    """
    Settings for priority-based rate limiting reservation.

    Defines what priority to assign to keys without explicit priority metadata.
    The priority_reservation mapping is configured separately via litellm.priority_reservation.
    """

    default_priority: float = Field(
        default=0.25,
        description="Priority level to assign to API keys without explicit priority metadata. Should match a key in litellm.priority_reservation.",
    )

    saturation_threshold: float = Field(
        default=0.50,
        description="Saturation threshold (0.0-1.0) at which strict priority enforcement begins. Below this threshold, generous mode allows priority borrowing. Above this threshold, strict mode enforces normalized priority limits.",
    )

    model_config = ConfigDict(protected_namespaces=())
