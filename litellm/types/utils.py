import json
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    get_args,
)

import httpx
from openai._models import BaseModel as OpenAIObject
from openai.types.audio.transcription_create_params import (
    FileTypes as FileTypes,
)
from openai.types.chat.chat_completion import ChatCompletion as ChatCompletion
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.moderation import Categories as Categories
from openai.types.moderation import (
    CategoryAppliedInputTypes as CategoryAppliedInputTypes,
)
from openai.types.moderation import CategoryScores as CategoryScores
from openai.types.moderation_create_response import Moderation as Moderation
from openai.types.moderation_create_response import (
    ModerationCreateResponse as ModerationCreateResponse,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    field_serializer,
    field_validator,
)
from typing_extensions import ReadOnly, Required, TypedDict

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.base import (
    BaseLiteLLMOpenAIResponseObject,
    LiteLLMPydanticObjectBase,
)
from litellm.types.mcp import MCPServerCostInfo

from ..litellm_core_utils.core_helpers import map_finish_reason, process_response_headers
from .agents import LiteLLMSendMessageResponse
from .guardrails import GuardrailEventHooks
from .llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from .llms.base import HiddenParams
from .llms.openai import (
    AllMessageValues,
    Batch,
    ChatCompletionAnnotation,
    ChatCompletionReasoningItem,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
    ChatCompletionUsageBlock,
    FileSearchTool,
    FineTuningJob,
    ImageURLListItem,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionFinishReason,
    OpenAIFileObject,
    OpenAIRealtimeStreamList,
    ResponsesAPIResponse,
    WebSearchOptions,
)
from .rerank import RerankResponse as RerankResponse

if TYPE_CHECKING:
    from .vector_stores import VectorStoreSearchResponse
else:
    VectorStoreSearchResponse = Any


def _generate_id():  # private helper function
    return "chatcmpl-" + str(uuid.uuid4())


class SafeAttributeModel:
    """
    A base model that provides safe attribute access.
    """

    def __delattr__(self, name) -> None:
        # Dropping an unset optional field stored in __dict__ goes straight to
        # object.__delattr__, skipping pydantic's __delattr__ whose per-call
        # class getattr lookup and _check_frozen dominate response construction.
        try:
            if (
                name in type(self).__pydantic_fields__
                and name in self.__dict__
                and not type(self).model_config.get("frozen")
            ):
                object.__delattr__(self, name)
                return
            super().__delattr__(name)
        except AttributeError:
            # noop if attribute does not exist
            pass


class LiteLLMCommonStrings(Enum):
    redacted_by_litellm = "redacted by litellm. 'litellm.turn_off_message_logging=True'"
    llm_provider_not_provided = "Unmapped LLM provider for this endpoint. You passed model={model}, custom_llm_provider={custom_llm_provider}. Check supported provider and route: https://docs.litellm.ai/docs/providers"


SupportedCacheControls: Final = ["ttl", "s-maxage", "no-cache", "no-store"]


class CostPerToken(TypedDict, total=False):
    # Required base rates — kept under total=False so we can mark them
    # Required individually while leaving the cache rates NotRequired.
    input_cost_per_token: Required[float]
    output_cost_per_token: Required[float]
    cache_read_input_token_cost: float
    cache_creation_input_token_cost: float


class ProviderField(TypedDict):
    field_name: str
    field_type: Literal["string"]
    field_description: str
    field_value: str


class ProviderSpecificModelInfo(TypedDict, total=False):
    supports_system_messages: bool | None
    supports_response_schema: bool | None
    supports_vision: bool | None
    supports_function_calling: bool | None
    supports_tool_choice: bool | None
    supports_assistant_prefill: bool | None
    supports_prompt_caching: bool | None
    supports_prompt_cache_breakpoint: ReadOnly[bool | None]
    supports_computer_use: bool | None
    supports_audio_input: bool | None
    supports_embedding_image_input: bool | None
    supports_audio_output: bool | None
    supports_pdf_input: bool | None
    supports_native_streaming: bool | None
    supports_native_structured_output: bool | None
    supports_parallel_function_calling: bool | None
    supports_web_search: bool | None
    supports_reasoning: bool | None
    supports_adaptive_thinking: bool | None
    supports_legacy_thinking: ReadOnly[bool | None]
    thinking_always_on: ReadOnly[bool | None]
    supports_tool_search: bool | None
    supports_mid_conversation_system: bool | None
    supports_url_context: bool | None
    supports_none_reasoning_effort: bool | None
    supports_minimal_reasoning_effort: bool | None
    supports_low_reasoning_effort: bool | None
    supports_xhigh_reasoning_effort: bool | None
    supports_max_reasoning_effort: bool | None
    supports_output_config: bool | None
    supports_image_size: bool | None
    bedrock_output_config_effort_ceiling: Literal["low", "medium", "high", "max", "xhigh"] | None
    bedrock_converse_supports_strict_tools: bool | None


class SearchContextCostPerQuery(TypedDict, total=False):
    search_context_size_low: float
    search_context_size_medium: float
    search_context_size_high: float


class AgenticLoopParams(TypedDict, total=False):
    """
    Parameters passed to agentic loop hooks (e.g., WebSearch interception).

    Stored in logging_obj.model_call_details["agentic_loop_params"] to provide
    agentic hooks with the original request context needed for follow-up calls.
    """

    model: str
    """The model string with provider prefix (e.g., 'bedrock/invoke/...')"""

    custom_llm_provider: str
    """The LLM provider name (e.g., 'bedrock', 'anthropic')"""


class ModelInfoBase(ProviderSpecificModelInfo, total=False):
    key: Required[str]  # the key in litellm.model_cost which is returned

    max_tokens: Required[int | None]
    max_input_tokens: Required[int | None]
    max_output_tokens: Required[int | None]
    input_cost_per_token: Required[float | None]
    input_cost_per_token_flex: float | None  # OpenAI flex service tier pricing
    input_cost_per_token_priority: float | None  # OpenAI priority service tier pricing
    input_cost_per_token_ultrafast: ReadOnly[float | None]  # OpenAI ultrafast service tier pricing
    cache_creation_input_token_cost: float | None
    cache_creation_input_token_cost_above_200k_tokens: float | None
    cache_creation_input_token_cost_above_272k_tokens: float | None
    cache_creation_input_token_cost_above_272k_tokens_priority: float | None
    cache_creation_input_token_cost_above_272k_tokens_flex: float | None
    cache_creation_input_token_cost_above_1hr: float | None
    cache_creation_input_token_cost_flex: float | None  # OpenAI flex service tier pricing
    cache_creation_input_token_cost_priority: float | None  # OpenAI priority service tier pricing
    cache_creation_input_token_cost_ultrafast: ReadOnly[float | None]  # OpenAI ultrafast service tier pricing
    cache_read_input_token_cost: float | None
    cache_read_input_token_cost_flex: float | None  # OpenAI flex service tier pricing
    cache_read_input_token_cost_priority: float | None  # OpenAI priority service tier pricing
    cache_read_input_token_cost_ultrafast: ReadOnly[float | None]  # OpenAI ultrafast service tier pricing
    cache_read_input_token_cost_above_200k_tokens: float | None
    cache_read_input_token_cost_above_200k_tokens_priority: float | None
    cache_read_input_token_cost_above_272k_tokens: float | None
    cache_read_input_token_cost_above_272k_tokens_priority: float | None
    cache_read_input_token_cost_above_272k_tokens_flex: float | None
    cache_read_input_token_cost_above_512k_tokens: float | None
    # Smallest prefix this model will actually cache, whatever caching mechanism its provider uses.
    # Absent means the provider-agnostic default applies; see MINIMUM_PROMPT_CACHE_TOKEN_COUNT.
    prompt_cache_min_tokens: int | None
    input_cost_per_character: float | None  # only for vertex ai models
    input_cost_per_audio_token: float | None
    input_cost_per_token_above_128k_tokens: float | None  # only for vertex ai models
    input_cost_per_token_above_200k_tokens: float | None  # only for vertex ai gemini-2.5-pro models
    input_cost_per_token_above_200k_tokens_priority: float | None
    input_cost_per_token_above_272k_tokens: float | None  # GPT-5.4/5.4-pro: prompts >272K priced at 2x input
    input_cost_per_token_above_272k_tokens_priority: float | None
    input_cost_per_token_above_272k_tokens_flex: float | None
    input_cost_per_token_above_512k_tokens: float | None  # MiniMax-M3: prompts >512K priced at 2x input
    input_cost_per_character_above_128k_tokens: float | None  # only for vertex ai models
    input_cost_per_query: float | None  # only for rerank models
    input_cost_per_image: float | None  # only for vertex ai models
    input_cost_per_image_token: float | None  # for gpt-image-1 and similar models
    input_cost_per_video_token: float | None  # for gemini omni models with video input
    input_cost_per_audio_per_second: float | None  # only for vertex ai models
    input_cost_per_video_per_second: float | None  # only for vertex ai models
    input_cost_per_second: float | None  # for OpenAI Speech models
    input_cost_per_token_batches: float | None
    output_cost_per_token_batches: float | None
    output_cost_per_token: Required[float | None]
    output_cost_per_token_flex: float | None  # OpenAI flex service tier pricing
    output_cost_per_token_priority: float | None  # OpenAI priority service tier pricing
    output_cost_per_token_ultrafast: ReadOnly[float | None]  # OpenAI ultrafast service tier pricing
    regional_processing_uplift_multiplier_eu: (
        float | None
    )  # OpenAI EU data-residency uplift multiplier applied to all token costs (e.g. 1.10 = +10%)
    regional_processing_uplift_multiplier_us: (
        float | None
    )  # OpenAI US data-residency uplift multiplier applied to all token costs (e.g. 1.10 = +10%)
    regional_endpoint_uplift_multiplier: ReadOnly[
        float | None
    ]  # Vertex AI non-global (regional) endpoint uplift multiplier applied to all token costs (e.g. 1.10 = +10%)
    output_cost_per_character: float | None  # only for vertex ai models
    output_cost_per_audio_token: float | None
    output_cost_per_token_above_128k_tokens: float | None  # only for vertex ai models
    output_cost_per_token_above_200k_tokens: float | None  # only for vertex ai gemini-2.5-pro models
    output_cost_per_token_above_200k_tokens_priority: float | None
    output_cost_per_token_above_272k_tokens: float | None  # GPT-5.4/5.4-pro: prompts >272K priced at 1.5x output
    output_cost_per_token_above_272k_tokens_priority: float | None
    output_cost_per_token_above_272k_tokens_flex: float | None
    output_cost_per_token_above_512k_tokens: float | None  # MiniMax-M3: prompts >512K priced at 2x output
    output_cost_per_character_above_128k_tokens: float | None  # only for vertex ai models
    output_cost_per_image: float | None
    output_cost_per_image_token: float | None
    output_cost_per_video_token: float | None  # for gemini omni models with video output
    output_vector_size: int | None
    output_cost_per_reasoning_token: float | None
    output_cost_per_reasoning_token_flex: float | None
    output_cost_per_reasoning_token_priority: float | None
    output_cost_per_video_per_second: float | None  # only for vertex ai models
    output_cost_per_audio_per_second: float | None  # only for vertex ai models
    output_cost_per_second: float | None  # for OpenAI Speech models
    output_cost_per_second_1080p: (
        float | None
    )  # video_generation tier: key output_cost_per_second_<resolution> (e.g. 1080p, 720p)
    output_cost_per_second_480p: ReadOnly[float | None]
    output_cost_per_second_4k: ReadOnly[float | None]
    ocr_cost_per_page: float | None  # for OCR models
    ocr_cost_per_credit: float | None  # for OCR models priced by credit
    annotation_cost_per_page: float | None  # for OCR models
    search_context_cost_per_query: SearchContextCostPerQuery | None  # Cost for using web search tool
    web_search_billing_unit: (
        Literal["per_query", "per_prompt"] | None
    )  # "per_query" (Gemini 3.x) or "per_prompt" (Gemini 2.x)
    google_maps_grounding_cost_per_query: ReadOnly[float | None]
    citation_cost_per_token: float | None  # Cost per citation token for Perplexity
    tiered_pricing: list[dict[str, Any]] | None  # Tiered pricing structure for models like Dashscope
    litellm_provider: Required[str]
    mode: Required[
        Literal[
            "completion",
            "embedding",
            "image_generation",
            "chat",
            "audio_transcription",
            "responses",
            "ocr",
            "realtime",
        ]
    ]
    supported_endpoints: list[str] | None
    use_openai_responses_path: bool | None
    tpm: int | None
    rpm: int | None
    provider_specific_entry: dict[str, float] | None
    uses_embed_content: bool | None


class ModelInfo(ModelInfoBase, total=False):
    """
    Model info for a given model, this is information found in litellm.model_prices_and_context_window.json
    """

    supported_openai_params: Required[list[str] | None]


class GenericStreamingChunk(TypedDict, total=False):
    text: Required[str]
    tool_use: ChatCompletionToolCallChunk | None
    is_finished: Required[bool]
    finish_reason: Required[str]
    usage: Required[ChatCompletionUsageBlock | None]
    index: int

    # use this dict if you want to return any provider specific fields in the response
    provider_specific_fields: dict[str, Any] | None


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
    aresponses_websocket = "_aresponses_websocket"
    create_batch = "create_batch"
    acreate_batch = "acreate_batch"
    aretrieve_batch = "aretrieve_batch"
    retrieve_batch = "retrieve_batch"
    acancel_batch = "acancel_batch"
    cancel_batch = "cancel_batch"
    pass_through = "pass_through_endpoint"
    anthropic_messages = "anthropic_messages"
    aanthropic_messages = "aanthropic_messages"
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
    video_create_character = "video_create_character"
    avideo_create_character = "avideo_create_character"
    video_get_character = "video_get_character"
    avideo_get_character = "avideo_get_character"
    video_edit = "video_edit"
    avideo_edit = "avideo_edit"
    video_extension = "video_extension"
    avideo_extension = "avideo_extension"
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

    ingest = "ingest"
    aingest = "aingest"
    query = "query"
    aquery = "aquery"

    #########################################################
    # Google Interactions API Call Types
    #########################################################
    create_interaction = "create_interaction"
    acreate_interaction = "acreate_interaction"

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
    list_container_files = "list_container_files"
    alist_container_files = "alist_container_files"
    upload_container_file = "upload_container_file"
    aupload_container_file = "aupload_container_file"
    create_sandbox = "create_sandbox"
    acreate_sandbox = "acreate_sandbox"
    delete_sandbox = "delete_sandbox"
    adelete_sandbox = "adelete_sandbox"
    run_code = "run_code"
    arun_code = "arun_code"
    code_interpreter_tool = "code_interpreter_tool"
    acode_interpreter_tool = "acode_interpreter_tool"

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
    # OCR Call Types
    #########################################################
    ocr = "ocr"
    aocr = "aocr"

    #########################################################
    # MCP Call Types
    #########################################################
    call_mcp_tool = "call_mcp_tool"
    list_mcp_tools = "list_mcp_tools"

    #########################################################
    # A2A Call Types
    #########################################################
    asend_message = "asend_message"
    send_message = "send_message"

    #########################################################
    # Claude Code Call Types
    #########################################################
    acreate_skill = "acreate_skill"


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
    "allm_passthrough_route",
    "anthropic_messages",
    "aanthropic_messages",
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
    "list_mcp_tools",
    "asend_message",
    "send_message",
    "aresponses",
    "responses",
    "acreate_skill",
    "acreate_realtime_client_secret",
    "arealtime_calls",
    "acreate_realtime_transcription_session",
]

# Mapping of API routes to their corresponding call types
API_ROUTE_TO_CALL_TYPES: Final[Mapping[str, Sequence[CallTypes]]] = {
    # Chat Completions
    "/chat/completions": [CallTypes.acompletion, CallTypes.completion],
    "/v1/chat/completions": [CallTypes.acompletion, CallTypes.completion],
    "/engines/{model}/chat/completions": [CallTypes.acompletion, CallTypes.completion],
    "/openai/deployments/{model}/chat/completions": [
        CallTypes.acompletion,
        CallTypes.completion,
    ],
    # Text Completions
    "/completions": [CallTypes.atext_completion, CallTypes.text_completion],
    "/v1/completions": [CallTypes.atext_completion, CallTypes.text_completion],
    "/engines/{model}/completions": [
        CallTypes.atext_completion,
        CallTypes.text_completion,
    ],
    "/openai/deployments/{model}/completions": [
        CallTypes.atext_completion,
        CallTypes.text_completion,
    ],
    # Embeddings
    "/embeddings": [CallTypes.aembedding, CallTypes.embedding],
    "/v1/embeddings": [CallTypes.aembedding, CallTypes.embedding],
    "/engines/{model}/embeddings": [CallTypes.aembedding, CallTypes.embedding],
    "/openai/deployments/{model}/embeddings": [
        CallTypes.aembedding,
        CallTypes.embedding,
    ],
    # Image Generation
    "/images/generations": [CallTypes.aimage_generation, CallTypes.image_generation],
    "/v1/images/generations": [CallTypes.aimage_generation, CallTypes.image_generation],
    "/engines/{model}/images/generations": [
        CallTypes.aimage_generation,
        CallTypes.image_generation,
    ],
    "/openai/deployments/{model}/images/generations": [
        CallTypes.aimage_generation,
        CallTypes.image_generation,
    ],
    # Image Edits
    "/images/edits": [CallTypes.aimage_edit, CallTypes.image_edit],
    "/v1/images/edits": [CallTypes.aimage_edit, CallTypes.image_edit],
    # Audio Transcriptions
    "/audio/transcriptions": [CallTypes.atranscription, CallTypes.transcription],
    "/v1/audio/transcriptions": [CallTypes.atranscription, CallTypes.transcription],
    # Audio Speech
    "/audio/speech": [CallTypes.aspeech, CallTypes.speech],
    "/v1/audio/speech": [CallTypes.aspeech, CallTypes.speech],
    # Moderations
    "/moderations": [CallTypes.amoderation, CallTypes.moderation],
    "/v1/moderations": [CallTypes.amoderation, CallTypes.moderation],
    # Rerank
    "/rerank": [CallTypes.arerank, CallTypes.rerank],
    "/v1/rerank": [CallTypes.arerank, CallTypes.rerank],
    "/v2/rerank": [CallTypes.arerank, CallTypes.rerank],
    # Search
    "/search": [CallTypes.asearch, CallTypes.search],
    "/v1/search": [CallTypes.asearch, CallTypes.search],
    # Batches
    "/batches": [CallTypes.acreate_batch, CallTypes.create_batch],
    "/v1/batches": [CallTypes.acreate_batch, CallTypes.create_batch],
    "/batches/{batch_id}": [CallTypes.aretrieve_batch, CallTypes.retrieve_batch],
    "/v1/batches/{batch_id}": [CallTypes.aretrieve_batch, CallTypes.retrieve_batch],
    # Files
    "/files": [
        CallTypes.acreate_file,
        CallTypes.create_file,
        CallTypes.afile_list,
        CallTypes.file_list,
    ],
    "/v1/files": [
        CallTypes.acreate_file,
        CallTypes.create_file,
        CallTypes.afile_list,
        CallTypes.file_list,
    ],
    "/files/{file_id}": [
        CallTypes.afile_retrieve,
        CallTypes.file_retrieve,
        CallTypes.afile_delete,
        CallTypes.file_delete,
    ],
    "/v1/files/{file_id}": [
        CallTypes.afile_retrieve,
        CallTypes.file_retrieve,
        CallTypes.afile_delete,
        CallTypes.file_delete,
    ],
    "/files/{file_id}/content": [CallTypes.afile_content, CallTypes.file_content],
    "/v1/files/{file_id}/content": [CallTypes.afile_content, CallTypes.file_content],
    # Assistants
    "/assistants": [
        CallTypes.aget_assistants,
        CallTypes.get_assistants,
        CallTypes.acreate_assistants,
        CallTypes.create_assistants,
    ],
    "/v1/assistants": [
        CallTypes.aget_assistants,
        CallTypes.get_assistants,
        CallTypes.acreate_assistants,
        CallTypes.create_assistants,
    ],
    "/assistants/{assistant_id}": [
        CallTypes.adelete_assistant,
        CallTypes.delete_assistant,
    ],
    "/v1/assistants/{assistant_id}": [
        CallTypes.adelete_assistant,
        CallTypes.delete_assistant,
    ],
    # Threads
    "/threads": [CallTypes.acreate_thread, CallTypes.create_thread],
    "/v1/threads": [CallTypes.acreate_thread, CallTypes.create_thread],
    "/threads/{thread_id}": [CallTypes.aget_thread, CallTypes.get_thread],
    "/v1/threads/{thread_id}": [CallTypes.aget_thread, CallTypes.get_thread],
    # Thread Messages
    "/threads/{thread_id}/messages": [
        CallTypes.a_add_message,
        CallTypes.add_message,
        CallTypes.aget_messages,
        CallTypes.get_messages,
    ],
    "/v1/threads/{thread_id}/messages": [
        CallTypes.a_add_message,
        CallTypes.add_message,
        CallTypes.aget_messages,
        CallTypes.get_messages,
    ],
    # Thread Runs
    "/threads/{thread_id}/runs": [
        CallTypes.arun_thread,
        CallTypes.run_thread,
        CallTypes.arun_thread_stream,
        CallTypes.run_thread_stream,
    ],
    "/v1/threads/{thread_id}/runs": [
        CallTypes.arun_thread,
        CallTypes.run_thread,
        CallTypes.arun_thread_stream,
        CallTypes.run_thread_stream,
    ],
    # Fine-tuning Jobs
    "/fine_tuning/jobs": [
        CallTypes.acreate_fine_tuning_job,
        CallTypes.create_fine_tuning_job,
        CallTypes.alist_fine_tuning_jobs,
        CallTypes.list_fine_tuning_jobs,
    ],
    "/v1/fine_tuning/jobs": [
        CallTypes.acreate_fine_tuning_job,
        CallTypes.create_fine_tuning_job,
        CallTypes.alist_fine_tuning_jobs,
        CallTypes.list_fine_tuning_jobs,
    ],
    "/fine_tuning/jobs/{fine_tuning_job_id}": [
        CallTypes.aretrieve_fine_tuning_job,
        CallTypes.retrieve_fine_tuning_job,
    ],
    "/v1/fine_tuning/jobs/{fine_tuning_job_id}": [
        CallTypes.aretrieve_fine_tuning_job,
        CallTypes.retrieve_fine_tuning_job,
    ],
    "/fine_tuning/jobs/{fine_tuning_job_id}/cancel": [
        CallTypes.acancel_fine_tuning_job,
        CallTypes.cancel_fine_tuning_job,
    ],
    "/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel": [
        CallTypes.acancel_fine_tuning_job,
        CallTypes.cancel_fine_tuning_job,
    ],
    # Video Generation
    "/videos": [
        CallTypes.acreate_video,
        CallTypes.create_video,
        CallTypes.avideo_list,
        CallTypes.video_list,
    ],
    "/v1/videos": [
        CallTypes.acreate_video,
        CallTypes.create_video,
        CallTypes.avideo_list,
        CallTypes.video_list,
    ],
    "/videos/{video_id}": [
        CallTypes.avideo_retrieve,
        CallTypes.video_retrieve,
        CallTypes.avideo_delete,
        CallTypes.video_delete,
    ],
    "/v1/videos/{video_id}": [
        CallTypes.avideo_retrieve,
        CallTypes.video_retrieve,
        CallTypes.avideo_delete,
        CallTypes.video_delete,
    ],
    "/videos/{video_id}/content": [CallTypes.avideo_content, CallTypes.video_content],
    "/v1/videos/{video_id}/content": [
        CallTypes.avideo_content,
        CallTypes.video_content,
    ],
    "/videos/{video_id}/remix": [CallTypes.avideo_remix, CallTypes.video_remix],
    "/v1/videos/{video_id}/remix": [CallTypes.avideo_remix, CallTypes.video_remix],
    "/videos/characters": [
        CallTypes.avideo_create_character,
        CallTypes.video_create_character,
    ],
    "/v1/videos/characters": [
        CallTypes.avideo_create_character,
        CallTypes.video_create_character,
    ],
    "/videos/characters/{character_id}": [
        CallTypes.avideo_get_character,
        CallTypes.video_get_character,
    ],
    "/v1/videos/characters/{character_id}": [
        CallTypes.avideo_get_character,
        CallTypes.video_get_character,
    ],
    "/videos/edits": [CallTypes.avideo_edit, CallTypes.video_edit],
    "/v1/videos/edits": [CallTypes.avideo_edit, CallTypes.video_edit],
    "/videos/extensions": [CallTypes.avideo_extension, CallTypes.video_extension],
    "/v1/videos/extensions": [CallTypes.avideo_extension, CallTypes.video_extension],
    # Vector Stores
    "/vector_stores": [CallTypes.avector_store_create, CallTypes.vector_store_create],
    "/v1/vector_stores": [
        CallTypes.avector_store_create,
        CallTypes.vector_store_create,
    ],
    "/vector_stores/{vector_store_id}/search": [
        CallTypes.avector_store_search,
        CallTypes.vector_store_search,
    ],
    "/v1/vector_stores/{vector_store_id}/search": [
        CallTypes.avector_store_search,
        CallTypes.vector_store_search,
    ],
    "/vector_stores/{vector_store_id}/files": [
        CallTypes.avector_store_file_create,
        CallTypes.vector_store_file_create,
        CallTypes.avector_store_file_list,
        CallTypes.vector_store_file_list,
    ],
    "/v1/vector_stores/{vector_store_id}/files": [
        CallTypes.avector_store_file_create,
        CallTypes.vector_store_file_create,
        CallTypes.avector_store_file_list,
        CallTypes.vector_store_file_list,
    ],
    "/vector_stores/{vector_store_id}/files/{file_id}": [
        CallTypes.avector_store_file_retrieve,
        CallTypes.vector_store_file_retrieve,
        CallTypes.avector_store_file_delete,
        CallTypes.vector_store_file_delete,
    ],
    "/v1/vector_stores/{vector_store_id}/files/{file_id}": [
        CallTypes.avector_store_file_retrieve,
        CallTypes.vector_store_file_retrieve,
        CallTypes.avector_store_file_delete,
        CallTypes.vector_store_file_delete,
    ],
    "/vector_stores/{vector_store_id}/files/{file_id}/content": [
        CallTypes.avector_store_file_content,
        CallTypes.vector_store_file_content,
    ],
    "/v1/vector_stores/{vector_store_id}/files/{file_id}/content": [
        CallTypes.avector_store_file_content,
        CallTypes.vector_store_file_content,
    ],
    "/vector_stores/{vector_store_id}/files/{file_id}/update": [
        CallTypes.avector_store_file_update,
        CallTypes.vector_store_file_update,
    ],
    "/v1/vector_stores/{vector_store_id}/files/{file_id}/update": [
        CallTypes.avector_store_file_update,
        CallTypes.vector_store_file_update,
    ],
    # Containers
    "/containers": [
        CallTypes.acreate_container,
        CallTypes.create_container,
        CallTypes.alist_containers,
        CallTypes.list_containers,
    ],
    "/v1/containers": [
        CallTypes.acreate_container,
        CallTypes.create_container,
        CallTypes.alist_containers,
        CallTypes.list_containers,
    ],
    "/containers/{container_id}": [
        CallTypes.aretrieve_container,
        CallTypes.retrieve_container,
        CallTypes.adelete_container,
        CallTypes.delete_container,
    ],
    "/v1/containers/{container_id}": [
        CallTypes.aretrieve_container,
        CallTypes.retrieve_container,
        CallTypes.adelete_container,
        CallTypes.delete_container,
    ],
    # Responses API
    "/responses": (CallTypes.aresponses, CallTypes.responses),
    "/v1/responses": (CallTypes.aresponses, CallTypes.responses),
    "/openai/v1/responses": (CallTypes.aresponses, CallTypes.responses),
    "/responses/{response_id}": (CallTypes.aresponses, CallTypes.responses),
    "/v1/responses/{response_id}": (CallTypes.aresponses, CallTypes.responses),
    "/openai/v1/responses/{response_id}": (CallTypes.aresponses, CallTypes.responses),
    "/responses/{response_id}/input_items": (CallTypes.alist_input_items,),
    "/v1/responses/{response_id}/input_items": (CallTypes.alist_input_items,),
    "/openai/v1/responses/{response_id}/input_items": (CallTypes.alist_input_items,),
    # Realtime API
    "/realtime": [CallTypes.arealtime],
    "/v1/realtime": [CallTypes.arealtime],
    "/openai/v1/realtime": [CallTypes.arealtime],
    # Provider-specific routes
    "/anthropic/v1/messages": [CallTypes.anthropic_messages],
    # Google GenAI routes
    "/generate_content": [CallTypes.agenerate_content, CallTypes.generate_content],
    "/models/{model}:generateContent": [
        CallTypes.agenerate_content,
        CallTypes.generate_content,
    ],
    "/generate_content_stream": [
        CallTypes.agenerate_content_stream,
        CallTypes.generate_content_stream,
    ],
    "/models/{model}:streamGenerateContent": [
        CallTypes.agenerate_content_stream,
        CallTypes.generate_content_stream,
    ],
    # MCP (Model Context Protocol)
    "/mcp/call_tool": [CallTypes.call_mcp_tool],
    # A2A (Agent-to-Agent)
    "/a2a/{agent_id}": [CallTypes.asend_message, CallTypes.send_message],
    "/a2a/{agent_id}/message/send": [CallTypes.asend_message, CallTypes.send_message],
    # Passthrough endpoints
    "/llm_passthrough": [
        CallTypes.llm_passthrough_route,
        CallTypes.allm_passthrough_route,
    ],
    "/v1/llm_passthrough": [
        CallTypes.llm_passthrough_route,
        CallTypes.allm_passthrough_route,
    ],
    "/v1/messages": [CallTypes.anthropic_messages],
    # OCR
    "/ocr": [CallTypes.aocr, CallTypes.ocr],
    "/v1/ocr": [CallTypes.aocr, CallTypes.ocr],
}


class PassthroughCallTypes(Enum):
    passthrough_image_generation = "passthrough-image-generation"


class TopLogprob(OpenAIObject):
    token: str
    """The token."""

    bytes: list[int] | None = None
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

    bytes: list[int] | None = None
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

    top_logprobs: list[TopLogprob]
    """List of the most likely tokens and their log probability, at this token
    position.

    In rare cases, there may be fewer than the number of requested `top_logprobs`
    returned.
    """

    # Some OpenAI-compatible providers return null for top_logprobs when
    # omitted; normalize to [] to preserve the typed List[TopLogprob] contract.
    @field_validator("top_logprobs", mode="before")
    @classmethod
    def ensure_top_logprobs_is_list(cls, v):
        """Normalize null top_logprobs to empty list.

        Some providers return null instead of [] when logprobs=true but
        top_logprobs is unset. The OpenAI spec requires an array.
        Fixes https://github.com/BerriAI/litellm/issues/21932
        """
        if v is None:
            return []
        return v

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)


class ChoiceLogprobs(OpenAIObject):
    content: list[ChatCompletionTokenLogprob] | None = None
    """A list of message content tokens with log probability information."""

    def __contains__(self, key) -> bool:
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
    name: str | None = None


class Function(OpenAIObject):
    arguments: str
    name: str | None  # can be None - openai e.g.: ChoiceDeltaToolCallFunction(arguments='{"', name=None), type=None)

    def __init__(
        self,
        arguments: dict | str | None = None,
        name: str | None = None,
        **params,
    ) -> None:
        if arguments is None:
            if params.get("parameters", None) is not None and isinstance(params["parameters"], dict):
                arguments = json.dumps(params["parameters"])
                params.pop("parameters")
            else:
                arguments = ""
        elif isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        else:
            arguments = arguments

        name = name

        # Build a dictionary with the structure your BaseModel expects
        data: Final = {"arguments": arguments, "name": name}

        super().__init__(**data)

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class ChatCompletionDeltaToolCall(OpenAIObject):
    id: str | None = None
    function: Function
    type: str | None = None
    index: int

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class _CustomToolCallAccess(OpenAIObject):
    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)


class ChatCompletionCustomToolCallPayload(_CustomToolCallAccess):
    name: str
    input: str


class ChatCompletionDeltaCustomToolCallPayload(_CustomToolCallAccess):
    name: str | None = None
    input: str | None = None


class ChatCompletionMessageCustomToolCall(_CustomToolCallAccess):
    id: str
    type: Literal["custom"] = "custom"
    custom: ChatCompletionCustomToolCallPayload


class ChatCompletionDeltaCustomToolCall(_CustomToolCallAccess):
    id: str | None = None
    type: str | None = None
    custom: ChatCompletionDeltaCustomToolCallPayload
    index: int


class ChatCompletionMessageToolCall(OpenAIObject):
    def __init__(
        self,
        function: dict | Function,
        id: str | None = None,
        type: str | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        if isinstance(function, dict):
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

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


def is_custom_tool_call_dict(tool_call: Mapping[str, Any]) -> bool:
    return tool_call.get("type") == "custom" or tool_call.get("custom") is not None


def chat_completion_tool_call_from_dict(
    tool_call: Mapping[str, Any],
) -> "ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall":
    if is_custom_tool_call_dict(tool_call):
        return ChatCompletionMessageCustomToolCall(
            **MappingProxyType({k: v for k, v in tool_call.items() if not (k in ("function", "type") and v is None)})
        )
    return ChatCompletionMessageToolCall(**tool_call)


from openai.types.chat.chat_completion_audio import ChatCompletionAudio


class ChatCompletionAudioResponse(ChatCompletionAudio):
    def __init__(
        self,
        data: str,
        expires_at: int,
        transcript: str,
        id: str | None = None,
        **params,
    ) -> None:
        if id is not None:
            id = id
        else:
            id = f"{uuid.uuid4()}"
        super().__init__(data=data, expires_at=expires_at, transcript=transcript, id=id, **params)

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


"""
Reference:
ChatCompletionMessage(content='This is a test', role='assistant', function_call=None, tool_calls=None))
"""


def add_provider_specific_fields(object: BaseModel, provider_specific_fields: dict[str, Any] | None) -> None:
    if not provider_specific_fields:  # set if provider_specific_fields is not empty
        return
    object.provider_specific_fields = provider_specific_fields  # rebind-ok: sets the field on the caller's model


class Message(SafeAttributeModel, OpenAIObject):
    content: str | None
    role: Literal["assistant", "user", "system", "tool", "function"]
    tool_calls: (
        list[ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall] | None
    )  # mutable-ok: public pydantic response field; only the union member is new
    function_call: FunctionCall | None
    audio: ChatCompletionAudioResponse | None = None
    images: list[ImageURLListItem] | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None = None
    reasoning_items: list[ChatCompletionReasoningItem] | None = None
    provider_specific_fields: dict[str, Any] | None = Field(default=None)
    annotations: list[ChatCompletionAnnotation] | None = None

    def __init__(
        self,
        content: str | None = None,
        role: Literal["assistant", "user", "system", "tool", "function"] = "assistant",
        function_call=None,
        tool_calls: list | None = None,
        audio: ChatCompletionAudioResponse | None = None,
        images: list[ImageURLListItem] | None = None,
        provider_specific_fields: dict[str, Any] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None = None,
        reasoning_items: list[ChatCompletionReasoningItem] | None = None,
        annotations: list[ChatCompletionAnnotation] | None = None,
        **params,
    ) -> None:
        init_values: Final[dict[str, Any]] = {
            "content": content,
            "role": role or "assistant",  # handle null input
            "function_call": (FunctionCall(**function_call) if function_call is not None else None),
            "tool_calls": (
                [
                    (chat_completion_tool_call_from_dict(tool_call) if isinstance(tool_call, dict) else tool_call)
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

        if reasoning_items is not None:
            init_values["reasoning_items"] = reasoning_items

        if annotations is not None:
            init_values["annotations"] = annotations

        if reasoning_content is not None:
            init_values["reasoning_content"] = reasoning_content

        super().__init__(
            **init_values,
            **params,
        )

        if audio is None:
            # delete audio from self
            # OpenAI compatible APIs like mistral API will raise an error if audio is passed in
            if hasattr(self, "audio"):
                del self.audio

        if images is None and hasattr(self, "images"):
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

        if reasoning_items is None:
            # ensure default response matches OpenAI spec
            if hasattr(self, "reasoning_items"):
                del self.reasoning_items

        add_provider_specific_fields(self, provider_specific_fields)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()


class Delta(SafeAttributeModel, OpenAIObject):
    if TYPE_CHECKING:
        # Stored in __pydantic_extra__ at runtime (extra='allow'), set directly in
        # __init__ rather than via self.<attr> = .... Declared here only so type
        # checkers still see them as attributes for consumers that read delta.content
        # etc.; the runtime branch is skipped so pydantic does not treat them as fields.
        content: str | None
        role: str | None
        function_call: FunctionCall | None
        tool_calls: (
            list[ChatCompletionDeltaToolCall | ChatCompletionDeltaCustomToolCall] | None
        )  # mutable-ok: public pydantic response field; only the union member is new
        audio: ChatCompletionAudioResponse | None
        images: list[ImageURLListItem] | None
        annotations: list[ChatCompletionAnnotation] | None

    reasoning_content: str | None = None
    thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None = None
    reasoning_items: list[ChatCompletionReasoningItem] | None = None
    provider_specific_fields: dict[str, Any] | None = Field(default=None)

    def __init__(
        self,
        content=None,
        role=None,
        function_call=None,
        tool_calls=None,
        audio: ChatCompletionAudioResponse | None = None,
        images: list[ImageURLListItem] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None = None,
        reasoning_items: list[ChatCompletionReasoningItem] | None = None,
        annotations: list[ChatCompletionAnnotation] | None = None,
        **params,
    ) -> None:
        # Map 'reasoning' to 'reasoning_content' for providers that return
        # delta.reasoning (e.g., Cerebras, Groq gpt-oss models).
        # Must be done before super().__init__ to prevent 'reasoning' from
        # leaking as an extra attribute on the parent model.
        if reasoning_content is None and "reasoning" in params:
            reasoning_content = params.pop("reasoning", None)

        super().__init__(**params)
        add_provider_specific_fields(self, params.get("provider_specific_fields", {}))

        if function_call is not None and isinstance(function_call, dict):
            function_call = FunctionCall(**function_call)

        if tool_calls is not None and isinstance(tool_calls, (list, tuple)):
            coerced_tool_calls: list[
                ChatCompletionDeltaToolCall | ChatCompletionDeltaCustomToolCall
            ] = []  # mutable-ok: public Delta.tool_calls contract is a list
            current_index = 0
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    if tool_call.get("index", None) is None:
                        tool_call["index"] = current_index
                        current_index += 1
                    if is_custom_tool_call_dict(tool_call):
                        coerced_tool_calls.append(
                            ChatCompletionDeltaCustomToolCall(
                                **MappingProxyType(
                                    {k: v for k, v in tool_call.items() if not (k == "function" and v is None)}
                                )
                            )
                        )
                    else:
                        if tool_call.get("type", None) is None:
                            tool_call["type"] = "function"
                        coerced_tool_calls.append(ChatCompletionDeltaToolCall(**tool_call))
                elif isinstance(tool_call, (ChatCompletionDeltaToolCall, ChatCompletionDeltaCustomToolCall)):
                    coerced_tool_calls.append(tool_call)
            tool_calls = coerced_tool_calls

        # Build the per-chunk state directly instead of round-tripping every
        # field through pydantic's __setattr__/__delattr__ (the dominant
        # streaming cost). These keys are not declared model fields, so they
        # live in __pydantic_extra__; the slow path set each of content, role,
        # function_call, tool_calls, audio, images and annotations (marking them
        # in __pydantic_fields_set__) and then deleted the ones OpenAI omits.
        extra = self.__pydantic_extra__
        if extra is None:  # pragma: no cover - extra='allow' guarantees a dict
            extra = self.__pydantic_extra__ = {}
        fields_set: Final = self.__pydantic_fields_set__
        fields_set.update(
            (
                "content",
                "role",
                "function_call",
                "tool_calls",
                "audio",
                "images",
                "annotations",
            )
        )
        extra["content"] = content
        extra["role"] = role
        extra["function_call"] = function_call
        extra["tool_calls"] = tool_calls
        extra["audio"] = audio
        if images is not None and len(images) > 0:
            extra["images"] = images
        if annotations is not None:
            extra["annotations"] = annotations

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

        if reasoning_items is not None:
            self.reasoning_items = reasoning_items
        else:
            # ensure default response matches OpenAI spec
            if hasattr(self, "reasoning_items"):
                del self.reasoning_items

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Choices(SafeAttributeModel, OpenAIObject):
    finish_reason: OpenAIChatCompletionFinishReason
    index: int
    message: Message
    logprobs: ChoiceLogprobs | Any | None = None

    provider_specific_fields: dict[str, Any] | None = Field(default=None)

    def __init__(
        self,
        finish_reason=None,
        index=0,
        message: Message | dict | None = None,
        logprobs: ChoiceLogprobs | dict | Any | None = None,
        enhancements=None,
        provider_specific_fields: dict[str, Any] | None = None,
        **params,
    ) -> None:
        if finish_reason is not None:
            mapped: Final = map_finish_reason(finish_reason)
            params["finish_reason"] = mapped
            if finish_reason != mapped:
                provider_specific_fields = dict(provider_specific_fields) if provider_specific_fields else {}
                provider_specific_fields["native_finish_reason"] = finish_reason
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
            elif isinstance(message, BaseModel):
                # Normalize provider/OpenAI SDK message models into LiteLLM's Message type.
                dump: Final = message.model_dump() if hasattr(message, "model_dump") else message.dict()
                params["message"] = Message(**dump)
        if logprobs is not None:
            if isinstance(logprobs, dict):
                params["logprobs"] = ChoiceLogprobs(**logprobs)
            else:
                params["logprobs"] = logprobs
        else:
            params["logprobs"] = None
        super().__init__(**params)

        if enhancements is not None:
            self.enhancements = enhancements

        self.provider_specific_fields = provider_specific_fields

        if self.logprobs is None:
            del self.logprobs
        if self.provider_specific_fields is None:
            del self.provider_specific_fields

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class CompletionTokensDetailsWrapper(CompletionTokensDetails):  # wrapper for older openai versions
    text_tokens: int | None = None
    """Text tokens generated by the model."""

    image_tokens: int | None = None
    """Image tokens generated by the model."""

    video_tokens: int | None = None
    """Video tokens generated by the model."""


class CacheCreationTokenDetails(BaseModel):
    ephemeral_5m_input_tokens: int | None = None
    ephemeral_1h_input_tokens: int | None = None


class PromptTokensDetailsWrapper(
    SafeAttributeModel, PromptTokensDetails
):  # extends with image generation fields (text_tokens, image_tokens)
    text_tokens: int | None = None
    """Text tokens sent to the model."""

    image_tokens: int | None = None
    """Image tokens sent to the model."""

    video_tokens: int | None = None
    """Video tokens sent to the model."""

    web_search_requests: int | None = None
    """Number of web search requests made by the tool call. Used for Anthropic to calculate web search cost."""

    google_maps_grounding_requests: int | None = None
    """Number of Grounding with Google Maps requests made by the tool call. Used for Gemini to calculate Maps cost."""

    tool_use_tokens: int | None = None
    """Prompt tokens consumed by server-side tool use (e.g. Gemini grounding via googleSearch)."""

    character_count: int | None = None
    """Character count sent to the model. Used for Vertex AI multimodal embeddings."""

    image_count: int | None = None
    """Number of images sent to the model. Used for Vertex AI multimodal embeddings."""

    video_length_seconds: float | None = None
    """Length of videos sent to the model. Used for Vertex AI multimodal embeddings."""

    audio_length_seconds: float | None = None
    """Length of audio sent to the model. Used for multimodal embeddings priced per audio-second."""

    cache_write_tokens: int | None = None
    """Number of cache write (creation) tokens sent to the model. OpenAI naming (prompt_tokens_details.cache_write_tokens); this is the canonical field."""

    cache_creation_tokens: int | None = None
    """Number of cache creation tokens sent to the model. Anthropic/Bedrock naming; kept in sync with cache_write_tokens (assigning either mirrors to the other)."""

    cache_creation_token_details: CacheCreationTokenDetails | None = None
    """Details of cache creation tokens sent to the model. Used for tracking 5m/1h cache creation tokens for Anthropic prompt caching."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name == "cache_write_tokens":
            super().__setattr__("cache_creation_tokens", value)
        elif name == "cache_creation_tokens":
            super().__setattr__("cache_write_tokens", value)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        extra_fields: Final = self.model_extra
        nested_cache_creation_input_tokens: Final = (
            extra_fields.get("cache_creation_input_tokens") if extra_fields is not None else None
        )
        self.cache_write_tokens = (
            self.cache_write_tokens
            if self.cache_write_tokens is not None
            else (
                self.cache_creation_tokens
                if self.cache_creation_tokens is not None
                else (
                    nested_cache_creation_input_tokens if isinstance(nested_cache_creation_input_tokens, int) else None
                )
            )
        )
        if self.character_count is None:
            del self.character_count
        if self.image_count is None:
            del self.image_count
        if self.video_length_seconds is None:
            del self.video_length_seconds
        if self.audio_length_seconds is None:
            del self.audio_length_seconds
        if self.web_search_requests is None:
            del self.web_search_requests
        if self.google_maps_grounding_requests is None:
            del self.google_maps_grounding_requests
        if self.tool_use_tokens is None:
            del self.tool_use_tokens
        if self.cache_write_tokens is None:
            del self.cache_write_tokens
        if self.cache_creation_tokens is None:
            del self.cache_creation_tokens
        if self.cache_creation_token_details is None:
            del self.cache_creation_token_details


class ServerToolUse(BaseModel):
    web_search_requests: int | None = None
    tool_search_requests: int | None = None
    browser_open_requests: int | None = None

    def __getitem__(self, key: str) -> int | None:
        if key not in self.__class__.model_fields:
            raise KeyError(key)
        return getattr(self, key)


class Usage(SafeAttributeModel, CompletionUsage):
    _cache_creation_input_tokens: int = PrivateAttr(
        0
    )  # hidden param for prompt caching. Might change, once openai introduces their equivalent.
    _cache_read_input_tokens: int = PrivateAttr(
        0
    )  # hidden param for prompt caching. Might change, once openai introduces their equivalent.

    server_tool_use: ServerToolUse | None = None
    cost: float | None = None

    completion_tokens_details: CompletionTokensDetailsWrapper | None = None
    """Breakdown of tokens used in a completion."""

    prompt_tokens_details: PromptTokensDetailsWrapper | None = None
    """Breakdown of tokens used in the prompt."""

    def __init__(
        self,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        prompt_tokens_details: PromptTokensDetailsWrapper | PromptTokensDetails | dict | None = None,
        completion_tokens_details: CompletionTokensDetailsWrapper | dict | None = None,
        server_tool_use: ServerToolUse | dict | None = None,
        cost: float | None = None,
        **params,
    ) -> None:
        # handle reasoning_tokens
        _completion_tokens_details: CompletionTokensDetailsWrapper | None = None

        # First, handle existing completion_tokens_details
        if completion_tokens_details:
            if isinstance(completion_tokens_details, dict):
                _completion_tokens_details = CompletionTokensDetailsWrapper(**completion_tokens_details)
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
        _prompt_tokens_details: PromptTokensDetailsWrapper | None = None

        # guarantee prompt_token_details is always a PromptTokensDetailsWrapper
        if prompt_tokens_details:
            if isinstance(prompt_tokens_details, dict):
                _prompt_tokens_details = PromptTokensDetailsWrapper(**prompt_tokens_details)
            elif isinstance(prompt_tokens_details, PromptTokensDetails):
                _prompt_tokens_details = PromptTokensDetailsWrapper(**prompt_tokens_details.model_dump())
            elif isinstance(prompt_tokens_details, PromptTokensDetailsWrapper):
                _prompt_tokens_details = prompt_tokens_details

        ## DEEPSEEK MAPPING ##
        if "prompt_cache_hit_tokens" in params and isinstance(params["prompt_cache_hit_tokens"], int):
            if _prompt_tokens_details is None:
                _prompt_tokens_details = PromptTokensDetailsWrapper(cached_tokens=params["prompt_cache_hit_tokens"])
            else:
                _prompt_tokens_details.cached_tokens = params["prompt_cache_hit_tokens"]

        ## ANTHROPIC MAPPING ##
        if "cache_read_input_tokens" in params and isinstance(params["cache_read_input_tokens"], int):
            if _prompt_tokens_details is None:
                _prompt_tokens_details = PromptTokensDetailsWrapper(cached_tokens=params["cache_read_input_tokens"])
            else:
                _prompt_tokens_details.cached_tokens = params["cache_read_input_tokens"]

        if "cache_creation_input_tokens" in params and isinstance(params["cache_creation_input_tokens"], int):
            if _prompt_tokens_details is None:
                _prompt_tokens_details = PromptTokensDetailsWrapper(
                    cache_write_tokens=params["cache_creation_input_tokens"]
                )
            else:
                _prompt_tokens_details.cache_write_tokens = params["cache_creation_input_tokens"]

        super().__init__(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
            completion_tokens_details=_completion_tokens_details or None,
            prompt_tokens_details=_prompt_tokens_details or None,
        )

        if isinstance(server_tool_use, dict):
            server_tool_use = ServerToolUse(**server_tool_use)

        if server_tool_use is not None:
            self.server_tool_use = server_tool_use
        else:  # maintain openai compatibility in usage object if possible
            del self.server_tool_use

        if cost is not None:
            self.cost = cost
        else:
            del self.cost

        ## ANTHROPIC MAPPING ##
        if "cache_creation_input_tokens" in params and isinstance(params["cache_creation_input_tokens"], int):
            self._cache_creation_input_tokens = params["cache_creation_input_tokens"]

        if "cache_read_input_tokens" in params and isinstance(params["cache_read_input_tokens"], int):
            self._cache_read_input_tokens = params["cache_read_input_tokens"]

        ## DEEPSEEK MAPPING ##
        if "prompt_cache_hit_tokens" in params and isinstance(params["prompt_cache_hit_tokens"], int):
            self._cache_read_input_tokens = params["prompt_cache_hit_tokens"]

        for k, v in params.items():
            setattr(self, k, v)

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class StreamingChoices(OpenAIObject):
    def __init__(
        self,
        finish_reason=None,
        index=0,
        delta: Delta | None = None,
        logprobs=None,
        enhancements=None,
        **params,
    ) -> None:
        # Fix Perplexity return both delta and message cause OpenWebUI repect text
        # https://github.com/BerriAI/litellm/issues/8455
        params.pop("message", None)
        super().__init__(**params)
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
            self.logprobs = logprobs

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class StreamingChatCompletionChunk(OpenAIChatCompletionChunk):
    def __init__(self, **kwargs) -> None:
        new_choices: Final = []
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

    model: str | None = None
    """The model used for completion."""

    object: str
    """The object type, which is always "text_completion" """

    system_fingerprint: str | None = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    _hidden_params: dict = {}

    _response_headers: dict | None = None

    def set_provider_response_headers(self, headers: httpx.Headers) -> None:
        """Surface a provider's raw response headers to the caller as `llm_provider-*` headers."""
        self._hidden_params["additional_headers"] = process_response_headers(headers)

    def model_dump(self, **kwargs):
        """Default to exclude_unset to avoid Pydantic serializer warnings for OpenAIObject-derived types."""
        if "exclude_unset" not in kwargs and "exclude_none" not in kwargs:
            kwargs["exclude_unset"] = True
        return super().model_dump(**kwargs)


class ModelResponseStream(ModelResponseBase):
    choices: list[StreamingChoices]
    provider_specific_fields: dict[str, Any] | None = Field(default=None)

    def __init__(
        self,
        choices: list[StreamingChoices] | StreamingChoices | dict | BaseModel | None = None,
        id: str | None = None,
        created: int | None = None,
        provider_specific_fields: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if choices is not None and isinstance(choices, list):
            new_choices: Final = []
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

        usage_to_set = None
        if "usage" in kwargs and kwargs["usage"] is not None:
            if isinstance(kwargs["usage"], dict):
                usage_to_set = Usage(**kwargs["usage"])
                kwargs["usage"] = usage_to_set
            elif isinstance(kwargs["usage"], BaseModel):
                dump: Final = (
                    kwargs["usage"].model_dump() if hasattr(kwargs["usage"], "model_dump") else kwargs["usage"].dict()
                )
                usage_to_set = Usage(**dump)
                kwargs["usage"] = usage_to_set

        kwargs["id"] = id
        kwargs["created"] = created
        kwargs["object"] = "chat.completion.chunk"
        kwargs["provider_specific_fields"] = provider_specific_fields

        super().__init__(**kwargs)

        if usage_to_set is not None:
            self.usage = usage_to_set

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()


class ModelResponse(ModelResponseBase):
    choices: list[Choices]
    """The list of completion choices the model generated for the input prompt."""

    def __init__(
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
        object = "chat.completion"
        if choices is not None and isinstance(choices, list):
            new_choices: Final = []
            for choice in choices:
                if isinstance(choice, Choices):
                    _new_choice = choice
                elif isinstance(choice, dict):
                    _new_choice = Choices(**choice)
                elif isinstance(choice, BaseModel):
                    dump = choice.model_dump() if hasattr(choice, "model_dump") else choice.dict()
                    _new_choice = Choices(**dump)
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
            elif isinstance(usage, BaseModel):
                dump = usage.model_dump() if hasattr(usage, "model_dump") else usage.dict()
                usage = Usage(**dump)
            else:
                usage = usage
        elif stream is None or stream is False:
            usage = None  # avoid constructing throwaway Usage; set by convert_to_model_response_object
        if hidden_params:
            self._hidden_params = hidden_params

        if _response_headers:
            self._response_headers = _response_headers

        init_values: Final = {
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

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()


class Embedding(OpenAIObject):
    embedding: list | str = []
    index: int
    object: Literal["embedding"]

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class EmbeddingResponse(OpenAIObject):
    model: str | None = None
    """The model used for embedding."""

    data: list
    """The actual embedding value"""

    object: Literal["list"]
    """The object type, which is always "list" """

    usage: Usage | None = None
    """Usage statistics for the embedding request."""

    _hidden_params: dict = {}
    _response_headers: dict | None = None
    _response_ms: float | None = None

    def __init__(
        self,
        model: str | None = None,
        usage: Usage | None = None,
        response_ms=None,
        data: list | list[Embedding] | None = None,
        hidden_params=None,
        _response_headers=None,
        **params,
    ) -> None:
        object: Final = "list"
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
        super().__init__(model=model, object=object, data=data, usage=usage)

        if hidden_params:
            self._hidden_params = hidden_params

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()


class Logprobs(OpenAIObject):
    text_offset: list[int] | None
    token_logprobs: list[float | None] | None
    tokens: list[str] | None
    top_logprobs: list[dict[str, float] | None] | None


class TextChoices(OpenAIObject):
    def __init__(self, finish_reason=None, index=0, text=None, logprobs=None, **params) -> None:
        super().__init__(**params)
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

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()
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
    model: str | None
    choices: list[TextChoices]
    usage: Usage | None
    _response_ms: int | None = None
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
    ) -> None:
        if stream:
            object = "text_completion.chunk"
            choices = [TextChoices()]
        else:
            object = "text_completion"
            if choices is not None and isinstance(choices, list):
                new_choices: Final = []
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

        super().__init__(
            id=id,
            object=object,
            created=created,
            model=model,
            choices=choices,
            usage=usage,
            **params,
        )

        if response_ms:
            self._response_ms = response_ms
        else:
            self._response_ms = None
        self._hidden_params = HiddenParams()

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
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
    provider_specific_fields: Provider-specific fields not part of OpenAI spec.

    https://platform.openai.com/docs/api-reference/images/object
    """

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    provider_specific_fields: dict[str, Any] | None = None

    def __init__(
        self,
        b64_json=None,
        url=None,
        revised_prompt=None,
        provider_specific_fields=None,
        **kwargs,
    ) -> None:
        super().__init__(b64_json=b64_json, url=url, revised_prompt=revised_prompt)
        if provider_specific_fields:
            self.provider_specific_fields = provider_specific_fields

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()
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

    usage: ImageUsage | None = None
    """
    Users might use litellm with older python versions, we don't want this to break for them.
    Happens when their OpenAIImageResponse has the old OpenAI usage class.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def __init__(
        self,
        created: int | None = None,
        data: list[ImageObject] | None = None,
        response_ms=None,
        usage: ImageUsage | None = None,
        hidden_params: dict | None = None,
        **kwargs,
    ) -> None:
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

        _data: Final[list[OpenAIImage]] = []
        for d in data:
            if isinstance(d, dict):
                _data.append(ImageObject(**d))
            elif isinstance(d, BaseModel):
                _data.append(ImageObject(**d.model_dump()))

        _usage: Final = usage or ImageUsage(
            input_tokens=0,
            input_tokens_details=ImageUsageInputTokensDetails(
                image_tokens=0,
                text_tokens=0,
            ),
            output_tokens=0,
            total_tokens=0,
        )
        super().__init__(created=created, data=_data, usage=_usage)

        self.quality = kwargs.get("quality", None)
        self.output_format = kwargs.get("output_format", None)
        self.size = kwargs.get("size", None)
        self._hidden_params = hidden_params or {}

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()


class TranscriptionUsageDurationObject(BaseModel):
    type: Literal["duration"]
    seconds: float


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
    text: str | None = None
    usage: TranscriptionUsageDurationObject | TranscriptionUsageTokensObject | None = None

    _hidden_params: dict = {}
    _response_headers: dict | None = None

    def __init__(self, text=None) -> None:
        super().__init__(text=text)

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()
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
    force_timeout: float | None
    custom_llm_provider: str | None
    api_base: str | None
    litellm_call_id: str | None
    model_alias_map: dict | None
    metadata: dict | None
    litellm_metadata: dict | None
    model_info: dict | None
    proxy_server_request: dict | None
    acompletion: bool | None
    preset_cache_key: str | None
    no_log: bool | None
    input_cost_per_second: float | None
    input_cost_per_token: float | None
    output_cost_per_token: float | None
    output_cost_per_second: float | None
    cooldown_time: float | None


class AdapterCompletionStreamWrapper:
    def __init__(self, completion_stream) -> None:
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
            verbose_logger.debug("AdapterCompletionStreamWrapper - %s", e)

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
    user_api_key_hash: str | None  # hash of the litellm virtual key used
    user_api_key_alias: str | None
    user_api_key_spend: float | None
    user_api_key_max_budget: float | None
    user_api_key_budget_reset_at: str | None
    user_api_key_user_spend: float | None
    user_api_key_user_max_budget: float | None
    user_api_key_team_spend: float | None
    user_api_key_team_max_budget: float | None
    user_api_key_org_id: str | None
    user_api_key_org_alias: str | None
    user_api_key_team_id: str | None
    user_api_key_project_id: str | None
    user_api_key_project_alias: str | None
    user_api_key_user_id: str | None
    user_api_key_user_email: str | None
    user_api_key_team_alias: str | None
    user_api_key_end_user_id: str | None
    user_api_key_request_route: str | None
    user_api_key_auth_metadata: dict[str, str] | None


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

    mcp_server_name: str | None
    """
    Name of the MCP server that the tool call was made to
    """

    mcp_server_logo_url: str | None
    """
    Optional logo URL of the MCP server that the tool call was made to

    (this is to render the logo on the logs page on litellm ui)
    """

    namespaced_tool_name: str | None
    """
    Namespaced tool name of the MCP tool that the tool call was made to

    Includes the server name prefix if it exists - eg. `deepwiki-mcp/get_page_content`
    """

    mcp_server_cost_info: MCPServerCostInfo | None
    """
    Cost per query for the MCP server tool call
    """

    mcp_session_id: str | None
    """
    The MCP `mcp-session-id` of the stateful session this tool call ran in, when
    the client is driving a stateful session. Absent for stateless calls.
    """

    mcp_auth_mode: str | None
    """
    The server's auth_type for this call (e.g. `true_passthrough`, `oauth_delegate`,
    `oauth2`). For the client-forwarded token modes this records that the caller's own
    upstream token was relayed, so an audit can attribute a relayed request to its mode
    without logging any credential.
    """

    mcp_server_resource: str | None
    """
    The origin (scheme + host + port) of the upstream MCP server the tool call was forwarded
    to. Redacted for logging: userinfo, the path, the query string, and the fragment are all
    stripped, because hosted MCP servers routinely embed the credential in the URL path and
    this value is readable by callers via request logs.
    Records which upstream received a relayed request; never a credential.
    """


class StandardLoggingVectorStoreRequest(TypedDict, total=False):
    """
    Logging information for a vector store request/payload
    """

    vector_store_id: str | None
    """
    ID of the vector store
    """

    custom_llm_provider: str | None
    """
    Custom LLM provider the vector store is associated with eg. bedrock, openai, anthropic, etc.
    """

    query: str | None
    """
    Query to the vector store
    """

    vector_store_search_response: VectorStoreSearchResponse | None
    """
    OpenAI format vector store search response
    """

    start_time: float | None
    """
    Start time of the vector store request
    """

    end_time: float | None
    """
    End time of the vector store request
    """


class StandardBuiltInToolsParams(TypedDict, total=False):
    """
    Standard built-in OpenAItools parameters

    This is used to calculate the cost of built-in tools, insert any standard built-in tools parameters here

    OpenAI charges users based on the `web_search_options` parameter
    """

    web_search_options: WebSearchOptions | None
    file_search: FileSearchTool | None


class StandardLoggingPromptManagementMetadata(TypedDict):
    prompt_id: str
    prompt_variables: dict | None
    prompt_integration: str


class StandardLoggingRoutingDecisionTierBoundaries(TypedDict):
    """Snapshot of the complexity scorer's tier boundaries at decision time, so a
    historical spend log row stays explainable after the router config changes."""

    simple_medium: float
    medium_complex: float
    complex_reasoning: float


RoutingDecisionCause = Literal[
    "heuristic_scorer",
    # The scorer found 2+ reasoning markers and forced REASONING regardless of score.
    # A distinct cause rather than a marker inside `signals`, because it is the fact
    # that tells a reader the score did NOT choose the tier; encoding it as free text
    # meant anything that filtered `signals` silently changed what the row claimed.
    "reasoning_override",
    "llm_classifier",
    # classifier_type 'heuristic_first': the local scorer produced at least one signal and landed at
    # or below heuristic_first_max_tier, so it decided the tier and the LLM classifier was never
    # called. Distinct from "heuristic_scorer", which is a router whose only classifier IS the
    # scorer, and from "classifier_fallback", which is the scorer running because a call failed:
    # only this cause means an LLM classifier was configured, reachable, and deliberately skipped.
    "heuristic_first_short_circuit",
    # The operator's classifier plugin (classifier_type 'custom') decided the tier.
    "classifier_plugin",
    # The LLM classifier or classifier plugin failed on a router with an operator-defined
    # tier set, so the request routed to the configured fallback_tier without being classified.
    "classifier_fallback",
    # The LLM classifier or classifier plugin failed and classifier_fallback is
    # 'default_model', so the request went to default_model without being classified.
    # Distinct from "default_fallback",
    # which is a tier having no model configured rather than classification not happening.
    "default_model_fallback",
    "literal_keyword_match",
    "semantic_keyword_match",
    # A plan-mode sentinel (Claude Code / Copilot plan mode) was detected on the request and
    # plan_mode_min_tier decided the tier: either it raised what the pipeline chose (classifier,
    # keyword rule, or session pin), or the floor was already the top configured tier and the
    # classifier was skipped. The matched sentinel rides in matched_keyword.
    "plan_mode",
    "session_affinity_pin",
    "session_affinity_escalation",
    "default_fallback",
    "keyword",
    "quality_tier",
    "bandit",
]


InternalCallOrigin = Literal[
    "autorouter_classifier",
    "shadow_eval_router",
    "shadow_eval_judge",
    "background_response_cost_poll",
]
"""Which internal litellm feature originated a billed sub-call, so a spend log row
records that it is not traffic the caller sent."""

AUTOROUTER_CLASSIFIER_CALL_ORIGIN: Final[InternalCallOrigin] = "autorouter_classifier"
SHADOW_EVAL_ROUTER_CALL_ORIGIN: Final[InternalCallOrigin] = "shadow_eval_router"
SHADOW_EVAL_JUDGE_CALL_ORIGIN: Final[InternalCallOrigin] = "shadow_eval_judge"
BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN: Final[InternalCallOrigin] = "background_response_cost_poll"


class StandardLoggingRoutingDecision(TypedDict, total=False):
    """Per-request provenance for a pre-routing strategy (auto-router) decision."""

    router_model_name: str
    router_type: Literal["complexity", "adaptive", "quality"]
    routed_model: str
    cause: RoutingDecisionCause
    tier: str
    tier_label: str
    request_type: str
    score: float
    signals: Sequence[str]
    matched_keyword: str
    escalation_keyword: str
    classifier_model: str
    classifier_cost: float
    escalated: bool
    tier_boundaries: StandardLoggingRoutingDecisionTierBoundaries
    reasoning_override_min_score: float  # writable-ok: Pydantic warns on ReadOnly TypedDict fields
    conversation_continuing: bool
    savings_baseline_model: str
    savings_baseline_deployment_id: str
    tier_litellm_params: Mapping[str, object]  # writable-ok: Pydantic warns on ReadOnly TypedDict fields


# Fields whose values quote the caller's prompt. Dropped when an operator turns message
# logging off. Every other field aggregates the prompt without reproducing it and is kept,
# so a redacted row stays explainable. `test_every_routing_decision_field_is_classified`
# fails if a field is added to the record without being placed in one set or the other.
PROMPT_QUOTING_ROUTING_DECISION_FIELDS: frozenset[str] = frozenset({"signals", "matched_keyword", "escalation_keyword"})
DERIVED_ROUTING_DECISION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "router_model_name",
        "router_type",
        "routed_model",
        "cause",
        "tier",
        "tier_label",
        "request_type",
        "score",
        "classifier_model",
        "classifier_cost",
        "escalated",
        "tier_boundaries",
        "reasoning_override_min_score",
        "conversation_continuing",
        "savings_baseline_model",
        "savings_baseline_deployment_id",
        "tier_litellm_params",
    }
)


class StandardLoggingMetadata(StandardLoggingUserAPIKeyMetadata):
    """
    Specific metadata k,v pairs logged to integration for easier cost tracking and prompt management
    """

    spend_logs_metadata: dict | None  # special param to log k,v pairs to spendlogs for a call
    requester_ip_address: str | None
    user_agent: str | None
    requester_metadata: dict | None
    requester_custom_headers: dict[str, str] | None  # Log any custom (`x-`) headers sent by the client to the proxy.
    prompt_management_metadata: StandardLoggingPromptManagementMetadata | None
    mcp_tool_call_metadata: StandardLoggingMCPToolCall | None
    vector_store_request_metadata: list[StandardLoggingVectorStoreRequest] | None
    routing_decision: StandardLoggingRoutingDecision | None
    applied_guardrails: list[str] | None
    usage_object: dict | None
    cold_storage_object_key: str | None  # S3/GCS object key for cold storage retrieval
    team_alias: str | None
    team_id: str | None


class StandardLoggingAdditionalHeaders(TypedDict, total=False):
    x_ratelimit_limit_requests: int
    x_ratelimit_limit_tokens: int
    x_ratelimit_remaining_requests: int
    x_ratelimit_remaining_tokens: int
    x_ratelimit_reset_requests: str
    x_ratelimit_reset_tokens: str


class StandardLoggingHiddenParams(TypedDict):
    model_id: (
        str | None
    )  # id of the model in the router, separates multiple models with the same name but different credentials
    cache_key: str | None
    api_base: str | None
    response_cost: str | float | None
    litellm_overhead_time_ms: float | None
    additional_headers: StandardLoggingAdditionalHeaders | None
    batch_models: list[str] | None
    litellm_model_name: str | None  # the model name sent to the provider by litellm
    usage_object: dict | None


class StandardLoggingModelInformation(TypedDict):
    model_map_key: str
    model_map_value: ModelInfo | None


class StandardLoggingModelCostFailureDebugInformation(TypedDict, total=False):
    """
    Debug information, if cost tracking fails.

    Avoid logging sensitive information like response or optional params
    """

    error_str: Required[str]
    traceback_str: Required[str]
    model: str
    cache_hit: bool | None
    custom_llm_provider: str | None
    base_model: str | None
    call_type: str
    custom_pricing: bool | None


class StandardLoggingPayloadErrorInformation(TypedDict, total=False):
    error_code: str | None
    error_class: str | None
    llm_provider: str | None
    traceback: str | None
    error_message: str | None
    # error_rate_limit_category:
    #   For 429 / rate-limit errors, the source of the rate limit. One of the
    #   string values defined by `litellm.exceptions.RateLimitErrorCategory`
    #   (vendor_rate_limit, vendor_batch_rate_limit, litellm_rate_limit,
    #   litellm_batch_rate_limit). None for non-rate-limit exceptions.
    #   Surfaced here so custom callbacks / metrics consumers can switch on
    #   the rate-limit source without reaching for the raw exception.
    error_rate_limit_category: str | None
    # error_rate_limit_type:
    #   For 429 / rate-limit errors, the dimension that was exceeded. One of
    #   the string values defined by `litellm.exceptions.RateLimitType`
    #   (requests, tokens, concurrent_requests, budget, max_iterations).
    #   None for non-rate-limit exceptions and for rate-limit exceptions that
    #   did not classify the failure (e.g. legacy vendor 429 with no header
    #   hints). Lets dashboards split rate-limit failures by cause without
    #   parsing free-text error messages.
    error_rate_limit_type: str | None
    error_budget_entity_type: str | None
    error_budget_entity_id: str | None
    error_budget_limit: float | None
    error_budget_spend: float | None


class GuardrailMode(TypedDict, total=False):
    tags: dict[str, str | list[str]] | None
    default: str | list[str] | None


GuardrailStatus = Literal["success", "guardrail_intervened", "guardrail_failed_to_respond", "not_run"]


class StandardLoggingGuardrailInformation(TypedDict, total=False):
    guardrail_name: str | None
    guardrail_provider: str | None
    guardrail_mode: GuardrailEventHooks | list[GuardrailEventHooks] | GuardrailMode | None
    guardrail_request: str | dict | None
    guardrail_response: dict | str | list[dict] | None
    guardrail_status: GuardrailStatus
    start_time: float | None
    end_time: float | None
    duration: float | None
    """
    Duration of the guardrail in seconds
    """

    masked_entity_count: dict[str, int] | None
    """
    Count of masked entities
    {
        "CREDIT_CARD": 2,
        "PHONE": 1
    }
    """

    guardrail_id: str | None
    """Unique identifier for the guardrail configuration, e.g. 'gd-eu-pii-001'"""

    policy_template: str | None
    """Name of the policy template this guardrail belongs to, e.g. 'EU AI Act Article 5'"""

    detection_method: str | None
    """How detection was performed: 'regex', 'keyword', 'llm-judge', 'presidio', etc."""

    confidence_score: float | None
    """For LLM-judge guardrails: confidence score 0.0-1.0"""

    classification: str | dict | None
    """For LLM-judge guardrails: structured classification output"""

    match_details: str | list[dict] | None
    """Detailed match information for each detected pattern"""

    patterns_checked: int | None
    """Total number of patterns evaluated by this guardrail"""

    alert_recipients: list[str] | None
    """Email addresses that were notified"""

    risk_score: float | None
    """Risk score 0-10 indicating how risky the request was (higher = riskier). Computed by the guardrail provider."""

    violation_categories: list[str] | None
    """Names of the policy items that intervened on this request (e.g. Bedrock
    topic-policy topic names, content-policy filter types, PII entity types).
    Populated by the provider hook before redaction so downstream loggers
    (OTEL, Langfuse, ...) can filter by violation category without parsing
    the raw guardrail_response blob. Empty/absent when the guardrail allowed
    the request through."""

    guardrail_action: str | None
    """Provider's raw top-level action string (e.g. Bedrock's ``GUARDRAIL_INTERVENED``
    or ``NONE``). Populated by the provider hook so the OTEL integration can
    surface it as a queryable span attribute without parsing the raw
    guardrail_response blob."""

    guardrail_usage: ReadOnly[Mapping[str, int] | None]
    """Provider-reported billable usage counters for this invocation, keyed by the
    provider's counter name (e.g. Bedrock's ``contentPolicyUnits``). Kept as a
    sibling of guardrail_response so spend-log prompt redaction never drops it."""

    guardrail_cost: ReadOnly[float | None]
    """USD cost of this guardrail invocation, priced from ``guardrail_usage`` by the
    provider hook. Summed into the request's ``response_cost`` so it counts against
    spend and budgets like token cost, unless ``guardrail_cost_in_spend`` is False."""

    guardrail_cost_in_spend: ReadOnly[bool | None]
    """Whether ``guardrail_cost`` participates in the request's ``response_cost`` and
    the spend/budget aggregates built from it. Absent, None, or True keeps the default
    (cost counts against spend, the Bedrock behavior); False reports the cost on
    logs, OTEL spans, and the UI while every spend and budget total ignores it."""


class EvalVerdict(TypedDict, total=False):
    criterion_name: str
    score: float  # 0-100
    reasoning: str
    passed: bool
    weight: int  # criterion weight (0-100) as configured in the guardrail


class StandardLoggingEvalInformation(TypedDict, total=False):
    eval_id: str | None
    eval_name: str
    overall_score: float
    passed: bool
    judge_model: str
    iteration: int
    eval_error: str | None
    start_time: str
    end_time: str
    duration: float
    verdicts: list[Any]
    threshold: float | None


class GuardrailTracingDetail(TypedDict, total=False):
    """
    Typed fields for guardrail tracing metadata.

    Passed to add_standard_logging_guardrail_information_to_request_data()
    to enrich the StandardLoggingGuardrailInformation with provider-specific details.
    """

    guardrail_id: str | None
    policy_template: str | None
    detection_method: str | None
    confidence_score: float | None
    classification: dict | None
    match_details: list[dict] | None
    patterns_checked: int | None
    alert_recipients: list[str] | None
    risk_score: float | None
    violation_categories: list[str] | None
    guardrail_action: str | None
    guardrail_usage: ReadOnly[Mapping[str, int] | None]
    guardrail_cost: ReadOnly[float | None]
    guardrail_cost_in_spend: ReadOnly[bool | None]


StandardLoggingPayloadStatus = Literal["success", "failure"]


class CachingDetails(TypedDict):
    """
    Track all caching related metrics, fields for a given request
    """

    cache_hit: bool | None
    """
    Whether the request hit the cache
    """
    cache_duration_ms: float | None
    """
    Duration for reading from cache
    """


class CostBreakdown(TypedDict, total=False):
    """
    Detailed cost breakdown for a request.

    ``service_tier``, ``data_residency``, and ``vertex_location`` record the pricing
    basis the cost was computed on, not what the caller asked for. A consumer that has
    to price a counterfactual against this request (what another model would have
    charged for it) needs the same basis to compare like with like, and re-deriving it
    from the request is not possible after the fact: the tier the biller used comes
    from ``optional_params``, which no log record carries.
    """

    service_tier: str | None
    data_residency: str | None
    vertex_location: ReadOnly[str | None]
    input_cost: float  # Cost of raw (non-cached) input tokens only
    cache_read_cost: float  # Cost of cache-read tokens (discounted rate)
    cache_creation_cost: float  # Cost of cache-write tokens (premium rate)
    output_cost: float  # Cost of output/completion tokens (includes reasoning if applicable)
    reasoning_cost: float  # Cost of reasoning tokens (subset of output_cost)
    total_cost: ReadOnly[float]  # Total cost (input + output + tool usage + guardrail)
    tool_usage_cost: float  # Cost of usage of built-in tools
    guardrail_cost: ReadOnly[float]  # Cost counted in spend; report-only (guardrail_cost_in_spend=False) is excluded
    additional_costs: dict[str, float]  # Free-form additional costs (e.g., {"azure_model_router_flat_cost": 0.00014})
    original_cost: float  # Cost before discount (optional)
    discount_percent: float  # Discount percentage applied (e.g., 0.05 = 5%) (optional)
    discount_amount: float  # Discount amount in USD (optional)
    margin_percent: float  # Margin percentage applied (e.g., 0.10 = 10%) (optional)
    margin_fixed_amount: float  # Fixed margin amount in USD (optional)
    margin_total_amount: float  # Total margin added in USD (optional)


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


class StandardAuditLogPayload(TypedDict):
    """Payload for audit log events dispatched to external callbacks."""

    id: str
    updated_at: str  # ISO-8601
    changed_by: str
    changed_by_api_key: str
    action: str  # "created" | "updated" | "deleted" | "blocked" | "rotated"
    table_name: str
    object_id: str
    before_value: str | None
    updated_values: str | None


class StandardLoggingPayload(TypedDict):
    id: str
    trace_id: str  # Trace multiple LLM calls belonging to same overall request (e.g. fallbacks/retries)
    session_id: str  # End-user/conversation session id (litellm_session_id), independent of trace_id
    litellm_call_id: str | None  # UUID returned in x-litellm-call-id response header
    call_type: str
    stream: bool | None
    response_cost: float
    cost_breakdown: CostBreakdown | None  # Detailed cost breakdown
    autorouter_savings: ReadOnly[float | None]  # None = not an auto-routed caller request; 0.0 is a real figure
    response_cost_failure_debug_info: StandardLoggingModelCostFailureDebugInformation | None
    status: StandardLoggingPayloadStatus
    status_fields: StandardLoggingPayloadStatusFields
    custom_llm_provider: str | None
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    startTime: float  # Note: making this camelCase was a mistake, everything should be snake case
    endTime: float
    completionStartTime: float
    response_time: float
    model_map_information: StandardLoggingModelInformation
    model: str
    model_id: str | None
    model_group: str | None
    api_base: str
    metadata: StandardLoggingMetadata
    cache_hit: bool | None
    cache_key: str | None
    saved_cache_cost: float
    request_tags: list
    end_user: str | None
    requester_ip_address: str | None
    user_agent: str | None
    messages: str | list | dict | None
    response: str | list | dict | None
    error_str: str | None
    error_information: StandardLoggingPayloadErrorInformation | None
    model_parameters: dict
    hidden_params: StandardLoggingHiddenParams
    guardrail_information: list[StandardLoggingGuardrailInformation] | None
    standard_built_in_tools_params: StandardBuiltInToolsParams | None


from collections.abc import AsyncIterator, Iterator


class CustomStreamingDecoder:
    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[GenericStreamingChunk | StreamingChatCompletionChunk | None]:
        raise NotImplementedError

    def iter_bytes(
        self, iterator: Iterator[bytes]
    ) -> Iterator[GenericStreamingChunk | StreamingChatCompletionChunk | None]:
        raise NotImplementedError


class StandardPassThroughResponseObject(TypedDict):
    response: str | dict


OPENAI_RESPONSE_HEADERS: Final = [
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
]


class StandardCallbackDynamicParams(TypedDict, total=False):
    # Langfuse dynamic params
    langfuse_public_key: str | None
    langfuse_secret: str | None
    langfuse_secret_key: str | None
    langfuse_host: str | None
    langfuse_environment: ReadOnly[str | None]

    # Langfuse prompt version
    langfuse_prompt_version: int | None

    # GCS dynamic params
    gcs_bucket_name: str | None
    gcs_path_service_account: str | None

    # Langsmith dynamic params
    langsmith_api_key: str | None
    langsmith_project: str | None
    langsmith_base_url: str | None
    langsmith_sampling_rate: float | None
    langsmith_tenant_id: str | None

    # Humanloop dynamic params
    humanloop_api_key: str | None

    # Arize dynamic params
    arize_api_key: str | None
    arize_space_key: str | None
    arize_space_id: str | None

    # PostHog dynamic params
    posthog_api_key: str | None
    posthog_api_url: str | None

    # Weave (W&B) dynamic params
    wandb_api_key: str | None
    weave_project_id: str | None

    # Datadog dynamic params
    dd_api_key: str | None
    dd_site: str | None
    dd_agent_host: str | None
    dd_agent_port: str | None

    # New Relic dynamic params (proxy-stamped team/key callback vars only;
    # request-supplied values are blocked)
    newrelic_api_key: str | None  # writable-ok: initialize_standard_callback_dynamic_params assigns into the dict
    newrelic_region: str | None  # writable-ok: initialize_standard_callback_dynamic_params assigns into the dict

    # Logging settings
    turn_off_message_logging: bool | None  # when true will not log messages
    litellm_disabled_callbacks: list[str] | None


class MirroredPricingParams(BaseModel):
    """Pricing overrides that ``Deployment.__init__`` mirrors from ``litellm_params``
    onto ``model_info``, so both blobs hold the same rate.

    Declared once and inherited by both sides of that mirror, so the two can't drift.
    """

    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    input_cost_per_character: float | None = None
    output_cost_per_character: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None
    tiered_pricing: list[dict[str, Any]] | None = None


class CustomPricingLiteLLMParams(MirroredPricingParams):
    ## CUSTOM PRICING ##
    input_cost_per_second: float | None = None
    output_cost_per_second: float | None = None
    output_cost_per_second_1080p: float | None = None
    output_cost_per_second_480p: float | None = None
    output_cost_per_second_4k: float | None = None
    input_cost_per_pixel: float | None = None
    output_cost_per_pixel: float | None = None

    # Include all ModelInfoBase fields as optional
    # This allows any model_info parameter to be set in litellm_params
    input_cost_per_token_flex: float | None = None
    input_cost_per_token_priority: float | None = None
    input_cost_per_token_ultrafast: float | None = None
    cache_creation_input_token_cost_above_1hr: float | None = None
    cache_creation_input_token_cost_above_200k_tokens: float | None = None
    cache_creation_input_token_cost_above_272k_tokens: float | None = None
    cache_creation_input_token_cost_above_272k_tokens_priority: float | None = None
    cache_creation_input_token_cost_above_272k_tokens_flex: float | None = None
    cache_creation_input_token_cost_flex: float | None = None
    cache_creation_input_token_cost_priority: float | None = None
    cache_creation_input_token_cost_ultrafast: float | None = None
    cache_creation_input_audio_token_cost: float | None = None
    cache_read_input_token_cost_flex: float | None = None
    cache_read_input_token_cost_priority: float | None = None
    cache_read_input_token_cost_ultrafast: float | None = None
    cache_read_input_token_cost_above_200k_tokens: float | None = None
    cache_read_input_token_cost_above_200k_tokens_priority: float | None = None
    cache_read_input_token_cost_above_272k_tokens_priority: float | None = None
    cache_read_input_token_cost_above_272k_tokens_flex: float | None = None
    cache_read_input_audio_token_cost: float | None = None
    input_cost_per_character_above_128k_tokens: float | None = None
    input_cost_per_audio_token: float | None = None
    input_cost_per_token_cache_hit: float | None = None
    input_cost_per_token_above_128k_tokens: float | None = None
    input_cost_per_token_above_200k_tokens: float | None = None
    input_cost_per_token_above_200k_tokens_priority: float | None = None
    input_cost_per_token_above_272k_tokens_priority: float | None = None
    input_cost_per_token_above_272k_tokens_flex: float | None = None
    input_cost_per_query: float | None = None
    input_cost_per_image: float | None = None
    input_cost_per_image_above_128k_tokens: float | None = None
    input_cost_per_audio_per_second: float | None = None
    input_cost_per_audio_per_second_above_128k_tokens: float | None = None
    input_cost_per_video_per_second: float | None = None
    input_cost_per_video_per_second_above_128k_tokens: float | None = None
    input_cost_per_video_per_second_above_15s_interval: float | None = None
    input_cost_per_video_per_second_above_8s_interval: float | None = None
    input_cost_per_token_batches: float | None = None
    output_cost_per_token_batches: float | None = None
    output_cost_per_token_flex: float | None = None
    output_cost_per_token_priority: float | None = None
    output_cost_per_token_ultrafast: float | None = None
    output_cost_per_audio_token: float | None = None
    output_cost_per_token_above_128k_tokens: float | None = None
    output_cost_per_token_above_200k_tokens: float | None = None
    output_cost_per_token_above_200k_tokens_priority: float | None = None
    output_cost_per_token_above_272k_tokens_priority: float | None = None
    output_cost_per_token_above_272k_tokens_flex: float | None = None
    output_cost_per_character_above_128k_tokens: float | None = None
    output_cost_per_image: float | None = None
    output_cost_per_image_token: float | None = None
    output_cost_per_video_token: float | None = None
    output_cost_per_reasoning_token: float | None = None
    output_cost_per_reasoning_token_flex: float | None = None
    output_cost_per_reasoning_token_priority: float | None = None
    output_cost_per_video_per_second: float | None = None
    output_cost_per_audio_per_second: float | None = None
    search_context_cost_per_query: dict[str, Any] | None = None
    google_maps_grounding_cost_per_query: float | None = None
    citation_cost_per_token: float | None = None
    cache_read_input_token_cost_above_272k_tokens: float | None = None
    cache_read_input_token_cost_above_512k_tokens: float | None = None
    input_cost_per_image_token: float | None = None
    input_cost_per_video_token: float | None = None
    input_cost_per_token_above_272k_tokens: float | None = None
    input_cost_per_token_above_512k_tokens: float | None = None
    output_cost_per_token_above_272k_tokens: float | None = None
    output_cost_per_token_above_512k_tokens: float | None = None
    output_vector_size: int | None = None
    ocr_cost_per_page: float | None = None
    ocr_cost_per_credit: float | None = None
    annotation_cost_per_page: float | None = None
    regional_processing_uplift_multiplier_eu: float | None = None
    regional_processing_uplift_multiplier_us: float | None = None
    regional_endpoint_uplift_multiplier: float | None = None

    @classmethod
    def strip_custom_pricing_fields(cls, model_info: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``model_info`` without per-deployment custom pricing fields.

        Used when registering a deployment's info under the shared
        ``{provider}/{model}`` key in ``litellm.model_cost``, so one deployment's
        pricing overrides don't pollute sibling deployments that share the same
        backend model. Full pricing stays under the deployment's unique model id.
        """
        return {k: v for k, v in model_info.items() if k not in cls.model_fields}


SHARED_BACKEND_MODEL_INFO_FIELDS: Final[frozenset[str]] = frozenset(
    ModelInfoBase.__required_keys__ | ModelInfoBase.__optional_keys__
) - frozenset(CustomPricingLiteLLMParams.model_fields)


def shared_backend_model_info(model_info: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields safe to register under a shared ``{provider}/{model}``
    key in ``litellm.model_cost``: cost-map schema fields (``ModelInfoBase``) minus
    per-deployment pricing overrides. Per-deployment metadata (``id``,
    ``access_via_team_ids``, arbitrary custom keys) never belongs on the shared key;
    it stays under the deployment's unique model id.
    """
    return {k: v for k, v in model_info.items() if k in SHARED_BACKEND_MODEL_INFO_FIELDS}


# Server-controlled fields that bound or drive an interceptor's agentic loop
# (depth, cycle fingerprints, ceiling, code-interpreter sandbox state). Listed
# in all_litellm_params so they are treated as LiteLLM-level and excluded from
# get_non_default_completion_params; otherwise the OpenAI param builder sweeps
# any unrecognized top-level key into extra_body and leaks them to the provider.
# This is what lets the loop carry state across rerun calls without a provider
# scrubber.
agentic_loop_internal_litellm_params: Final = [
    "_agentic_loop_depth",
    "_agentic_loop_fingerprints",
    "_agentic_loop_api_surface",
    "max_agentic_loops",
    "_code_interpreter_interception_active",
    "_code_interpreter_interception_sandbox_key",
    "_code_interpreter_interception_session_scoped",
    "_code_interpreter_interception_converted_stream",
    "_websearch_interception_emit_native_blocks",
    "_websearch_interception_converted_stream",
]

# Proxy-owned callback credentials, stamped from admin-configured team/key callback
# settings. Listed in all_litellm_params for the same reason as the agentic-loop
# fields above: an unrecognized top-level key is swept into extra_body and sent to
# the provider.
TRUSTED_CALLBACK_VARS_FIELD: Final = "litellm_trusted_callback_vars"

# Bedrock managed-batch deployment config, read from litellm_params by the batch and
# files transformations. Listed for the same reason as the fields above: these sit on
# a deployment that also serves chat, so leaking them into extra_body makes Bedrock
# reject every non-batch request to that deployment.
bedrock_batch_litellm_params: Final = (
    "aws_batch_role_arn",
    "s3_bucket_name",
    "s3_region_name",
    "s3_output_bucket_name",
    "bedrock_tags",
)

all_litellm_params = (
    agentic_loop_internal_litellm_params
    + [TRUSTED_CALLBACK_VARS_FIELD, *bedrock_batch_litellm_params]
    + [
        "metadata",
        "litellm_metadata",
        "keepalive_seconds",
        "allow_client_keepalive_override",
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
        "api_key",
        "api_version",
        "prompt_id",
        "prompt_variables",
        "litellm_system_prompt",
        "provider_specific_header",
        "prompt_version",
        "api_base",
        "force_timeout",
        "logger_fn",
        "verbose",
        "custom_llm_provider",
        "model_file_id_mapping",
        "litellm_logging_obj",
        "litellm_call_id",
        "_litellm_strip_stream_usage",
        "use_client",
        "id",
        "fallbacks",
        "routing_strategy",
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
        "client_side_timeout",
        "complete_response",
        "self",
        "client",
        "rpm",
        "tpm",
        "itpm",
        "otpm",
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
        "enable_prompt_caching",
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
        "attempted_targets",
        "max_budget",
        "budget_duration",
        "use_in_pass_through",
        "merge_reasoning_content_in_choices",
        "litellm_credential_name",
        "allowed_openai_params",
        "litellm_session_id",
        "use_litellm_proxy",
        "use_chat_completions_api",
        "rust",
        "prompt_label",
        "shared_session",
        "search_tool_name",
        "order",
        "enable_tag_filtering",
        "enable_json_schema_validation",
        "use_xai_oauth",
        "auto_router_config_path",
        "auto_router_config",
        "auto_router_default_model",
        "auto_router_embedding_model",
        "auto_router_max_input_chars",
        "complexity_router_config",
        "complexity_router_default_model",
        "adaptive_router_config",
        "adaptive_router_default_model",
        "quality_router_config",
        "quality_router_default_model",
    ]
    + list(StandardCallbackDynamicParams.__annotations__.keys())
    + list(CustomPricingLiteLLMParams.model_fields.keys())
)


class KeyGenerationConfig(TypedDict, total=False):
    required_params: list[str]  # specify params that must be present in the key generation request


class TeamUIKeyGenerationConfig(KeyGenerationConfig):
    allowed_team_member_roles: list[str]


class PersonalUIKeyGenerationConfig(KeyGenerationConfig):
    allowed_user_roles: list[str]


class StandardKeyGenerationConfig(TypedDict, total=False):
    team_key_generation: TeamUIKeyGenerationConfig
    personal_key_generation: PersonalUIKeyGenerationConfig


class BudgetConfig(BaseModel):
    max_budget: float | None = None
    budget_duration: str | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None

    def __init__(self, **data: Any) -> None:
        # Map time_period to budget_duration if present
        if "time_period" in data:
            data["budget_duration"] = data.pop("time_period")

        # Map budget_limit to max_budget if present
        if "budget_limit" in data:
            data["max_budget"] = data.pop("budget_limit")

        super().__init__(**data)


GenericBudgetConfigType = dict[str, BudgetConfig]


class LlmProviders(str, Enum):
    OPENAI = "openai"
    CHATGPT = "chatgpt"
    OPENAI_LIKE = "openai_like"  # embedding only
    JINA_AI = "jina_ai"
    XAI = "xai"
    ZAI = "zai"
    CUSTOM_OPENAI = "custom_openai"
    TEXT_COMPLETION_OPENAI = "text-completion-openai"
    COHERE = "cohere"
    COHERE_CHAT = "cohere_chat"
    CLARIFAI = "clarifai"
    ANTHROPIC = "anthropic"
    ANTHROPIC_TEXT = "anthropic_text"
    BYTEZ = "bytez"
    REPLICATE = "replicate"
    REDUCTO = "reducto"
    RUNWAYML = "runwayml"
    AWS_POLLY = "aws_polly"
    HUGGINGFACE = "huggingface"
    TOGETHER_AI = "together_ai"
    OPENROUTER = "openrouter"
    DATAROBOT = "datarobot"
    VERTEX_AI = "vertex_ai"
    VERTEX_AI_BETA = "vertex_ai_beta"
    GEMINI = "gemini"
    AI21 = "ai21"
    BASETEN = "baseten"
    BLACK_FOREST_LABS = "black_forest_labs"
    AZURE = "azure"
    AZURE_TEXT = "azure_text"
    AZURE_AI = "azure_ai"
    SAGEMAKER = "sagemaker"
    SAGEMAKER_CHAT = "sagemaker_chat"
    SAGEMAKER_NOVA = "sagemaker_nova"
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
    A2A = "a2a"
    GIGACHAT = "gigachat"
    NVIDIA_NIM = "nvidia_nim"
    NVIDIA_RIVA = "nvidia_riva"
    SONIOX = "soniox"
    CEREBRAS = "cerebras"
    AI21_CHAT = "ai21_chat"
    VOLCENGINE = "volcengine"
    CODESTRAL = "codestral"
    TEXT_COMPLETION_CODESTRAL = "text-completion-codestral"
    DASHSCOPE = "dashscope"
    MODELSCOPE = "modelscope"
    MOONSHOT = "moonshot"
    PUBLICAI = "publicai"
    V0 = "v0"
    MORPH = "morph"
    LAMBDA_AI = "lambda_ai"
    INCEPTION = "inception"
    TEXT_COMPLETION_INCEPTION = "text-completion-inception"
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
    RAGFLOW = "ragflow"
    COMPACTIFAI = "compactifai"
    DOCKER_MODEL_RUNNER = "docker_model_runner"
    CUSTOM = "custom"
    LITELLM_PROXY = "litellm_proxy"
    HOSTED_VLLM = "hosted_vllm"
    TENCENT = "tencent"
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
    SAP_GENERATIVE_AI_HUB = "sap"
    ASSEMBLYAI = "assemblyai"
    CHARITY_ENGINE = "charity_engine"
    GITHUB_COPILOT = "github_copilot"
    SNOWFLAKE = "snowflake"
    GRADIENT_AI = "gradient_ai"
    LLAMA = "meta_llama"
    NSCALE = "nscale"
    PG_VECTOR = "pg_vector"
    S3_VECTORS = "s3_vectors"
    VALKEY = "valkey"
    HELICONE = "helicone"
    HYPERBOLIC = "hyperbolic"
    RECRAFT = "recraft"
    FAL_AI = "fal_ai"
    STABILITY = "stability"
    HEROKU = "heroku"
    AIML = "aiml"
    COMETAPI = "cometapi"
    OCI = "oci"
    AUTO_ROUTER = "auto_router"
    VERCEL_AI_GATEWAY = "vercel_ai_gateway"
    DOTPROMPT = "dotprompt"
    MANUS = "manus"
    WANDB = "wandb"
    OVHCLOUD = "ovhcloud"
    SCALEWAY = "scaleway"
    LEMONADE = "lemonade"
    AMAZON_NOVA = "amazon_nova"
    A2A_AGENT = "a2a_agent"
    LANGGRAPH = "langgraph"
    LANGFLOW = "langflow"
    MINIMAX = "minimax"
    SYNTHETIC = "synthetic"
    APERTIS = "apertis"
    NANOGPT = "nano-gpt"
    POE = "poe"
    CHUTES = "chutes"
    NEOSANTARA = "neosantara"
    PARASAIL = "parasail"
    XIAOMI_MIMO = "xiaomi_mimo"
    TENSORMESH = "tensormesh"
    LIBERTAI = "libertai"
    PINSTRIPES = "pinstripes"
    COGNITION = "cognition"
    SCX_AI = "scx-ai"
    DARKBLOOM = "darkbloom"
    META = "meta"
    LITELLM_AGENT = "litellm_agent"
    CURSOR = "cursor"
    BEDROCK_MANTLE = "bedrock_mantle"
    GDC = "gdc"


# Create a set of all provider values for quick lookup
LlmProvidersSet: Final = {provider.value for provider in LlmProviders}

# File and Batch API providers that are OpenAI-compatible
OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS: set[str] = {
    LlmProviders.OPENAI.value,
    LlmProviders.HOSTED_VLLM.value,
    LlmProviders.LITELLM_PROXY.value,
}

ListBatchesSupportedProvider = Literal["openai", "azure", "hosted_vllm", "litellm_proxy", "vertex_ai"]

LIST_BATCHES_SUPPORTED_PROVIDERS: Final[frozenset[str]] = frozenset(get_args(ListBatchesSupportedProvider))


class SearchProviders(str, Enum):
    """
    Enum for search provider types.
    Separate from LlmProviders for semantic clarity.
    """

    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    PARALLEL_AI = "parallel_ai"
    EXA_AI = "exa_ai"
    BRAVE = "brave"
    GOOGLE_PSE = "google_pse"
    DATAFORSEO = "dataforseo"
    FIRECRAWL = "firecrawl"
    FASTCRW = "fastcrw"
    SEARXNG = "searxng"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    SEARCHAPI = "searchapi"
    SERPER = "serper"
    YOU_COM = "you_com"
    APISERPENT = "apiserpent"
    TINYFISH = "tinyfish"
    AGENTCORE = "agentcore"
    NIMBLE = "nimble"
    BING_GROUNDING = "bing_grounding"


# Create a set of all search provider values for quick lookup
SearchProvidersSet: Final = {provider.value for provider in SearchProviders}


class SandboxProviders(str, Enum):
    """
    Enum for code execution sandbox provider types.
    Separate from LlmProviders for semantic clarity.
    """

    E2B = "e2b"
    OPENSANDBOX = "opensandbox"


class LiteLLMLoggingBaseClass:
    """
    Base class for logging pre and post call

    Meant to simplify type checking for logging obj.
    """

    def pre_call(self, input, api_key, model=None, additional_args=None) -> None:
        pass

    def post_call(self, original_response, input=None, api_key=None, additional_args=None) -> None:
        pass


class TokenCountResponse(LiteLLMPydanticObjectBase):
    total_tokens: int
    request_model: str
    model_used: str
    tokenizer_type: str
    original_response: dict | None = None
    """
    Original Response from upstream API call - if an API call was made for token counting
    """
    error: bool = False
    error_message: str | None = None
    """
    HTTP status code from the token counting API (e.g., 200 for success, 429 for rate limit, 400 for bad request)
    """
    status_code: int | None = None


class CustomHuggingfaceTokenizer(TypedDict):
    identifier: str
    revision: str  # usually 'main'
    auth_token: str | None


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
    seed: int | None = None

    def __init__(self, **kwargs) -> None:
        if "error" in kwargs and kwargs["error"] is not None:
            # check if error is all None - if so, set error to None
            if all(value is None for value in kwargs["error"].values()):
                kwargs["error"] = None
        super().__init__(**kwargs)
        self._hidden_params = kwargs.get("_hidden_params", {})


class LiteLLMBatch(Batch):
    _hidden_params: dict = {}
    usage: Usage | None = None

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()


class LiteLLMRealtimeStreamLoggingObject(LiteLLMPydanticObjectBase):
    # Events are already well-formed provider dicts. Validating them against the
    # OpenAIRealtimeEvents union makes Pydantic try every member per event, which
    # floods thousands of ValidationErrors for events outside the union (e.g.
    # rate_limits.updated), blocks the event loop, and discards the session usage.
    results: SkipValidation[OpenAIRealtimeStreamList]
    usage: Usage
    _hidden_params: dict = {}

    @field_serializer("results")
    def _serialize_results(self, results: OpenAIRealtimeStreamList) -> list[dict[str, Any]]:
        return [dict(event) for event in results]

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()


class RawRequestTypedDict(TypedDict, total=False):
    raw_request_api_base: str | None
    raw_request_body: dict | None
    raw_request_headers: dict | None
    error: str | None


from litellm.models.credentials import (  # noqa: E402
    CreateCredentialItem as CreateCredentialItem,
)
from litellm.models.credentials import CredentialBase as CredentialBase  # noqa: E402
from litellm.models.credentials import CredentialItem as CredentialItem  # noqa: E402


class ExtractedFileData(TypedDict):
    """
    TypedDict for storing processed file data

    Attributes:
        filename: Name of the file if provided
        content: The file content in bytes
        content_type: MIME type of the file
        headers: Any additional headers for the file
    """

    filename: str | None
    content: bytes
    content_type: str | None
    headers: Mapping[str, str]


class SpecialEnums(Enum):
    LITELM_MANAGED_FILE_ID_PREFIX = "litellm_proxy"
    LITELLM_MANAGED_FILE_COMPLETE_STR = (
        "litellm_proxy:{};unified_id,{};target_model_names,{};llm_output_file_id,{};llm_output_file_model_id,{}"
    )

    LITELLM_MANAGED_RESPONSE_COMPLETE_STR = "litellm:custom_llm_provider:{};model_id:{};response_id:{}"

    LITELLM_MANAGED_BATCH_COMPLETE_STR = "litellm_proxy;model_id:{};llm_batch_id:{}"

    LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR = (
        "litellm_proxy:responses_api:response_id:{};user_id:{};team_id:{}"
    )

    LITELLM_MANAGED_GENERIC_RESPONSE_COMPLETE_STR = "litellm_proxy;model_id:{};generic_response_id:{}"  # generic implementation of 'managed batches' - used for finetuning and any future work.

    LITELLM_MANAGED_VIDEO_COMPLETE_STR = "litellm:custom_llm_provider:{};model_id:{};video_id:{}"

    LITELLM_PASSTHROUGH_MANAGED_ID_COMPLETE_STR = "litellm_proxy:passthrough;provider:{};unified_id,{};raw_id,{}"


class ServiceTier(Enum):
    """Enum for service tier types used in cost calculations."""

    AUTO = "auto"
    FLEX = "flex"
    PRIORITY = "priority"
    FAST = "fast"
    ULTRAFAST = "ultrafast"


class DataResidency(Enum):
    """
    OpenAI data-residency / regional-processing regions.

    Inferred from the OpenAI api_base host (eu.api.openai.com -> EU,
    us.api.openai.com -> US). Used to apply the regional-processing
    cost uplift (see ``regional_processing_uplift_multiplier_<region>``
    on ModelInfo).
    """

    US = "us"
    EU = "eu"


LLMResponseTypes = (
    ModelResponse
    | EmbeddingResponse
    | ImageResponse
    | OpenAIFileObject
    | LiteLLMBatch
    | LiteLLMFineTuningJob
    | AnthropicMessagesResponse
    | ResponsesAPIResponse
    | LiteLLMSendMessageResponse
)


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
    success: list[str]
    failure: list[str]
    success_and_failure: list[str]


CostResponseTypes = ModelResponse | TextCompletionResponse | EmbeddingResponse | ImageResponse | TranscriptionResponse


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

    saturation_check_cache_ttl: int = Field(
        default=60,
        description="TTL in seconds for local cache when reading saturation check values from Redis.",
    )

    model_config = ConfigDict(protected_namespaces=())


class GenericGuardrailAPIInputs(TypedDict, total=False):
    texts: list[str]  # extracted text from the LLM response - for basic text guardrails
    images: list[str]  # extracted images from the LLM response - for image guardrails
    tools: list[ChatCompletionToolParam]  # tools sent to the LLM
    tool_calls: list[ChatCompletionToolCallChunk] | list[ChatCompletionMessageToolCall]  # tool calls sent from the LLM
    structured_messages: list[
        AllMessageValues
    ]  # structured messages sent to the LLM - indicates if text is from system or user
    model: str | None  # the model being used for the LLM call
    stream_holdback_chars: list[
        int
    ]  # trailing chars to withhold from streaming emission per text (word-boundary safety)
