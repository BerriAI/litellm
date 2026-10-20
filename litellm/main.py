# LiteLLM main module: public completion, embedding, streaming, and moderation entrypoints.
#
# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan

import asyncio
import contextvars
import datetime
import inspect
import json
import random
import sys
import traceback
from collections.abc import AsyncIterator, Coroutine, Iterable, Mapping
from concurrent import futures
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, Union, cast, get_args

from litellm._logging import redact_string
from litellm._uuid import uuid

if TYPE_CHECKING:
    from aiohttp import ClientSession

import dotenv
import httpx
import openai
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import overload

import litellm

# client must be imported from litellm as it's a decorator used at function definition time
from litellm import client

# Other utils are imported directly to avoid circular imports
from litellm.utils import exception_type, filter_out_litellm_params, get_litellm_params

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.utils import TokenCountResponse

import os as os  # noqa: PLC0414  # litellm.* re-export

from typing_extensions import (
    assert_never as assert_never,  # noqa: PLC0414  # litellm.* re-export
)

from litellm.constants import AZURE_OPENAI_AUDIO_PROVIDERS, OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS
from litellm.constants import (
    DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT as DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.constants import (
    DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT as DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.constants import (
    NADIR_DEFAULT_API_BASE as NADIR_DEFAULT_API_BASE,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.exceptions import LiteLLMUnknownProvider
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.asyncify import asyncify, run_async_function
from litellm.litellm_core_utils.audio_utils.utils import (
    calculate_request_duration,
    get_audio_file_for_health_check,
)
from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    maybe_run_chat_completion_agentic_loop,
)
from litellm.litellm_core_utils.completion_timeout import CompletionTimeout
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.get_litellm_params import (
    AWS_CREDENTIAL_KWARGS_KEYS as AWS_CREDENTIAL_KWARGS_KEYS,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.get_litellm_params import (
    OAUTH_TOKEN_EXCHANGE_KWARGS_KEYS as OAUTH_TOKEN_EXCHANGE_KWARGS_KEYS,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.get_litellm_params import OPTIONAL_KWARGS_KEYS
from litellm.litellm_core_utils.get_litellm_params import (
    PROVIDER_AFFINITY_HEADER_KWARG_KEY as PROVIDER_AFFINITY_HEADER_KWARG_KEY,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.get_litellm_params import (
    InvalidControlOption as InvalidControlOption,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.get_litellm_params import (
    parse_control_options as parse_control_options,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.get_litellm_params import (
    with_control_options as with_control_options,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.get_llm_provider_logic import (
    invalid_provider_routing_hints_message,
)
from litellm.litellm_core_utils.get_provider_specific_headers import (
    ProviderSpecificHeaderUtils as ProviderSpecificHeaderUtils,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.health_check_utils import (
    _filter_model_params,
    create_health_check_response,
)
from litellm.litellm_core_utils.hidden_params import HIDDEN_PARAMS_ATTR, get_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.mock_functions import (
    mock_embedding,
    mock_image_generation,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_content_from_model_response as get_content_from_model_response,  # noqa: PLC0414  # public re-export
)
from litellm.litellm_core_utils.provider_affinity import (
    add_provider_affinity_header as add_provider_affinity_header,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.request_timeout_resolver import (
    get_configured_request_timeout as get_configured_request_timeout,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
from litellm.llms.base_llm import (
    BaseConfig as BaseConfig,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.base_llm import BaseImageGenerationConfig
from litellm.llms.base_llm.base_model_iterator import (
    convert_model_response_to_streaming as convert_model_response_to_streaming,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.base_llm.chat.transformation import (
    with_attribution_headers as with_attribution_headers,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.bedrock.common_utils import (
    BedrockModelInfo as BedrockModelInfo,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.bedrock.common_utils import (
    bedrock_route_for_request as bedrock_route_for_request,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.bedrock.common_utils import (
    without_bedrock_route_prefix as without_bedrock_route_prefix,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.bedrock_mantle.chat.claude_transformation import (
    bedrock_mantle_chat_config as bedrock_mantle_chat_config,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.cohere.common_utils import (
    CohereModelInfo as CohereModelInfo,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.llms.custom_httpx.http_handler import (
    http2_enabled as http2_enabled,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.openai.chat.gpt_5_transformation import (
    OpenAIGPT5Config as OpenAIGPT5Config,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.openai_like.json_loader import (
    JSONProviderRegistry as JSONProviderRegistry,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.vertex_ai.common_utils import (
    VertexAIModelRoute as VertexAIModelRoute,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.llms.vertex_ai.common_utils import (
    get_vertex_ai_model_route as get_vertex_ai_model_route,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.native.chat import execute_chat
from litellm.native.chat import resolve_dispatched_chat_response as _resolve_dispatched_chat_response
from litellm.native.chat_mock import MOCK_RESPONSE_TYPE as MOCK_RESPONSE_TYPE  # noqa: PLC0414  # litellm.* re-export
from litellm.native.chat_mock import handle_mock_timeout_async as _handle_mock_timeout_async
from litellm.native.chat_mock import mock_completion as mock_completion  # noqa: PLC0414  # litellm.* re-export
from litellm.native.chat_providers.handlers import *  # noqa: F403  # litellm.* re-export of the moved handlers
from litellm.native.chat_providers.support import print_verbose
from litellm.native.request_support import (
    admission_input_tokens as admission_input_tokens,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.native.request_support import (
    admitted_prompt_token_counter as admitted_prompt_token_counter,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.native.request_support import proxy_auth_headers as _proxy_auth_headers
from litellm.native.request_support import register_custom_pricing_for_request as _register_custom_pricing_for_request
from litellm.native.request_support import resolve_control_options as _resolve_control_options
from litellm.realtime_api.main import realtime_health_check
from litellm.secret_managers.main import (
    get_secret_bool as get_secret_bool,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.secret_managers.main import (
    get_secret_str,
)
from litellm.types.completion import (
    CompletionDispatchContext as CompletionDispatchContext,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.types.litellm_params import (
    ControlOptions as ControlOptions,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.types.litellm_params import RetryStrategy
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    BaseUrlRoutingHint,
    ProviderRoutingHints,
    RawRequestTypedDict,
    StreamingChoices,
    is_decisions_model_mode,
)
from litellm.types.utils import (
    ModelResponseStream as ModelResponseStream,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.types.workload_identity import (
    ANTHROPIC_WIF_KWARGS_KEYS as ANTHROPIC_WIF_KWARGS_KEYS,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.types.workload_identity import (
    OPENAI_WIF_KWARGS_KEYS as OPENAI_WIF_KWARGS_KEYS,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    Choices,
    CustomStreamWrapper,
    EmbeddingResponse,
    Message,
    ModelResponse,
    ProviderConfigManager,
    TextChoices,
    TextCompletionResponse,
    TextCompletionStreamWrapper,
    TranscriptionResponse,
    add_provider_specific_params_to_optional_params,
    convert_to_model_response_object,
    create_pretrained_tokenizer,
    create_tokenizer,
    get_llm_provider,
    get_model_info,
    get_non_default_completion_params,
    get_non_default_transcription_params,
    get_optional_params_embeddings,
    get_optional_params_image_gen,
    get_optional_params_transcription,
    get_secret,
    get_standard_openai_params,
    read_config_args,
    should_run_mock_completion,
    validate_and_fix_openai_messages,
    validate_and_fix_openai_tools,
    validate_and_fix_thinking_param,
    validate_chat_completion_tool_choice,
    validate_openai_optional_params,
)
from litellm.utils import (
    Usage as Usage,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    async_mock_completion_streaming_obj as async_mock_completion_streaming_obj,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    get_requester_metadata as get_requester_metadata,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    mock_completion_streaming_obj as mock_completion_streaming_obj,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    peek_reasoning_summary_aliases as peek_reasoning_summary_aliases,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    pre_process_non_default_params as pre_process_non_default_params,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    strip_reasoning_summary_aliases_from_optional_params as strip_reasoning_summary_aliases_from_optional_params,  # noqa: PLC0414  # litellm.* re-export
)
from litellm.utils import (
    token_counter as token_counter,  # noqa: PLC0414  # public re-export
)

from ._logging import verbose_logger
from .caching.caching import disable_cache, enable_cache, update_cache
from .litellm_core_utils.core_helpers import (
    normalize_drop_params,
)
from .litellm_core_utils.core_helpers import (
    safe_deep_copy as safe_deep_copy,  # noqa: PLC0414  # litellm.* re-export
)
from .litellm_core_utils.fallback_utils import (
    async_completion_with_fallbacks,
    completion_with_fallbacks,
)
from .litellm_core_utils.prompt_templates.common_utils import add_system_prompt_to_messages, get_completion_messages
from .litellm_core_utils.prompt_templates.common_utils import (
    update_messages_with_model_file_ids as update_messages_with_model_file_ids,  # noqa: PLC0414  # litellm.* re-export
)
from .litellm_core_utils.prompt_templates.factory import custom_prompt, ollama_pt, prompt_factory
from .litellm_core_utils.prompt_templates.factory import (
    function_call_prompt as function_call_prompt,  # noqa: PLC0414  # litellm.* re-export
)
from .litellm_core_utils.prompt_templates.factory import (
    map_system_message_pt as map_system_message_pt,  # noqa: PLC0414  # litellm.* re-export
)
from .litellm_core_utils.prompt_templates.factory import (
    stringify_json_tool_call_content as stringify_json_tool_call_content,  # noqa: PLC0414  # litellm.* re-export
)
from .litellm_core_utils.streaming_chunk_builder_utils import (
    ChunkProcessor as ChunkProcessor,  # noqa: PLC0414  # public re-export
)
from .llms.azure.audio_transcriptions import AzureAudioTranscription
from .llms.azure.azure import (
    check_dynamic_azure_params as check_dynamic_azure_params,  # noqa: PLC0414  # litellm.* re-export
)
from .llms.azure_ai.embed import AzureAIEmbedding
from .llms.bedrock.embed.embedding import BedrockEmbedding
from .llms.bedrock.image_edit.handler import BedrockImageEdit
from .llms.bedrock.image_generation.image_handler import BedrockImageGeneration
from .llms.clarifai.chat.transformation import ClarifaiConfig
from .llms.cohere.embed import handler as cohere_embed
from .llms.custom_llm import (
    CustomLLM,
)
from .llms.custom_llm import (
    custom_chat_llm_router as custom_chat_llm_router,  # noqa: PLC0414  # litellm.* re-export
)
from .llms.databricks.common_utils import DatabricksBase
from .llms.databricks.embed.handler import DatabricksEmbeddingHandler
from .llms.deprecated_providers import (
    aleph_alpha as aleph_alpha,  # noqa: PLC0414  # litellm.* re-export
)
from .llms.gemini.common_utils import get_api_key_from_env
from .llms.groq.chat.handler import GroqChatCompletion
from .llms.heroku.chat.transformation import HerokuChatConfig
from .llms.huggingface.embedding.handler import HuggingFaceEmbedding
from .llms.nvidia_riva.audio_transcription.handler import (
    NvidiaRivaAudioTranscription,
)
from .llms.nvidia_riva.audio_transcription.transformation import (
    NvidiaRivaAudioTranscriptionConfig,
)
from .llms.oci.chat.transformation import OCIChatConfig
from .llms.ollama.completion import handler as ollama
from .llms.oobabooga.chat import oobabooga
from .llms.openai.image_variations.handler import OpenAIImageVariationsHandler
from .llms.openai.transcriptions.handler import OpenAIAudioTranscription
from .llms.openai.workload_identity import build_openai_client
from .llms.openai_like.embedding.handler import OpenAILikeEmbeddingHandler
from .llms.sagemaker.chat.handler import SagemakerChatHandler
from .llms.sap.chat.handler import GenAIHubOrchestration
from .llms.vertex_ai import (
    vertex_ai_non_gemini as vertex_ai_non_gemini,  # noqa: PLC0414  # litellm.* re-export
)
from .llms.vertex_ai.gemini_embeddings.batch_embed_content_handler import (
    GoogleBatchEmbeddings,
)
from .llms.vertex_ai.image_generation.image_generation_handler import (
    VertexImageGeneration,
)
from .llms.vertex_ai.multimodal_embeddings.embedding_handler import (
    VertexMultimodalEmbedding,
)
from .llms.vertex_ai.vertex_embeddings.embedding_handler import VertexEmbedding
from .llms.watsonx.common_utils import IBMWatsonXMixin
from .types.llms.anthropic import AnthropicThinkingParam
from .types.llms.openai import (
    ChatCompletionAssistantMessage,
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContentParam,
    ChatCompletionUserMessage,
    HttpxBinaryResponseContent,
    OpenAIModerationResponse,
    OpenAIWebSearchOptions,
)
from .types.utils import (
    AdapterCompletionStreamWrapper,
    CompletionTokensDetails,
    FileTypes,
    HiddenParams,
    LlmProviders,
    PromptTokensDetails,
)
from .types.utils import (
    ChatCompletionMessageToolCall as ChatCompletionMessageToolCall,  # noqa: PLC0414  # litellm.* re-export
)
from .types.utils import (
    ProviderSpecificHeader as ProviderSpecificHeader,  # noqa: PLC0414  # litellm.* re-export
)

####### ENVIRONMENT VARIABLES ###################
openai_audio_transcriptions: Final = OpenAIAudioTranscription()
nvidia_riva_audio_transcriptions: Final = NvidiaRivaAudioTranscription()
openai_image_variations = OpenAIImageVariationsHandler()
groq_chat_completions: Final = GroqChatCompletion()
sap_gen_ai_hub_emb: Final = GenAIHubOrchestration()
azure_ai_embedding: Final = AzureAIEmbedding()
azure_audio_transcriptions: Final = AzureAudioTranscription()
huggingface_embed: Final = HuggingFaceEmbedding()
bedrock_embedding: Final = BedrockEmbedding()
bedrock_image_generation = BedrockImageGeneration()
bedrock_image_edit = BedrockImageEdit()
vertex_embedding: Final = VertexEmbedding()
vertex_multimodal_embedding: Final = VertexMultimodalEmbedding()
vertex_image_generation: Final = VertexImageGeneration()
google_batch_embeddings: Final = GoogleBatchEmbeddings()
# vertex_text_to_speech is now replaced by VertexAITextToSpeechConfig
openai_like_embedding: Final = OpenAILikeEmbeddingHandler()
databricks_embedding: Final = DatabricksEmbeddingHandler()
sagemaker_chat_completion: Final = SagemakerChatHandler()
heroku_transformation: Final = HerokuChatConfig()
oci_transformation: Final = OCIChatConfig()


####### COMPLETION ENDPOINTS ################


class LiteLLM:
    def __init__(
        self,
        *,
        api_key=None,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float | None = 600,
        max_retries: int | None = litellm.num_retries,
        default_headers: Mapping[str, str] | None = None,
    ):
        self.params = locals()
        self.chat = Chat(self.params, router_obj=None)


class Chat:
    def __init__(self, params, router_obj: "Router | None"):
        self.params = params
        if self.params.get("acompletion", False) is True:
            self.params.pop("acompletion")
            self.completions: AsyncCompletions | Completions = AsyncCompletions(self.params, router_obj=router_obj)
        else:
            self.completions = Completions(self.params, router_obj=router_obj)


class Completions:
    def __init__(self, params, router_obj: "Router | None"):
        self.params = params
        self.router_obj = router_obj

    def create(self, messages, model=None, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
        model = model or self.params.get("model")
        if self.router_obj is not None:
            response = self.router_obj.completion(model=model, messages=messages, **self.params)
        else:
            response = completion(model=model, messages=messages, **self.params)
        return response


class AsyncCompletions:
    def __init__(self, params, router_obj: "Router | None"):
        self.params = params
        self.router_obj = router_obj

    async def create(self, messages, model=None, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
        model = model or self.params.get("model")
        if self.router_obj is not None:
            response = await self.router_obj.acompletion(model=model, messages=messages, **self.params)
        else:
            response = await acompletion(model=model, messages=messages, **self.params)
        return response


@tracer.wrap()
@client
async def acompletion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: list = [],
    functions: list | None = None,
    function_call: str | None = None,
    timeout: float | httpx.Timeout | openai.Timeout | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stream_options: dict | None = None,
    stop=None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    modalities: list[ChatCompletionModality] | None = None,
    prediction: ChatCompletionPredictionContentParam | None = None,
    audio: ChatCompletionAudioParam | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict | None = None,
    user: str | None = None,
    # openai v1.0+ new params
    response_format: dict | type[BaseModel] | None = None,
    seed: int | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    parallel_tool_calls: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    deployment_id=None,
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    safety_identifier: str | None = None,
    service_tier: str | None = None,
    store: bool | None = None,
    prompt_cache_key: str | None = None,
    # set api_base, api_version, api_key
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,  # pass in a list of api_base,keys, etc.
    extra_headers: dict | None = None,
    # Optional liteLLM function params
    thinking: AnthropicThinkingParam | None = None,
    web_search_options: OpenAIWebSearchOptions | None = None,
    include_server_side_tool_invocations: bool | None = None,
    # Session management
    shared_session: Optional["ClientSession"] = None,
    # Per-request JSON schema validation (overrides litellm.enable_json_schema_validation)
    enable_json_schema_validation: bool | None = None,
    **kwargs,
) -> ModelResponse | CustomStreamWrapper:
    """
    Asynchronously executes a litellm.completion() call for any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)

    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stream_options (dict, optional): A dictionary containing options for the streaming response. Only use this if stream is True.
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        max_completion_tokens (integer, optional): An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
        modalities (List[ChatCompletionModality], optional): Output types that you would like the model to generate for this request. You can use `["text", "audio"]`
        prediction (ChatCompletionPredictionContentParam, optional): Configuration for a Predicted Output, which can greatly improve response times when large parts of the model response are known ahead of time. This is most common when you are regenerating a file with only minor changes to most of the content.
        audio (ChatCompletionAudioParam, optional): Parameters for audio output. Required when audio output is requested with modalities: ["audio"]
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc.
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys
        timeout (float, optional): The maximum execution time in seconds for the completion request.

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Notes:
        - This function is an asynchronous version of the `completion` function.
        - The `completion` function is called using `run_in_executor` to execute synchronously in the event loop.
        - If `stream` is True, the function returns an async generator that yields completion lines.
    """
    fallbacks = kwargs.get("fallbacks", None)
    mock_timeout = kwargs.get("mock_timeout", None)

    if mock_timeout is True:
        stream_timeout: Final = kwargs.get("stream_timeout")
        await _handle_mock_timeout_async(
            mock_timeout,
            stream_timeout
            if stream is True and isinstance(stream_timeout, int | float | str | httpx.Timeout)
            else timeout,
            model,
        )

    loop: Final = asyncio.get_event_loop()
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    _ = _resolve_control_options(kwargs, model)

    ## PROMPT MANAGEMENT HOOKS ##
    #########################################################
    #########################################################
    litellm_logging_obj: Final = kwargs.get("litellm_logging_obj", None)

    from litellm.integrations.anthropic_cache_control_hook import (
        AnthropicCacheControlHook,
    )

    AnthropicCacheControlHook.maybe_seed_default_injection_points(
        non_default_params=kwargs,
        messages=cast(list[AllMessageValues], messages),  # cast-ok: acompletion types messages as a bare List
        model=model,
        custom_llm_provider=cast(str | None, custom_llm_provider),  # cast-ok: read from untyped kwargs
        tools=tools,
        enable_prompt_caching=cast(bool | None, kwargs.get("enable_prompt_caching")),  # cast-ok: untyped kwargs
        api_base=kwargs.get("api_base") or base_url,
    )

    if isinstance(litellm_logging_obj, LiteLLMLoggingObj) and (
        litellm_logging_obj.should_run_prompt_management_hooks(
            prompt_id=kwargs.get("prompt_id", None),
            non_default_params=kwargs,
            tools=tools,
        )
    ):
        (
            model,
            messages,
            _,
        ) = await litellm_logging_obj.async_get_chat_completion_prompt(
            model=model,
            messages=messages,
            non_default_params=kwargs,
            prompt_id=kwargs.get("prompt_id", None),
            prompt_variables=kwargs.get("prompt_variables", None),
            tools=tools,
            prompt_label=kwargs.get("prompt_label", None),
            prompt_version=kwargs.get("prompt_version", None),
            request_kwargs=kwargs,
        )
        #########################################################
        # if the chat completion logging hook removed all tools,
        # set tools to None
        # eg. in certain cases when users send vector stores as tools
        # we don't want the tools to go to the upstream llm
        # relevant issue: https://github.com/BerriAI/litellm/issues/11404
        #########################################################
        if tools is not None and len(tools) == 0:
            tools = None

    #########################################################
    #########################################################

    # Log shared session usage
    if shared_session is not None:
        verbose_logger.debug("🔄 SHARED SESSION: acompletion called with shared_session (ID: %s)", id(shared_session))
    else:
        verbose_logger.debug("🔄 NO SHARED SESSION: acompletion called without shared_session")

    # Adjusted to use explicit arguments instead of *args and **kwargs
    completion_kwargs: Final = {
        "model": model,
        "messages": messages,
        "functions": functions,
        "function_call": function_call,
        "timeout": timeout,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stream_options": stream_options,
        "stop": stop,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
        "modalities": modalities,
        "prediction": prediction,
        "audio": audio,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "user": user,
        "response_format": response_format,
        "seed": seed,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "deployment_id": deployment_id,
        "base_url": base_url,
        "api_version": api_version,
        "api_key": api_key,
        "model_list": model_list,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
        "safety_identifier": safety_identifier,
        "service_tier": service_tier,
        "store": store,
        "prompt_cache_key": prompt_cache_key,
        "extra_headers": extra_headers,
        "acompletion": True,  # assuming this is a required parameter
        "thinking": thinking,
        "web_search_options": web_search_options,
        "include_server_side_tool_invocations": include_server_side_tool_invocations,
        "shared_session": shared_session,
        "enable_json_schema_validation": enable_json_schema_validation,
    }
    if custom_llm_provider is None:
        _supplemental: Final = cast(  # cast-ok: kwargs values are request-scoped objects
            "dict[str, object]", {k: kwargs[k] for k in OPTIONAL_KWARGS_KEYS if k in kwargs}
        )
        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=cast(  # cast-ok: api_base arrives as a str in the request kwargs
                "str | None", kwargs.get("api_base")
            )
            or base_url,
            litellm_params=(GenericLiteLLMParams.model_validate(_supplemental) if _supplemental else None),
        )

    fallbacks = fallbacks or litellm.model_fallbacks
    if fallbacks is not None:
        response = await async_completion_with_fallbacks(**completion_kwargs, kwargs={"fallbacks": fallbacks, **kwargs})
        if response is None:
            raise Exception(
                "No response from fallbacks. Got none. Turn on `litellm.set_verbose=True` to see more details."
            )
        return response

    ### APPLY MOCK DELAY ###

    mock_delay: Final = kwargs.get("mock_delay")
    mock_response: Final = kwargs.get("mock_response")
    mock_tool_calls: Final = kwargs.get("mock_tool_calls")
    mock_timeout = kwargs.get("mock_timeout")
    if mock_delay and should_run_mock_completion(
        mock_response=mock_response,
        mock_tool_calls=mock_tool_calls,
        mock_timeout=mock_timeout,
    ):
        await asyncio.sleep(mock_delay)

    try:
        # Use a partial function to pass your keyword arguments
        kwargs.pop("acompletion", None)
        func: Final = partial(completion, **completion_kwargs, **kwargs)

        # Add the context to the function
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)

        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if isinstance(init_response, dict) or isinstance(init_response, ModelResponse):  ## CACHING SCENARIO
            if isinstance(init_response, dict):
                response = _model_response_from_cached_dict(init_response)
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await _resolve_dispatched_chat_response(init_response)
        else:
            response = init_response

        if (
            custom_llm_provider == "text-completion-openai"
            or custom_llm_provider == "text-completion-codestral"
            or custom_llm_provider == "text-completion-inception"
        ) and isinstance(response, TextCompletionResponse):
            response = litellm.OpenAITextCompletionConfig().convert_to_chat_model_response_object(
                response_object=response,
                model_response_object=litellm.ModelResponse(),
            )
        # Provider-agnostic dispatch point for the chat-completions agentic loop
        # (code-interpreter interception, etc). Chat routing forks per provider
        # before this (OpenAI goes through the OpenAI SDK in openai.py, others
        # through the shared httpx handler), so a dispatch inside any single
        # provider handler would miss the others. Here is where every fork
        # reconverges, so the loop runs once for all providers. Responses needs
        # no equivalent: every provider already funnels through one shared
        # handler where the loop is dispatched.
        if isinstance(response, litellm.ModelResponse):
            looped: Final = await maybe_run_chat_completion_agentic_loop(
                response=response,
                model=model,
                messages=messages,
                optional_params={
                    k: v
                    for k, v in completion_kwargs.items()
                    if v is not None
                    and k
                    not in (
                        "model",
                        "messages",
                        "stream",
                        "acompletion",
                        "deployment_id",
                    )
                },
                kwargs=kwargs,
                logging_obj=kwargs.get("litellm_logging_obj"),
                custom_llm_provider=custom_llm_provider,
                stream=bool(stream),
            )
            if looped is not None:
                response = looped
        if isinstance(response, CustomStreamWrapper):
            response.set_logging_event_loop(
                loop=loop
            )  # sets the logging event loop if the user does sync streaming (e.g. on proxy for sagemaker calls)
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )


def _model_response_from_cached_dict(cached_response_dict: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(**cached_response_dict)


def _transcription_response_from_cached_dict(cached_response_dict: Mapping[str, object]) -> TranscriptionResponse:
    return TranscriptionResponse(**cached_response_dict)


async def _async_streaming(response, model, custom_llm_provider, args):
    try:
        print_verbose(f"received response in _async_streaming: {response}")
        if asyncio.iscoroutine(response):
            response = await response
        async for line in response:
            print_verbose(f"line in async streaming: {line}")
            yield line
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
        )


from litellm.responses.bridge_check import (  # noqa: E402  # re-export at the old definition site
    responses_api_bridge_check as responses_api_bridge_check,  # pyright: ignore[reportUnknownVariableType]  # noqa: PLC0414  # re-export of moved function
)


def _locals_snapshot(values: Mapping[str, object]) -> Mapping[str, object]:
    return values


_PROVIDER_ROUTING_HINTS: Final = TypeAdapter(ProviderRoutingHints)
_BASE_URL_HINT: Final = TypeAdapter(BaseUrlRoutingHint)


@tracer.wrap()
@client
def completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: list = [],
    timeout: float | str | httpx.Timeout | openai.Timeout | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stream_options: dict[str, object] | None = None,
    stop=None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    modalities: list[ChatCompletionModality] | None = None,
    prediction: ChatCompletionPredictionContentParam | None = None,
    audio: ChatCompletionAudioParam | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict[str, float] | None = None,
    user: str | None = None,
    # openai v1.0+ new params
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    response_format: dict[str, object] | type[BaseModel] | None = None,
    seed: int | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    parallel_tool_calls: bool | None = None,
    web_search_options: OpenAIWebSearchOptions | None = None,
    include_server_side_tool_invocations: bool | None = None,
    deployment_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
    safety_identifier: str | None = None,
    service_tier: str | None = None,
    store: bool | None = None,
    prompt_cache_key: str | None = None,
    # soon to be deprecated params by OpenAI
    functions: list[Mapping[str, object]] | None = None,
    function_call: str | None = None,
    # set api_base, api_version, api_key
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    thinking: AnthropicThinkingParam | None = None,
    # Session management
    shared_session: Optional["ClientSession"] = None,
    # Per-request JSON schema validation (overrides litellm.enable_json_schema_validation)
    enable_json_schema_validation: bool | None = None,
    **kwargs,
) -> ModelResponse | CustomStreamWrapper:
    """
    Perform a completion() using any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)
    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stream_options (dict, optional): A dictionary containing options for the streaming response. Only set this when you set stream: true.
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        max_completion_tokens (integer, optional): An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
        modalities (List[ChatCompletionModality], optional): Output types that you would like the model to generate for this request.. You can use `["text", "audio"]`
        prediction (ChatCompletionPredictionContentParam, optional): Configuration for a Predicted Output, which can greatly improve response times when large parts of the model response are known ahead of time. This is most common when you are regenerating a file with only minor changes to most of the content.
        audio (ChatCompletionAudioParam, optional): Parameters for audio output. Required when audio output is requested with modalities: ["audio"]
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        logprobs (bool, optional): Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message
        top_logprobs (int, optional): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc.
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys
        extra_headers (dict, optional): Additional headers to include in the request.

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
        max_retries (int, optional): The number of retries to attempt (default is 0).
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Note:
        - This function is used to perform completions() using the specified language model.
        - It supports various optional parameters for customizing the completion behavior.
        - If 'mock_response' is provided, a mock completion response is returned for testing or debugging.
    """
    ### VALIDATE Request ###
    if model is None:
        raise ValueError("model param not passed in.")
    # validate messages
    messages = validate_and_fix_openai_messages(messages=messages)
    tools = validate_and_fix_openai_tools(tools=tools)
    # validate tool_choice
    tool_choice = validate_chat_completion_tool_choice(tool_choice=tool_choice, model=model)
    # validate optional params
    stop = validate_openai_optional_params(stop=stop)
    thinking = validate_and_fix_thinking_param(thinking=thinking)

    ######### unpacking kwargs #####################
    args: Final = _locals_snapshot(locals())

    control_options: Final = _resolve_control_options(kwargs, model)
    skip_mcp_handler: Final = kwargs.pop("_skip_mcp_handler", False)
    if not skip_mcp_handler and tools:
        from litellm.responses.mcp.chat_completions_handler import acompletion_with_mcp
        from litellm.responses.mcp.litellm_proxy_mcp_handler import (
            LiteLLM_Proxy_MCP_Handler,
        )
        from litellm.types.llms.openai import ToolParam

        # Check if MCP tools are present (following responses pattern)
        # Cast tools to Optional[Iterable[ToolParam]] for type checking
        tools_for_mcp: Final = cast(Iterable[ToolParam] | None, tools)
        if LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(tools=tools_for_mcp):
            return acompletion_with_mcp(  # pyright: ignore[reportReturnType]  # MCP path returns a coroutine that acompletion() awaits; completion()'s sync return type omits it
                model=model,
                messages=messages,
                functions=functions,
                function_call=function_call,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stream=stream,
                stream_options=stream_options,
                stop=stop,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                response_format=response_format,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                safety_identifier=safety_identifier,
                service_tier=service_tier,
                store=store,
                prompt_cache_key=prompt_cache_key,
                base_url=base_url,
                api_version=api_version,
                api_key=api_key,
                model_list=model_list,
                extra_headers=extra_headers,
                thinking=thinking,
                web_search_options=web_search_options,
                shared_session=shared_session,
                enable_json_schema_validation=enable_json_schema_validation,
                **kwargs,
            )
    custom_llm_provider: Final = kwargs.get("custom_llm_provider", None)
    litellm_logging_obj: Final = kwargs.get("litellm_logging_obj", None)
    fallbacks = kwargs.get("fallbacks", None)
    ensure_alternating_roles: Final[bool | None] = kwargs.get("ensure_alternating_roles", None)
    user_continue_message: Final[ChatCompletionUserMessage | None] = kwargs.get("user_continue_message", None)
    assistant_continue_message: ChatCompletionAssistantMessage | None = kwargs.get("assistant_continue_message", None)
    ### PROMPT MANAGEMENT ###
    prompt_id: Final = cast(str | None, kwargs.get("prompt_id", None))
    prompt_variables: Final = cast(dict | None, kwargs.get("prompt_variables", None))
    litellm_system_prompt: Final = kwargs.get("litellm_system_prompt", None)
    ### COPY MESSAGES ### - related issue https://github.com/BerriAI/litellm/discussions/4489
    messages = get_completion_messages(
        messages=messages,
        ensure_alternating_roles=ensure_alternating_roles or False,
        user_continue_message=user_continue_message,
        assistant_continue_message=assistant_continue_message,
    )
    ######## end of unpacking kwargs ###########
    non_default_params: Final = get_non_default_completion_params(kwargs=kwargs)
    ## PROMPT MANAGEMENT HOOKS ##

    from litellm.integrations.anthropic_cache_control_hook import (
        AnthropicCacheControlHook,
    )

    AnthropicCacheControlHook.maybe_seed_default_injection_points(
        non_default_params=non_default_params,
        messages=cast(list[AllMessageValues], messages),  # cast-ok: completion types messages as a bare List
        model=model,
        custom_llm_provider=cast(str | None, kwargs.get("custom_llm_provider")),  # cast-ok: untyped kwargs
        tools=tools,
        enable_prompt_caching=cast(bool | None, kwargs.get("enable_prompt_caching")),  # cast-ok: untyped kwargs
        api_base=kwargs.get("api_base") or base_url,
    )

    if isinstance(litellm_logging_obj, LiteLLMLoggingObj) and (
        litellm_logging_obj.should_run_prompt_management_hooks(
            prompt_id=prompt_id, non_default_params=non_default_params
        )
    ):
        (
            model,
            messages,
            _,
        ) = litellm_logging_obj.get_chat_completion_prompt(
            model=model,
            messages=messages,
            non_default_params=non_default_params,
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            prompt_label=kwargs.get("prompt_label", None),
            prompt_version=kwargs.get("prompt_version", None),
            request_kwargs=kwargs,
        )

    ### LITELLM SYSTEM PROMPT ###
    if litellm_system_prompt:
        messages = add_system_prompt_to_messages(
            messages=messages,
            system_prompt=litellm_system_prompt,
            merge_with_first_system=True,
        )

    try:
        fallbacks = fallbacks or litellm.model_fallbacks
        if fallbacks is not None:
            return completion_with_fallbacks(  # pyright: ignore[reportReturnType]  # fallback runner is untyped; resolves to ModelResponse|CustomStreamWrapper at runtime
                **args
            )
        if model_list is not None:
            deployments: Final = [m["litellm_params"] for m in model_list if m["model_name"] == model]
            return litellm.batch_completion_models(  # pyright: ignore[reportReturnType]  # batch path returns a list of responses, outside completion()'s single-response return type
                deployments=deployments, **args
            )
        return execute_chat(
            model=model,
            messages=messages,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            modalities=modalities,
            prediction=prediction,
            audio=audio,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            web_search_options=web_search_options,
            include_server_side_tool_invocations=include_server_side_tool_invocations,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            safety_identifier=safety_identifier,
            service_tier=service_tier,
            store=store,
            prompt_cache_key=prompt_cache_key,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            thinking=thinking,
            shared_session=shared_session,
            kwargs=kwargs,
            non_default_params=non_default_params,
            control_options=control_options,
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
        )
    except Exception as e:
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


def completion_with_retries(*args, **kwargs):
    """
    Executes a litellm.completion() with 3 retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(f"tenacity import failed please run `pip install tenacity`. Error{e}")

    num_retries: Final = kwargs.pop("num_retries", 3)
    # reset retries in .completion()
    kwargs["max_retries"] = 0
    kwargs["num_retries"] = 0
    retry_strategy: Final[RetryStrategy] = kwargs.pop("retry_strategy", "constant_retry")
    original_function: Final = kwargs.pop("original_function", completion)
    if retry_strategy == "exponential_backoff_retry":
        retryer = tenacity.Retrying(
            wait=tenacity.wait_exponential(multiplier=1, max=10),
            stop=tenacity.stop_after_attempt(num_retries),
            reraise=True,
        )
    else:
        retryer = tenacity.Retrying(stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    return retryer(original_function, *args, **kwargs)


async def acompletion_with_retries(*args, **kwargs):
    """
    [DEPRECATED]. Use 'acompletion' or router.acompletion instead!
    Executes a litellm.completion() with 3 retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(f"tenacity import failed please run `pip install tenacity`. Error{e}")

    num_retries: Final = kwargs.pop("num_retries", 3)
    kwargs["max_retries"] = 0
    kwargs["num_retries"] = 0
    retry_strategy: Final[RetryStrategy] = kwargs.pop("retry_strategy", "constant_retry")
    original_function: Final = kwargs.pop("original_function", completion)
    if retry_strategy == "exponential_backoff_retry":
        retryer = tenacity.AsyncRetrying(
            wait=tenacity.wait_exponential(multiplier=1, max=10),
            stop=tenacity.stop_after_attempt(num_retries),
            reraise=True,
        )
    else:
        retryer = tenacity.AsyncRetrying(stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    return await retryer(original_function, *args, **kwargs)


def responses_with_retries(*args, **kwargs):
    """
    Executes a litellm.responses() with retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(f"tenacity import failed please run `pip install tenacity`. Error{e}")

    from litellm.responses.dispatch import responses

    num_retries: Final = kwargs.pop("num_retries", 3)
    # reset retries in .responses()
    kwargs["max_retries"] = 0
    kwargs["num_retries"] = 0
    retry_strategy: Final[RetryStrategy] = kwargs.pop("retry_strategy", "constant_retry")
    original_function: Final = kwargs.pop("original_function", responses)
    if retry_strategy == "exponential_backoff_retry":
        retryer = tenacity.Retrying(
            wait=tenacity.wait_exponential(multiplier=1, max=10),
            stop=tenacity.stop_after_attempt(num_retries),
            reraise=True,
        )
    else:
        retryer = tenacity.Retrying(stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    return retryer(original_function, *args, **kwargs)


async def aresponses_with_retries(*args, **kwargs):
    """
    Executes a litellm.aresponses() with retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(f"tenacity import failed please run `pip install tenacity`. Error{e}")

    from litellm.responses.dispatch import aresponses

    num_retries: Final = kwargs.pop("num_retries", 3)
    kwargs["max_retries"] = 0
    kwargs["num_retries"] = 0
    retry_strategy: Final[RetryStrategy] = kwargs.pop("retry_strategy", "constant_retry")
    original_function: Final = kwargs.pop("original_function", aresponses)
    if retry_strategy == "exponential_backoff_retry":
        retryer = tenacity.AsyncRetrying(
            wait=tenacity.wait_exponential(multiplier=1, max=10),
            stop=tenacity.stop_after_attempt(num_retries),
            reraise=True,
        )
    else:
        retryer = tenacity.AsyncRetrying(stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    return await retryer(original_function, *args, **kwargs)


### EMBEDDING ENDPOINTS ####################
@client
async def aembedding(*args, **kwargs) -> EmbeddingResponse:
    """
    Asynchronously calls the `embedding` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `embedding` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `embedding` function.

    Returns:
    - `response` (Any): The response returned by the `embedding` function.
    """
    loop: Final = asyncio.get_event_loop()
    model: Final = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Embedding ###
    kwargs["aembedding"] = True
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    try:
        # Use a partial function to pass your keyword arguments
        func: Final = partial(embedding, *args, **kwargs)

        # Add the context to the function
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=kwargs.get("api_base", None),
        )

        # Await normally
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        response: EmbeddingResponse | None = None
        if isinstance(init_response, dict):
            response = EmbeddingResponse(**init_response)
        elif isinstance(init_response, EmbeddingResponse):  ## CACHING SCENARIO
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response
        if response is not None and isinstance(response, EmbeddingResponse) and hasattr(response, "_hidden_params"):
            response_hidden_params: Final = get_hidden_params(response)
            if response_hidden_params is not None:
                response_hidden_params["custom_llm_provider"] = custom_llm_provider

        if response is None:
            raise ValueError("Unable to get Embedding Response. Please pass a valid llm_provider.")
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


# fmt: off

# Overload for when aembedding=True (returns coroutine)
@overload
def embedding(
    model,
    input=[],
    # Optional params
    dimensions: int | None = None,
    encoding_format: str | None = None,
    timeout: float = 600,  # default to 10 minutes
    # set api_base, api_version, api_key
    api_base: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
    caching: bool = False,
    user: str | None = None,
    custom_llm_provider=None,
    litellm_call_id=None,
    logger_fn=None,
    *,
    aembedding: Literal[True],
    **kwargs,
) -> Coroutine[object, object, EmbeddingResponse]: 
    ...


# Overload for when aembedding=False or not specified (returns EmbeddingResponse)
@overload
def embedding(
    model,
    input=[],
    # Optional params
    dimensions: int | None = None,
    encoding_format: str | None = None,
    timeout: float = 600,  # default to 10 minutes
    # set api_base, api_version, api_key
    api_base: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
    caching: bool = False,
    user: str | None = None,
    custom_llm_provider=None,
    litellm_call_id=None,
    logger_fn=None,
    *,
    aembedding: Literal[False] = False,
    **kwargs,
) -> EmbeddingResponse: 
    ...

# fmt: on


@client
def embedding(
    model,
    input=[],
    # Optional params
    dimensions: int | None = None,
    encoding_format: str | None = None,
    timeout: float = 600,  # default to 10 minutes
    # set api_base, api_version, api_key
    api_base: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
    caching: bool = False,
    user: str | None = None,
    custom_llm_provider=None,
    litellm_call_id=None,
    logger_fn=None,
    **kwargs,
) -> EmbeddingResponse | Coroutine[object, object, EmbeddingResponse]:
    """
    Embedding function that calls an API to generate embeddings for the given input.

    Parameters:
    - model: The embedding model to use.
    - input: The input for which embeddings are to be generated.
    - encoding_format: Optional[str] The format to return the embeddings in. Can be either `float` or `base64`
    - dimensions: The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.
    - timeout: The timeout value for the API call, default 10 mins
    - litellm_call_id: The call ID for litellm logging.
    - litellm_logging_obj: The litellm logging object.
    - logger_fn: The logger function.
    - api_base: Optional. The base URL for the API.
    - api_version: Optional. The version of the API.
    - api_key: Optional. The API key to use.
    - api_type: Optional. The type of the API.
    - caching: A boolean indicating whether to enable caching.
    - custom_llm_provider: The custom llm provider.

    Returns:
    - response: The response received from the API call.

    Raises:
    - exception_type: If an exception occurs during the API call.
    """
    azure: Final = kwargs.get("azure", None)
    client: Final = kwargs.pop("client", None)
    drop_params_kwarg: Final = (
        cast(object, kwargs["drop_params"]) if "drop_params" in kwargs else None  # cast-ok: untyped request kwargs
    )
    drop_unsupported_params: Final = litellm.drop_params is True or normalize_drop_params(drop_params_kwarg) is True
    shared_session: Final = kwargs.get("shared_session", None)
    max_retries: Final = kwargs.get("max_retries", None)
    litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
    mock_response: Final[list[float] | None] = kwargs.get("mock_response", None)
    azure_ad_token_provider: Final = kwargs.get("azure_ad_token_provider", None)
    aembedding: Final[bool | None] = kwargs.get("aembedding", None)
    extra_headers: Final = kwargs.get("extra_headers", None)
    headers = kwargs.get("headers", None) or extra_headers
    if headers is None:
        headers = {}
    if extra_headers is not None:
        headers.update(extra_headers)
    # Inject proxy auth headers if configured
    if litellm.proxy_auth is not None:
        try:
            proxy_headers: Final = _proxy_auth_headers(litellm.proxy_auth)
            headers.update(proxy_headers)
        except Exception as e:
            verbose_logger.warning("Failed to get proxy auth headers: %s", e)
    ### CUSTOM MODEL COST ###
    input_cost_per_token: Final = kwargs.get("input_cost_per_token", None)
    output_cost_per_token: Final = kwargs.get("output_cost_per_token", None)
    cost_per_second: Final = kwargs.get("cost_per_second", None)
    input_cost_per_second: Final = kwargs.get("input_cost_per_second", None)
    output_cost_per_second: Final = kwargs.get("output_cost_per_second", None)
    openai_params: Final = [
        "user",
        "dimensions",
        "request_timeout",
        "api_base",
        "api_version",
        "api_key",
        "deployment_id",
        "organization",
        "base_url",
        "default_headers",
        "timeout",
        "max_retries",
        "encoding_format",
    ]
    default_params: Final = [*openai_params, "aembedding", "extra_headers"]
    non_default_params: Final = filter_out_litellm_params(kwargs, excluding=default_params)

    litellm_params_dict: Final = get_litellm_params(**kwargs)

    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
        litellm_params=GenericLiteLLMParams(**kwargs),
    )

    if dynamic_api_key is not None:
        api_key = dynamic_api_key

    allowed_openai_params: Final[list[str] | None] = (
        cast(  # cast-ok: allowed_openai_params arrives as a str list in the request kwargs
            "list[str] | None", kwargs.get("allowed_openai_params", None)
        )
    )
    optional_params: Final = get_optional_params_embeddings(
        model=model,
        user=user,
        dimensions=dimensions,
        encoding_format=encoding_format,
        custom_llm_provider=custom_llm_provider,
        allowed_openai_params=allowed_openai_params,
        **non_default_params,
    )

    ### REGISTER CUSTOM MODEL PRICING -- IF GIVEN ###
    if (
        (input_cost_per_token is not None and output_cost_per_token is not None)
        or input_cost_per_second is not None
        or output_cost_per_second is not None
        or cost_per_second is not None
    ):
        _register_custom_pricing_for_request(
            model=model,
            custom_llm_provider=custom_llm_provider,
            kwargs=kwargs,
            model_info=kwargs.get("model_info"),
        )

    logging: Final[LiteLLMLoggingObj] = litellm_logging_obj
    logging.update_environment_variables(
        model=model,
        user=user,
        optional_params=optional_params,
        litellm_params=litellm_params_dict,
        custom_llm_provider=custom_llm_provider,
    )

    if mock_response is not None:
        return mock_embedding(model=model, mock_response=mock_response)
    try:
        response: EmbeddingResponse | Coroutine[object, object, EmbeddingResponse] | None = None

        if azure is True or custom_llm_provider == "azure":
            # azure configs

            api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")

            api_version = (
                api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
                or litellm.AZURE_DEFAULT_API_VERSION
            )

            azure_ad_token: Final = optional_params.pop("azure_ad_token", None) or get_secret_str("AZURE_AD_TOKEN")

            api_key = api_key or litellm.api_key or litellm.azure_key or get_secret_str("AZURE_API_KEY")

            if api_base is None:
                raise ValueError("No API Base provided for Azure OpenAI LLM provider. Set 'AZURE_API_BASE' in .env")

            ## EMBEDDING CALL
            response = azure_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                azure_ad_token_provider=azure_ad_token_provider,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                max_retries=max_retries,
                headers=headers or extra_headers,
                litellm_params=litellm_params_dict,
            )
        elif custom_llm_provider == "github_copilot":
            api_key = api_key or litellm.api_key
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
            )
        elif (
            custom_llm_provider == "openai"
            or custom_llm_provider == "together_ai"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "litellm_proxy"
            or (model in litellm.open_ai_embedding_models and custom_llm_provider is None)
        ):
            api_base = (
                api_base
                or litellm.api_base
                or get_secret_str("OPENAI_BASE_URL")
                or get_secret_str("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            openai.organization = (
                litellm.organization
                or get_secret_str("OPENAI_ORGANIZATION")
                or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")

            if headers is not None and headers != {}:
                optional_params["extra_headers"] = headers

            requested_encoding_format: Final = (
                encoding_format
                or optional_params.get("encoding_format")
                or get_secret_str("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT")
            )
            if requested_encoding_format is None or requested_encoding_format.strip().lower() == "none":
                optional_params.pop("encoding_format", None)
            else:
                optional_params["encoding_format"] = requested_encoding_format

            api_version = None

            ## EMBEDDING CALL
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                max_retries=max_retries,
                shared_session=shared_session,
                litellm_params=litellm_params_dict,
            )
        elif custom_llm_provider == "databricks":
            api_base = api_base or litellm.api_base or get_secret("DATABRICKS_API_BASE")

            # set API KEY
            api_key = DatabricksBase.resolve_api_key(api_key)

            ## EMBEDDING CALL
            response = databricks_embedding.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "hosted_vllm":
            api_base = api_base or litellm.api_base or get_secret_str("HOSTED_VLLM_API_BASE")

            # set API KEY
            if api_key is None:
                api_key = litellm.api_key or get_secret_str("HOSTED_VLLM_API_KEY")

            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                headers=headers or {},
            )
        elif custom_llm_provider == "edenai":
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                headers=headers,
            )
        elif (
            custom_llm_provider == "openai_like"
            or custom_llm_provider == "llamafile"
            or custom_llm_provider == "lm_studio"
        ):
            api_base = api_base or litellm.api_base or get_secret_str("OPENAI_LIKE_API_BASE")

            # set API KEY
            if api_key is None:
                api_key = api_key or litellm.api_key or litellm.openai_like_key or get_secret_str("OPENAI_LIKE_API_KEY")

            if headers is not None and headers != {}:
                optional_params["extra_headers"] = headers

            ## EMBEDDING CALL
            response = openai_like_embedding.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "oci":
            if headers is None:
                headers = {}
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                headers=headers,
            )
        elif custom_llm_provider == "cohere" or custom_llm_provider == "cohere_chat":
            cohere_key: Final = (
                api_key
                or litellm.cohere_key
                or get_secret_str("COHERE_API_KEY")
                or get_secret_str("CO_API_KEY")
                or litellm.api_key
            )

            # Use the merged headers variable (already merged at the top of the function)
            # Don't overwrite it with just extra_headers
            if headers is None:
                headers = {}

            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=cohere_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                headers=headers,
            )
        elif custom_llm_provider == "openrouter":
            api_base = (
                api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.openrouter_key
                or get_secret_str("OPENROUTER_API_KEY")
                or get_secret_str("OR_API_KEY")
            )

            openrouter_site_url: Final = get_secret("OR_SITE_URL") or "https://litellm.ai"
            openrouter_app_name: Final = get_secret("OR_APP_NAME") or "liteLLM"

            openrouter_headers: Final = {
                "HTTP-Referer": openrouter_site_url,
                "X-Title": openrouter_app_name,
            }

            _headers: Final = headers or litellm.headers
            if _headers:
                openrouter_headers.update(_headers)

            headers = openrouter_headers

            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                headers=headers,
            )
        elif custom_llm_provider == "vercel_ai_gateway":
            api_base = (
                api_base
                or litellm.api_base
                or get_secret_str("VERCEL_AI_GATEWAY_API_BASE")
                or "https://ai-gateway.vercel.sh/v1"
            )

            api_key = (
                api_key
                or litellm.api_key
                or get_secret_str("VERCEL_AI_GATEWAY_API_KEY")
                or get_secret_str("VERCEL_OIDC_TOKEN")
            )

            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                headers=headers,
            )
        elif custom_llm_provider == "huggingface":
            api_key = api_key or litellm.huggingface_key or get_secret("HUGGINGFACE_API_KEY") or litellm.api_key
            response = huggingface_embed.embedding(
                model=model,
                input=input,
                encoding=_load_module_encoding(),
                api_key=api_key,
                api_base=api_base,
                logging_obj=logging,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                headers=headers,
            )
        elif custom_llm_provider == "bedrock":
            if isinstance(input, str):
                transformed_input = [input]
            else:
                transformed_input = input
            response = bedrock_embedding.embeddings(
                model=model,
                input=transformed_input,
                encoding=_get_encoding(),
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                client=client,
                timeout=timeout,
                aembedding=aembedding,
                litellm_params=litellm_params_dict,
                api_base=api_base,
                print_verbose=print_verbose,
                extra_headers=headers,
                api_key=api_key,
            )
        elif custom_llm_provider == "triton":
            if api_base is None:
                raise ValueError("api_base is required for triton. Please pass `api_base`")
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={},
            )
        elif custom_llm_provider == "gemini":
            gemini_api_key: Final = api_key or get_api_key_from_env() or litellm.api_key

            api_base = api_base or litellm.api_base or get_secret_str("GEMINI_API_BASE")

            response = google_batch_embeddings.batch_embeddings(
                model=model,
                input=input,
                encoding=_get_encoding(),
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                vertex_project=None,
                vertex_location=None,
                vertex_credentials=None,
                aembedding=aembedding,
                print_verbose=print_verbose,
                custom_llm_provider="gemini",
                api_key=gemini_api_key,
                api_base=api_base,
                client=client,
                extra_headers=headers,
                drop_params=drop_unsupported_params,
            )

        elif custom_llm_provider == "vertex_ai":
            vertex_ai_project: Final = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret_str("VERTEXAI_PROJECT")
                or get_secret_str("VERTEX_PROJECT")
            )
            vertex_ai_location: Final = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret_str("VERTEXAI_LOCATION")
                or get_secret_str("VERTEX_LOCATION")
            )
            vertex_credentials: Final = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret_str("VERTEXAI_CREDENTIALS")
                or get_secret_str("VERTEX_CREDENTIALS")
            )

            api_base = (
                api_base or litellm.api_base or get_secret_str("VERTEXAI_API_BASE") or get_secret_str("VERTEX_API_BASE")
            )

            try:
                model_info: Final = get_model_info(model=model, custom_llm_provider="vertex_ai")
                uses_embed_content = model_info.get("uses_embed_content", False)
            except Exception:
                uses_embed_content = False

            if uses_embed_content:
                response = google_batch_embeddings.batch_embeddings(
                    model=model,
                    input=input,
                    encoding=_get_encoding(),
                    logging_obj=logging,
                    optional_params=optional_params,
                    model_response=EmbeddingResponse(),
                    vertex_project=vertex_ai_project,
                    vertex_location=vertex_ai_location,
                    vertex_credentials=vertex_credentials,
                    aembedding=aembedding,
                    print_verbose=print_verbose,
                    custom_llm_provider="vertex_ai",
                    api_key=None,
                    api_base=api_base,
                    client=client,
                    extra_headers=headers,
                    drop_params=drop_unsupported_params,
                )
            elif (
                "image" in optional_params
                or "video" in optional_params
                or model in vertex_multimodal_embedding.SUPPORTED_MULTIMODAL_EMBEDDING_MODELS
            ):
                response = vertex_multimodal_embedding.multimodal_embedding(
                    model=model,
                    input=input,
                    encoding=_get_encoding(),
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params_dict,
                    model_response=EmbeddingResponse(),
                    vertex_project=vertex_ai_project,
                    vertex_location=vertex_ai_location,
                    vertex_credentials=vertex_credentials,
                    aembedding=aembedding,
                    print_verbose=print_verbose,
                    custom_llm_provider="vertex_ai",
                    client=client,
                    api_base=api_base,
                )
            else:
                response = vertex_embedding.embedding(
                    model=model,
                    input=input,
                    encoding=_get_encoding(),
                    logging_obj=logging,
                    optional_params=optional_params,
                    model_response=EmbeddingResponse(),
                    vertex_project=vertex_ai_project,
                    vertex_location=vertex_ai_location,
                    vertex_credentials=vertex_credentials,
                    custom_llm_provider="vertex_ai",
                    timeout=timeout,
                    aembedding=aembedding,
                    print_verbose=print_verbose,
                    api_key=api_key,
                    api_base=api_base,
                    client=client,
                    litellm_params=litellm_params_dict,
                )
        elif custom_llm_provider == "oobabooga":
            response = oobabooga.embedding(
                model=model,
                input=input,
                encoding=_get_encoding(),
                api_base=api_base,
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                api_key=api_key,
            )
        elif custom_llm_provider == "ollama":
            api_base = litellm.api_base or api_base or get_secret_str("OLLAMA_API_BASE") or "http://localhost:11434"

            if isinstance(input, str):
                input = [input]
            if not all(isinstance(item, str) for item in input):
                raise litellm.BadRequestError(
                    message=f"Invalid input for ollama embeddings. input={input}",
                    model=model,
                    llm_provider="ollama",
                )
            ollama_embeddings_fn: Final = ollama.ollama_aembeddings if aembedding is True else ollama.ollama_embeddings
            response = ollama_embeddings_fn(
                api_base=api_base,
                model=model,
                prompts=input,
                encoding=_get_encoding(),
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
            )
        elif custom_llm_provider == "sagemaker":
            response = sagemaker_llm.embedding(
                model=model,
                input=input,
                encoding=_get_encoding(),
                logging_obj=logging,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                print_verbose=print_verbose,
            )
        elif custom_llm_provider == "mistral":
            api_key = api_key or litellm.api_key or get_secret_str("MISTRAL_API_KEY")
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "fireworks_ai":
            api_key = api_key or litellm.api_key or get_secret_str("FIREWORKS_AI_API_KEY")
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "nebius":
            api_key = api_key or litellm.api_key or get_secret_str("NEBIUS_API_KEY")
            api_base = api_base or litellm.api_base or get_secret_str("NEBIUS_API_BASE") or "api.studio.nebius.ai/v1"

            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "wandb":
            api_key = api_key or litellm.api_key or get_secret_str("WANDB_API_KEY")
            api_base = (
                api_base or litellm.api_base or get_secret_str("WANDB_API_BASE") or "https://api.inference.wandb.ai/v1"
            )

            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "sambanova":
            api_key = api_key or litellm.api_key or get_secret_str("SAMBANOVA_API_KEY")
            api_base = (
                api_base or litellm.api_base or get_secret_str("SAMBANOVA_API_BASE") or "https://api.sambanova.ai/v1"
            )
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={},
            )
        elif custom_llm_provider == "voyage" or custom_llm_provider == "infinity":
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={},
            )
        elif custom_llm_provider == "watsonx":
            credentials: Final = IBMWatsonXMixin.get_watsonx_credentials(
                optional_params=optional_params, api_key=api_key, api_base=api_base
            )

            api_key = credentials["api_key"]
            api_base = credentials["api_base"]

            if "token" in credentials:
                optional_params["token"] = credentials["token"]

            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                litellm_params={},
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "xinference":
            api_key = (
                api_key or litellm.api_key or get_secret_str("XINFERENCE_API_KEY") or "stub-xinference-key"
            )  # xinference does not need an api key, pass a stub key if user did not set one
            api_base = (
                api_base or litellm.api_base or get_secret_str("XINFERENCE_API_BASE") or "http://127.0.0.1:9997/v1"
            )
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "sap":
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                litellm_params={},
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "azure_ai":
            from litellm.llms.azure_ai.common_utils import get_azure_ai_entra_token

            api_base = (
                api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
                or litellm.api_base
                or get_secret_str("AZURE_AI_API_BASE")
            )
            # set API KEY
            api_key = (
                api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
                or get_secret_str("AZURE_AI_API_KEY")
                or get_azure_ai_entra_token(litellm_params=litellm_params_dict)
            )

            ## EMBEDDING CALL
            response = azure_ai_embedding.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "jina_ai":
            if isinstance(input, str):
                transformed_input = [input]
            else:
                transformed_input = input
            response = base_llm_http_handler.embedding(
                model=model,
                input=transformed_input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                litellm_params={},
                client=client,
                aembedding=aembedding,
            )
        elif custom_llm_provider == "volcengine":
            volcengine_key: Final = (
                api_key or litellm.api_key or get_secret_str("ARK_API_KEY") or get_secret_str("VOLCENGINE_API_KEY")
            )
            if volcengine_key is None:
                raise ValueError(
                    "Missing API key for Volcengine. Set ARK_API_KEY or VOLCENGINE_API_KEY environment variable or pass api_key parameter."
                )
            if extra_headers is not None and isinstance(extra_headers, dict):
                headers = extra_headers
            else:
                headers = {}
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                timeout=timeout,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging,
                api_base=api_base,
                optional_params=optional_params,
                litellm_params={},
                model_response=EmbeddingResponse(),
                api_key=volcengine_key,
                client=client,
                aembedding=aembedding,
                headers=headers,
            )
        elif custom_llm_provider in ("dashscope", "qwencloud", "qwen_ai_platform"):
            from litellm.llms.dashscope.common_utils import (
                missing_dashscope_family_key_message,
                resolve_dashscope_family_api_key,
            )

            dashscope_key: Final = resolve_dashscope_family_api_key(
                custom_llm_provider=custom_llm_provider,
                api_key=api_key or litellm.api_key,
            )
            if dashscope_key is None:
                raise ValueError(missing_dashscope_family_key_message(custom_llm_provider))
            if extra_headers is not None and isinstance(extra_headers, dict):
                headers = extra_headers
            else:
                headers = {}
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                timeout=timeout,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging,
                api_base=api_base,
                optional_params=optional_params,
                litellm_params={},
                model_response=EmbeddingResponse(),
                api_key=dashscope_key,
                client=client,
                aembedding=aembedding,
                headers=headers,
            )
        elif custom_llm_provider == "ovhcloud":
            api_key = api_key or litellm.api_key or get_secret_str("OVHCLOUD_API_KEY")
            api_base = (
                api_base
                or litellm.api_base
                or get_secret_str("OVHCLOUD_API_BASE")
                or "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
            )
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={},
            )
        elif custom_llm_provider == "cometapi":
            api_key = api_key or litellm.cometapi_key or get_secret_str("COMETAPI_KEY") or litellm.api_key
            api_base = (
                api_base or litellm.api_base or get_secret_str("COMETAPI_API_BASE") or "https://api.cometapi.com/v1"
            )
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={},
            )
        elif custom_llm_provider in litellm._custom_providers:
            custom_handler: CustomLLM | None = None
            for item in litellm.custom_provider_map:
                if item["provider"] == custom_llm_provider:
                    custom_handler = item["custom_handler"]

            if custom_handler is None:
                raise LiteLLMUnknownProvider(model=model, custom_llm_provider=custom_llm_provider)

            handler_fn: Final = custom_handler.embedding if not aembedding else custom_handler.aembedding

            response = handler_fn(
                model=model,
                input=input,
                logging_obj=logging,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                optional_params=optional_params,
                model_response=EmbeddingResponse(),
                print_verbose=print_verbose,
                litellm_params=litellm_params_dict,
            )
        elif custom_llm_provider == "snowflake":
            api_key = api_key or get_secret_str("SNOWFLAKE_JWT")
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={},
            )
        elif custom_llm_provider == "gigachat":
            api_key = (
                api_key
                or litellm.api_key
                or litellm.gigachat_key
                or get_secret_str("GIGACHAT_CREDENTIALS")
                or get_secret_str("GIGACHAT_API_KEY")
            )
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={"ssl_verify": kwargs.get("ssl_verify", None)},
            )
        elif custom_llm_provider == "perplexity":
            response = base_llm_http_handler.embedding(
                model=model,
                input=input,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                timeout=timeout,
                model_response=EmbeddingResponse(),
                optional_params=optional_params,
                client=client,
                aembedding=aembedding,
                litellm_params={},
            )
        else:
            raise LiteLLMUnknownProvider(model=model, custom_llm_provider=custom_llm_provider)
        if response is not None and hasattr(response, "_hidden_params") and isinstance(response, EmbeddingResponse):
            response_hidden_params: Final = get_hidden_params(response)
            if response_hidden_params is not None:
                response_hidden_params["custom_llm_provider"] = custom_llm_provider

        if response is None:
            raise LiteLLMUnknownProvider(model=model, custom_llm_provider=custom_llm_provider)
        return response
    except Exception as e:
        ## LOGGING
        litellm_logging_obj.post_call(
            input=input,
            api_key=api_key,
            original_response=str(e),
        )
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            original_exception=e,
            custom_llm_provider=custom_llm_provider,
            extra_kwargs=kwargs,
        )


###### Text Completion ################
async def _resolve_dispatched_text_completion_response(
    pending: Coroutine[
        object,
        object,
        TextCompletionResponse | ModelResponse | CustomStreamWrapper | TextCompletionStreamWrapper,
    ],
) -> TextCompletionResponse | ModelResponse | CustomStreamWrapper | TextCompletionStreamWrapper:
    return await pending


async def _resolve_pending_chat_response(pending: Coroutine[object, object, ModelResponse]) -> ModelResponse:
    return await pending


@client
async def atext_completion(*args, **kwargs) -> TextCompletionResponse | TextCompletionStreamWrapper:
    """
    Implemented to handle async streaming for the text completion endpoint
    """
    loop: Final = asyncio.get_event_loop()
    model: Final = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO COMPLETION ###
    kwargs["acompletion"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func: Final = partial(text_completion, *args, **kwargs)

        # Add the context to the function
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)

        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if isinstance(init_response, dict) or isinstance(init_response, TextCompletionResponse):  ## CACHING SCENARIO
            if isinstance(init_response, dict):
                response = TextCompletionResponse(**init_response)
            else:
                response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await _resolve_dispatched_text_completion_response(init_response)
        else:
            response = init_response

        if (
            kwargs.get("stream", False) is True
            or isinstance(response, TextCompletionStreamWrapper)
            or isinstance(response, CustomStreamWrapper)
        ):  # return an async generator
            return TextCompletionStreamWrapper(
                completion_stream=_async_streaming(
                    response=response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    args=args,
                ),
                model=model,
                custom_llm_provider=custom_llm_provider,
                stream_options=kwargs.get("stream_options"),
            )
        else:
            ## OpenAI / Azure Text Completion Returns here
            if isinstance(response, TextCompletionResponse):
                return response
            elif asyncio.iscoroutine(response):
                response = await _resolve_pending_chat_response(response)

            text_completion_response = TextCompletionResponse()
            text_completion_response = litellm.utils.LiteLLMResponseObjectHandler.convert_chat_to_text_completion(
                text_completion_response=text_completion_response,
                response=response,
                custom_llm_provider=custom_llm_provider,
            )
            return text_completion_response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def text_completion(
    prompt: str | list[str | list[str | list[int]]],  # Required: The prompt(s) to generate completions for.
    model: str | None = None,  # Optional: either `model` or `engine` can be set
    best_of: int | None = None,  # Optional: Generates best_of completions server-side.
    echo: bool | None = None,  # Optional: Echo back the prompt in addition to the completion.
    frequency_penalty: float | None = None,  # Optional: Penalize new tokens based on their existing frequency.
    logit_bias: dict[int, int] | None = None,  # Optional: Modify the likelihood of specified tokens.
    logprobs: int | None = None,  # Optional: Include the log probabilities on the most likely tokens.
    max_tokens: int | None = None,  # Optional: The maximum number of tokens to generate in the completion.
    n: int | None = None,  # Optional: How many completions to generate for each prompt.
    presence_penalty: float
    | None = None,  # Optional: Penalize new tokens based on whether they appear in the text so far.
    stop: str | list[str] | None = None,  # Optional: Sequences where the API will stop generating further tokens.
    stream: bool | None = None,  # Optional: Whether to stream back partial progress.
    stream_options: dict | None = None,
    suffix: str | None = None,  # Optional: The suffix that comes after a completion of inserted text.
    temperature: float | None = None,  # Optional: Sampling temperature to use.
    top_p: float | None = None,  # Optional: Nucleus sampling parameter.
    user: str | None = None,  # Optional: A unique identifier representing your end-user.
    # set api_base, api_version, api_key
    api_base: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    custom_llm_provider: str | None = None,
    *args,
    **kwargs,
):
    import copy

    """
    Generate text completions using the OpenAI API.

    Args:
        model (str): ID of the model to use.
        prompt (Union[str, List[Union[str, List[Union[str, List[int]]]]]): The prompt(s) to generate completions for.
        best_of (Optional[int], optional): Generates best_of completions server-side. Defaults to 1.
        echo (Optional[bool], optional): Echo back the prompt in addition to the completion. Defaults to False.
        frequency_penalty (Optional[float], optional): Penalize new tokens based on their existing frequency. Defaults to 0.
        logit_bias (Optional[Dict[int, int]], optional): Modify the likelihood of specified tokens. Defaults to None.
        logprobs (Optional[int], optional): Include the log probabilities on the most likely tokens. Defaults to None.
        max_tokens (Optional[int], optional): The maximum number of tokens to generate in the completion. Defaults to 16.
        n (Optional[int], optional): How many completions to generate for each prompt. Defaults to 1.
        presence_penalty (Optional[float], optional): Penalize new tokens based on whether they appear in the text so far. Defaults to 0.
        stop (Optional[Union[str, List[str]]], optional): Sequences where the API will stop generating further tokens. Defaults to None.
        stream (Optional[bool], optional): Whether to stream back partial progress. Defaults to False.
        suffix (Optional[str], optional): The suffix that comes after a completion of inserted text. Defaults to None.
        temperature (Optional[float], optional): Sampling temperature to use. Defaults to 1.
        top_p (Optional[float], optional): Nucleus sampling parameter. Defaults to 1.
        user (Optional[str], optional): A unique identifier representing your end-user.
    Returns:
        TextCompletionResponse: A response object containing the generated completion and associated metadata.

    Example:
        Your example of how to use this function goes here.
    """
    if "engine" in kwargs:
        _engine: Final = kwargs["engine"]
        if model is None and isinstance(_engine, str):
            # only use engine when model not passed
            model = _engine
        kwargs.pop("engine")

    text_completion_response = TextCompletionResponse()

    optional_params: Final[dict[str, Any]] = {}
    # default values for all optional params are none, litellm only passes them to the llm when they are set to non None values
    if best_of is not None:
        optional_params["best_of"] = best_of
    if echo is not None:
        optional_params["echo"] = echo
    if frequency_penalty is not None:
        optional_params["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        optional_params["logit_bias"] = logit_bias
    if logprobs is not None:
        optional_params["logprobs"] = logprobs
    if max_tokens is not None:
        optional_params["max_tokens"] = max_tokens
    if n is not None:
        optional_params["n"] = n
    if presence_penalty is not None:
        optional_params["presence_penalty"] = presence_penalty
    if stop is not None:
        optional_params["stop"] = stop
    if stream is not None:
        optional_params["stream"] = stream
    if stream_options is not None:
        optional_params["stream_options"] = stream_options
    if suffix is not None:
        optional_params["suffix"] = suffix
    if temperature is not None:
        optional_params["temperature"] = temperature
    if top_p is not None:
        optional_params["top_p"] = top_p
    if user is not None:
        optional_params["user"] = user
    if api_base is not None:
        optional_params["api_base"] = api_base
    if api_version is not None:
        optional_params["api_version"] = api_version
    if api_key is not None:
        optional_params["api_key"] = api_key
    if custom_llm_provider is not None:
        optional_params["custom_llm_provider"] = custom_llm_provider

    # get custom_llm_provider
    _model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
    )

    if custom_llm_provider == "huggingface":
        # if echo == True, for TGI llms we need to set top_n_tokens to 3
        if echo is True:
            # for tgi llms
            if "top_n_tokens" not in kwargs:
                kwargs["top_n_tokens"] = 3

        # processing prompt - users can pass raw tokens to OpenAI Completion()
        if isinstance(prompt, list):
            import concurrent.futures

            from litellm.rust_bridge.tokenizer import get_encoding

            tokenizer: Final = get_encoding("p50k_base")
            ## if it's a 2d list - each element in the list is a text_completion() request
            if len(prompt) > 0 and isinstance(prompt[0], list):
                responses: Final = [None for x in prompt]  # init responses

                def process_prompt(i, individual_prompt):
                    decoded_prompt: Final = tokenizer.decode(individual_prompt)
                    all_params: Final = {**kwargs, **optional_params}
                    response: Final[TextCompletionResponse] = text_completion(
                        model=model,
                        prompt=decoded_prompt,
                        num_retries=3,  # ensure this does not fail for the batch
                        *args,
                        **all_params,
                    )

                    text_completion_response["id"] = response.get("id", None)
                    text_completion_response["object"] = "text_completion"
                    text_completion_response["created"] = response.get("created", None)
                    text_completion_response["model"] = response.get("model", None)
                    return response["choices"][0]

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    completed_futures: Final = [
                        executor.submit(process_prompt, i, individual_prompt)
                        for i, individual_prompt in enumerate(prompt)
                    ]
                    for i, future in enumerate(concurrent.futures.as_completed(completed_futures)):
                        responses[i] = future.result()
                    text_completion_response.choices = responses

                return text_completion_response
    # else:
    # check if non default values passed in for best_of, echo, logprobs, suffix
    # these are the params supported by Completion() but not ChatCompletion

    # default case, non OpenAI requests go through here
    # handle prompt formatting if prompt is a string vs. list of strings
    messages = []
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], str):
        for p in prompt:
            message = {"role": "user", "content": p}
            messages.append(message)
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif (
        (
            custom_llm_provider == "openai"
            or custom_llm_provider == "azure"
            or custom_llm_provider == "azure_text"
            or custom_llm_provider == "text-completion-codestral"
            or custom_llm_provider == "text-completion-openai"
        )
        and isinstance(prompt, list)
        and len(prompt) > 0
        and (isinstance(prompt[0], list) or isinstance(prompt[0], int))
    ):
        # Support for token IDs as prompt (list of integers or list of lists of integers)
        messages = [{"role": "user", "content": prompt}]
    else:
        raise Exception(
            f"Unmapped prompt format. Your prompt is neither a list of strings nor a string. prompt={prompt}. File an issue - https://github.com/BerriAI/litellm/issues"
        )

    kwargs.pop("prompt", None)

    if (
        _model is not None and (custom_llm_provider == "openai")
    ):  # for openai compatible endpoints - e.g. vllm, call the native /v1/completions endpoint for text completion calls
        if _model not in litellm.open_ai_chat_completion_models:
            model = "text-completion-openai/" + _model
            optional_params.pop("custom_llm_provider", None)

    if model is None:
        raise ValueError("model is not set. Set either via 'model' or 'engine' param.")
    kwargs["text_completion"] = True
    response = completion(
        model=model,
        messages=messages,
        *args,
        **kwargs,
        **optional_params,
    )
    if kwargs.get("acompletion", False) is True:
        return response
    if stream is True or kwargs.get("stream", False) is True or isinstance(response, CustomStreamWrapper):
        response = TextCompletionStreamWrapper(
            completion_stream=response,
            model=model,
            stream_options=stream_options,
            custom_llm_provider=custom_llm_provider,
        )
        return response
    elif isinstance(response, TextCompletionStreamWrapper):
        return response

    # OpenAI Text / Azure Text will return here
    if isinstance(response, TextCompletionResponse):
        return response

    text_completion_response = litellm.utils.LiteLLMResponseObjectHandler.convert_chat_to_text_completion(
        response=response,
        text_completion_response=text_completion_response,
    )

    return text_completion_response


###### Adapter Completion ################


async def aadapter_completion(*, adapter_id: str, **kwargs) -> BaseModel | AdapterCompletionStreamWrapper | None:
    """
    Implemented to handle async calls for adapter_completion()
    """
    try:
        translation_obj: CustomLogger | None = None
        for item in litellm.adapters:
            if item["id"] == adapter_id:
                translation_obj = item["adapter"]

        if translation_obj is None:
            raise ValueError(
                f"No matching adapter given. Received 'adapter_id'={adapter_id}, litellm.adapters={litellm.adapters}"
            )

        new_kwargs: Final = translation_obj.translate_completion_input_params(kwargs=kwargs)

        response: Final[ModelResponse | CustomStreamWrapper] = await acompletion(**new_kwargs)
        translated_response: BaseModel | AdapterCompletionStreamWrapper | None = None
        if isinstance(response, ModelResponse):
            translated_response = translation_obj.translate_completion_output_params(response=response)
        if isinstance(response, CustomStreamWrapper):
            translated_response = translation_obj.translate_completion_output_params_streaming(
                completion_stream=response
            )

        return translated_response
    except Exception as e:
        raise e


async def aadapter_generate_content(
    **kwargs,
) -> dict[str, object] | AsyncIterator[bytes]:
    from litellm.google_genai.adapters.handler import GenerateContentToCompletionHandler

    coro: Final = cast(
        Coroutine[object, object, dict[str, object] | AsyncIterator[bytes]],
        GenerateContentToCompletionHandler.generate_content_handler(**kwargs, _is_async=True),
    )
    return await coro


def adapter_completion(*, adapter_id: str, **kwargs) -> BaseModel | AdapterCompletionStreamWrapper | None:
    translation_obj: CustomLogger | None = None
    for item in litellm.adapters:
        if item["id"] == adapter_id:
            translation_obj = item["adapter"]

    if translation_obj is None:
        raise ValueError(
            f"No matching adapter given. Received 'adapter_id'={adapter_id}, litellm.adapters={litellm.adapters}"
        )

    new_kwargs: Final = translation_obj.translate_completion_input_params(kwargs=kwargs)

    response: Final[ModelResponse | CustomStreamWrapper] = completion(**new_kwargs)
    translated_response: BaseModel | AdapterCompletionStreamWrapper | None = None
    if isinstance(response, ModelResponse):
        translated_response = translation_obj.translate_completion_output_params(response=response)
    elif isinstance(response, CustomStreamWrapper) or inspect.isgenerator(response):
        translated_response = translation_obj.translate_completion_output_params_streaming(completion_stream=response)

    return translated_response


##### Moderation #######################


def moderation(input: str, model: str | None = None, api_key: str | None = None, **kwargs) -> OpenAIModerationResponse:
    static_api_key: Final = api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")
    api_base: Final = kwargs.get("api_base", None)

    openai_client: Final = kwargs.get("client", None) or build_openai_client(
        api_key=static_api_key,
        api_base=api_base,
        litellm_params=kwargs,
    )

    if model is not None:
        response = openai_client.moderations.create(input=input, model=model)
    else:
        response = openai_client.moderations.create(input=input)

    response_dict: Final[dict] = response.model_dump()
    return litellm.utils.LiteLLMResponseObjectHandler.convert_to_moderation_response(
        response_object=response_dict,
    )


@client
async def amoderation(
    input: str,
    model: str | None = None,
    api_key: str | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> OpenAIModerationResponse:
    from openai import AsyncOpenAI

    static_api_key: Final = api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")
    optional_params: Final = GenericLiteLLMParams.model_validate(kwargs)
    litellm_logging_obj: Final[LiteLLMLoggingObj | None] = kwargs.get("litellm_logging_obj", None)
    _dynamic_api_base = None
    try:
        (
            model,
            custom_llm_provider,
            _dynamic_api_key,
            _dynamic_api_base,
        ) = litellm.get_llm_provider(
            model=model or "",
            custom_llm_provider=custom_llm_provider,
            api_base=optional_params.api_base,
            api_key=optional_params.api_key,
        )
    except litellm.BadRequestError:
        # `model` is optional field for moderation - get_llm_provider will throw BadRequestError if model is not set / not recognized
        pass

    openai_client: Final = kwargs.get("client", None)
    if openai_client is None or not isinstance(openai_client, AsyncOpenAI):
        # call helper to get OpenAI client
        # _get_openai_client maintains in-memory caching logic for OpenAI clients
        _openai_client: AsyncOpenAI = openai_chat_completions.get_openai_client(
            is_async=True,
            api_key=static_api_key,
            api_base=optional_params.api_base or _dynamic_api_base,
            litellm_params=optional_params.model_dump(exclude_none=True),
        )
    else:
        _openai_client = openai_client

    # update litellm_logging_obj with environment variables
    custom_llm_provider = custom_llm_provider or litellm.LlmProviders.OPENAI.value
    if litellm_logging_obj is not None:
        litellm_logging_obj.update_environment_variables(
            model=model,
            user=kwargs.get("user", None),
            optional_params={},
            litellm_params={
                **kwargs,
            },
            custom_llm_provider=custom_llm_provider,
        )
        moderation_request: Final = {"input": input, "model": model}
        litellm_logging_obj.pre_call(
            input=input,
            api_key=static_api_key,
            additional_args={
                "complete_input_dict": moderation_request,
                "api_base": str(_openai_client.base_url),
            },
        )

    if model is not None:
        response = await _openai_client.moderations.create(input=input, model=model)
    else:
        response = await _openai_client.moderations.create(input=input)
    response_dict: Final[dict] = response.model_dump()
    return litellm.utils.LiteLLMResponseObjectHandler.convert_to_moderation_response(
        response_object=response_dict,
    )


##### Transcription #######################


@client
async def atranscription(*args, **kwargs) -> TranscriptionResponse:
    """
    Calls openai + azure whisper endpoints.

    Allows router to load balance between them
    """
    loop: Final = asyncio.get_event_loop()
    model: Final = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Image Generation ###
    kwargs["atranscription"] = True
    file: Final = kwargs.get("file", None)
    try:
        provider_hints: Final = _PROVIDER_ROUTING_HINTS.validate_python(kwargs)
        api_base_hint: Final = provider_hints.get("api_base") or _BASE_URL_HINT.validate_python(kwargs).get("base_url")
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=invalid_provider_routing_hints_message(error), model=model, llm_provider=""
        ) from error
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func: Final = partial(transcription, *args, **kwargs)

        # Add the context to the function
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=provider_hints.get("custom_llm_provider"),
            api_base=api_base_hint,
        )

        # Await normally
        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if isinstance(init_response, dict):
            response = _transcription_response_from_cached_dict(init_response)
        elif isinstance(init_response, TranscriptionResponse):  ## CACHING SCENARIO
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)
        if not isinstance(response, TranscriptionResponse):
            raise ValueError(
                f"Invalid response from transcription provider, expected TranscriptionResponse, but got {type(response)}"
            )

        # Store duration in _hidden_params for cost calculation without
        # exposing it in the response body. Adding duration to the response
        # tricks the OpenAI SDK's "best match deserialization" into thinking
        # a plain Transcription is a TranscriptionVerbose/Diarized type.
        if response is not None and not isinstance(response, Coroutine) and file is not None:
            existing_duration: Final = getattr(response, "duration", None)
            if existing_duration is None:
                calculated_duration: Final = calculate_request_duration(file)
                if calculated_duration is not None:
                    response.hidden_params["audio_transcription_duration"] = calculated_duration

        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def transcription(
    model: str,
    file: FileTypes,
    ## OPTIONAL OPENAI PARAMS ##
    language: str | None = None,
    prompt: str | None = None,
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] | None = None,
    timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    temperature: int | None = None,  # openai defaults this to 0
    ## LITELLM PARAMS ##
    user: str | None = None,
    timeout=600,  # default to 10 minutes
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
    max_retries: int | None = None,
    custom_llm_provider=None,
    base_url: str | None = None,
    **kwargs,
) -> TranscriptionResponse | Coroutine[object, object, TranscriptionResponse]:
    """
    Calls openai + azure whisper endpoints.

    Allows router to load balance between them
    """
    litellm_call_id: Final = kwargs.get("litellm_call_id", None)
    proxy_server_request: Final = kwargs.get("proxy_server_request", None)
    model_info: Final = kwargs.get("model_info", None)
    metadata: Final = kwargs.get("metadata", None)
    atranscription: Final = kwargs.pop("atranscription", False)
    litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
    extra_headers: Final = kwargs.get("extra_headers", None)
    shared_session: Final = kwargs.get("shared_session", None)
    kwargs.pop("tags", [])
    non_default_params: Final = get_non_default_transcription_params(kwargs)

    client: openai.AsyncOpenAI | openai.OpenAI | openai.AzureOpenAI | openai.AsyncAzureOpenAI | None = kwargs.pop(
        "client", None
    )

    if litellm_logging_obj:
        litellm_logging_obj.model_call_details["client"] = str(client)

    if max_retries is None:
        max_retries = openai.DEFAULT_MAX_RETRIES

    model_response: Final = litellm.utils.TranscriptionResponse()

    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base or base_url,
        api_key=api_key,
    )

    api_key = dynamic_api_key if dynamic_api_key is not None else api_key

    optional_params: Final = get_optional_params_transcription(
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
        temperature=temperature,
        custom_llm_provider=custom_llm_provider,
        **non_default_params,
    )

    litellm_params_dict: Final = get_litellm_params(**kwargs)

    litellm_logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params={},
        litellm_params={
            "litellm_call_id": litellm_call_id,
            "proxy_server_request": proxy_server_request,
            "model_info": model_info,
            "metadata": metadata,
            "preset_cache_key": None,
            "stream_response": {},
            **kwargs,
        },
        custom_llm_provider=custom_llm_provider,
    )

    response: TranscriptionResponse | Coroutine[object, object, TranscriptionResponse] | None = None

    provider_config: Final = ProviderConfigManager.get_provider_audio_transcription_config(
        model=model,
        provider=LlmProviders(custom_llm_provider),
    )

    uses_openai_transport: Final = custom_llm_provider in OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS and not (
        provider_config is not None and provider_config.has_native_transcription_endpoint
    )

    if custom_llm_provider in AZURE_OPENAI_AUDIO_PROVIDERS and provider_config is None:
        # azure configs
        api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")

        api_version = api_version or litellm.api_version or get_secret_str("AZURE_API_VERSION")

        azure_ad_token: Final = kwargs.pop("azure_ad_token", None) or get_secret_str("AZURE_AD_TOKEN")

        api_key = api_key or litellm.api_key or litellm.azure_key or get_secret_str("AZURE_API_KEY")

        optional_params["extra_headers"] = extra_headers

        response = azure_audio_transcriptions.audio_transcriptions(
            model=model,
            audio_file=file,
            optional_params=optional_params,
            model_response=model_response,
            atranscription=atranscription,
            client=client,
            timeout=timeout,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            max_retries=max_retries,
            litellm_params=litellm_params_dict,
            custom_llm_provider=custom_llm_provider,
        )
    elif uses_openai_transport:
        api_base = (
            api_base
            or litellm.api_base
            or get_secret("OPENAI_BASE_URL")
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        openai.organization = (
            litellm.organization
            or get_secret("OPENAI_ORGANIZATION")
            or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
        )
        static_api_key: Final = api_key or litellm.api_key or litellm.openai_key or get_secret("OPENAI_API_KEY")
        response = openai_audio_transcriptions.audio_transcriptions(
            model=model,
            audio_file=file,
            optional_params=optional_params,
            model_response=model_response,
            atranscription=atranscription,
            client=client,
            timeout=timeout,
            logging_obj=litellm_logging_obj,
            max_retries=max_retries,
            api_base=api_base,
            api_key=static_api_key,
            provider_config=provider_config,
            litellm_params=litellm_params_dict,
            shared_session=shared_session,
        )
    elif custom_llm_provider == "nvidia_riva":
        # NVIDIA Riva is gRPC-based, not HTTP. It has its own dedicated handler
        # rather than going through base_llm_http_handler.
        response = nvidia_riva_audio_transcriptions.audio_transcriptions(
            model=model,
            audio_file=file,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            model_response=model_response,
            atranscription=atranscription,
            timeout=timeout,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            provider_config=(
                provider_config if isinstance(provider_config, NvidiaRivaAudioTranscriptionConfig) else None
            ),
        )
    elif custom_llm_provider == "soniox":
        from litellm.llms.soniox.audio_transcription.handler import (
            SonioxAudioTranscriptionHandler,
        )

        response = SonioxAudioTranscriptionHandler().audio_transcriptions(
            model=model,
            audio_file=file,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            model_response=model_response,
            atranscription=atranscription,
            client=(
                client
                if client is not None and (isinstance(client, HTTPHandler) or isinstance(client, AsyncHTTPHandler))
                else None
            ),
            timeout=timeout,
            max_retries=max_retries,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            headers=extra_headers,
            provider_config=provider_config,
        )
    elif custom_llm_provider == "bedrock":
        from litellm.llms.bedrock.audio_transcription import BedrockAudioTranscriptionRustDispatch

        dispatch: Final = BedrockAudioTranscriptionRustDispatch()
        if atranscription:
            response = dispatch.async_audio_transcriptions(
                model=model,
                audio_file=file,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                optional_params=optional_params,
                timeout=timeout,
            )
        else:
            response = dispatch.audio_transcriptions(
                model=model,
                audio_file=file,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                optional_params=optional_params,
                timeout=timeout,
            )
    elif provider_config is not None:
        response = base_llm_http_handler.audio_transcriptions(
            model=model,
            audio_file=file,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            model_response=model_response,
            atranscription=atranscription,
            client=(
                client
                if client is not None and (isinstance(client, HTTPHandler) or isinstance(client, AsyncHTTPHandler))
                else None
            ),
            timeout=timeout,
            max_retries=max_retries,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            headers={},
            provider_config=provider_config,
            shared_session=shared_session,
        )

    # Store duration in _hidden_params for cost calculation without
    # exposing it in the response body (see sync path comment above).
    if response is not None and not isinstance(response, Coroutine):
        existing_duration: Final = getattr(response, "duration", None)
        if existing_duration is None:
            calculated_duration: Final = calculate_request_duration(file)
            if calculated_duration is not None:
                response_hidden_params: Final = cast(  # cast-ok: preserve dynamic mapping behavior
                    dict[str, object], getattr(response, HIDDEN_PARAMS_ATTR)
                )
                response_hidden_params["audio_transcription_duration"] = calculated_duration

    if response is None:
        raise ValueError("Unmapped provider passed in. Unable to get the response.")
    return response


@client
async def aspeech(*args, **kwargs) -> HttpxBinaryResponseContent:
    """
    Calls openai tts endpoints.
    """
    loop: Final = asyncio.get_event_loop()
    model: Final = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Image Generation ###
    kwargs["aspeech"] = True
    try:
        provider_hints: Final = _PROVIDER_ROUTING_HINTS.validate_python(kwargs)
        api_base_hint: Final = provider_hints.get("api_base") or _BASE_URL_HINT.validate_python(kwargs).get("base_url")
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=invalid_provider_routing_hints_message(error), model=model, llm_provider=""
        ) from error
    custom_llm_provider = provider_hints.get("custom_llm_provider")
    try:
        # Use a partial function to pass your keyword arguments
        func: Final = partial(speech, *args, **kwargs)

        # Add the context to the function
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=provider_hints.get("custom_llm_provider"),
            api_base=api_base_hint,
        )

        init_response: Final = await loop.run_in_executor(None, func_with_context)
        return await init_response if asyncio.iscoroutine(init_response) else init_response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def speech(
    model: str,
    input: str,
    voice: str | dict | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
    organization: str | None = None,
    project: str | None = None,
    max_retries: int | None = None,
    metadata: dict | None = None,
    timeout: float | httpx.Timeout | openai.Timeout | None = None,
    response_format: str | None = None,
    speed: int | None = None,
    instructions: str | None = None,
    client=None,
    headers: dict | None = None,
    custom_llm_provider: str | None = None,
    aspeech: bool | None = None,
    base_url: str | None = None,
    **kwargs,
) -> HttpxBinaryResponseContent | Coroutine[object, object, HttpxBinaryResponseContent]:
    user: Final = kwargs.get("user", None)
    litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
    proxy_server_request: Final = kwargs.get("proxy_server_request", None)
    extra_headers: Final = kwargs.get("extra_headers", None)
    model_info: Final = kwargs.get("model_info", None)
    shared_session: Final = kwargs.get("shared_session", None)
    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model=model, custom_llm_provider=custom_llm_provider, api_base=api_base or base_url
    )
    kwargs.pop("tags", [])

    optional_params = {}
    if response_format is not None:
        optional_params["response_format"] = response_format
    if speed is not None:
        optional_params["speed"] = speed
    if instructions is not None:
        optional_params["instructions"] = instructions

    timeout_or_default: Final = litellm.request_timeout if timeout is None else timeout
    http_timeout: Final = (
        CompletionTimeout.normalize(timeout_or_default)
        if isinstance(timeout_or_default, openai.Timeout)
        else timeout_or_default
    )
    if max_retries is None:
        max_retries = litellm.num_retries or openai.DEFAULT_MAX_RETRIES
    litellm_params_dict: Final = get_litellm_params(metadata=metadata, api_key=api_key or dynamic_api_key, **kwargs)

    # Get provider-specific text-to-speech config and map parameters
    text_to_speech_provider_config = ProviderConfigManager.get_provider_text_to_speech_config(
        model=model,
        provider=litellm.LlmProviders(custom_llm_provider),
    )

    # Map OpenAI params to provider-specific params if config exists
    if text_to_speech_provider_config is not None:
        voice, optional_params = text_to_speech_provider_config.map_openai_params(
            model=model,
            optional_params=optional_params,
            voice=voice,
            drop_params=False,
            kwargs=kwargs,
        )

    logging_obj: Final[LiteLLMLoggingObj] = cast(LiteLLMLoggingObj, kwargs.get("litellm_logging_obj"))
    logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params=optional_params,
        litellm_params={
            "litellm_call_id": litellm_call_id,
            "proxy_server_request": proxy_server_request,
            "model_info": model_info,
            "metadata": metadata,
            "preset_cache_key": None,
            "stream_response": {},
            **kwargs,
        },
        custom_llm_provider=custom_llm_provider,
    )
    response: HttpxBinaryResponseContent | Coroutine[object, object, HttpxBinaryResponseContent] | None = None
    if custom_llm_provider == "edenai":
        litellm_params_dict["api_base"] = api_base
        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice if isinstance(voice, str) else None,
            text_to_speech_provider_config=text_to_speech_provider_config or litellm.EdenAITextToSpeechConfig(),
            text_to_speech_optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=http_timeout,
            extra_headers=extra_headers,
            client=client,
            _is_async=aspeech or False,
        )
    elif custom_llm_provider == "openai" or (
        custom_llm_provider in litellm.openai_compatible_providers
        and custom_llm_provider not in AZURE_OPENAI_AUDIO_PROVIDERS
    ):
        if voice is None or not (isinstance(voice, str)):
            raise litellm.BadRequestError(
                message="'voice' is required to be passed as a string for OpenAI TTS",
                model=model,
                llm_provider=custom_llm_provider,
            )
        api_base = (
            api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
            or litellm.api_base
            or get_secret("OPENAI_BASE_URL")
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        # set API KEY
        api_key = (
            api_key
            or litellm.api_key  # for deepinfra/perplexity/anyscale we check in get_llm_provider and pass in the api key from there
            or litellm.openai_key
            or get_secret("OPENAI_API_KEY")
        )

        organization = (
            organization
            or litellm.organization
            or get_secret("OPENAI_ORGANIZATION")
            or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
        )

        project = (
            project
            or litellm.project
            or get_secret("OPENAI_PROJECT")
            or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
        )

        headers = headers or litellm.headers

        response = openai_chat_completions.audio_speech(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            organization=organization,
            project=project,
            max_retries=max_retries,
            timeout=http_timeout,
            logging_obj=logging_obj,
            client=client,  # pass AsyncOpenAI, OpenAI client
            aspeech=aspeech,
            shared_session=shared_session,
            litellm_params=litellm_params_dict,
        )
    elif custom_llm_provider in AZURE_OPENAI_AUDIO_PROVIDERS:
        # Check if this is Azure Speech Service (Cognitive Services TTS)
        if model.startswith("speech/"):
            from litellm.llms.azure.text_to_speech.transformation import (
                AzureAVATextToSpeechConfig,
            )

            # Azure AVA (Cognitive Services) Text-to-Speech
            if text_to_speech_provider_config is None:
                raise litellm.BadRequestError(
                    message="Azure Speech Service configuration not found",
                    model=model,
                    llm_provider=custom_llm_provider,
                )

            # Cast to specific Azure config type to access dispatch method
            azure_config: Final = cast(AzureAVATextToSpeechConfig, text_to_speech_provider_config)

            response = azure_config.dispatch_text_to_speech(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                litellm_params_dict=litellm_params_dict,
                logging_obj=logging_obj,
                timeout=http_timeout,
                extra_headers=extra_headers,
                base_llm_http_handler=base_llm_http_handler,
                aspeech=aspeech or False,
                api_base=api_base,
                api_key=api_key,
                **kwargs,
            )
        else:
            # Azure OpenAI TTS
            if voice is None or not (isinstance(voice, str)):
                raise litellm.BadRequestError(
                    message="'voice' is required to be passed as a string for Azure TTS",
                    model=model,
                    llm_provider=custom_llm_provider,
                )
            api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")

            api_version = api_version or litellm.api_version or get_secret("AZURE_API_VERSION")

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )

            azure_ad_token: Final[str | None] = optional_params.get("extra_body", {}).pop(
                "azure_ad_token", None
            ) or get_secret("AZURE_AD_TOKEN")
            azure_ad_token_provider: Final = kwargs.get("azure_ad_token_provider", None)

            if extra_headers:
                optional_params["extra_headers"] = extra_headers

            response = azure_chat_completions.audio_speech(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                azure_ad_token_provider=azure_ad_token_provider,
                organization=organization,
                max_retries=max_retries,
                timeout=http_timeout,
                logging_obj=logging_obj,
                client=client,  # pass AsyncOpenAI, OpenAI client
                aspeech=aspeech,
                litellm_params=litellm_params_dict,
            )
    elif custom_llm_provider == "elevenlabs":
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        if text_to_speech_provider_config is None:
            text_to_speech_provider_config = ElevenLabsTextToSpeechConfig()

        elevenlabs_config: Final = cast(ElevenLabsTextToSpeechConfig, text_to_speech_provider_config)

        voice_id = voice if isinstance(voice, str) else None
        if voice_id is None or not voice_id.strip():
            raise litellm.BadRequestError(
                message="'voice' must resolve to an ElevenLabs voice id for ElevenLabs TTS",
                model=model,
                llm_provider=custom_llm_provider,
            )
        voice_id = voice_id.strip()

        query_params: Final = kwargs.pop(ElevenLabsTextToSpeechConfig.ELEVENLABS_QUERY_PARAMS_KEY, None)
        if isinstance(query_params, dict):
            litellm_params_dict[ElevenLabsTextToSpeechConfig.ELEVENLABS_QUERY_PARAMS_KEY] = query_params

        litellm_params_dict[ElevenLabsTextToSpeechConfig.ELEVENLABS_VOICE_ID_KEY] = voice_id

        if api_base is not None:
            litellm_params_dict["api_base"] = api_base
        if api_key is not None:
            litellm_params_dict["api_key"] = api_key

        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_id,
            text_to_speech_provider_config=elevenlabs_config,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=http_timeout,
            extra_headers=extra_headers,
            client=client,
            _is_async=aspeech or False,
        )
    elif custom_llm_provider == "vertex_ai" or custom_llm_provider == "vertex_ai_beta":
        from litellm.llms.vertex_ai.text_to_speech.transformation import (
            VertexAILyriaTextToSpeechConfig,
            VertexAITextToSpeechConfig,
        )

        generic_optional_params: Final = GenericLiteLLMParams.model_validate(kwargs)

        # Handle Gemini models separately (they use speech_to_completion_bridge)
        if "gemini" in model:
            from .endpoints.speech.speech_to_completion_bridge.handler import (
                speech_to_completion_bridge_handler,
            )

            return speech_to_completion_bridge_handler.speech(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                litellm_params=litellm_params_dict,
                headers=headers or {},
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
            )

        # Vertex AI Text-to-Speech (Google Cloud TTS)
        if text_to_speech_provider_config is None:
            text_to_speech_provider_config = (  # rebind-ok: model metadata selects the Vertex TTS implementation
                VertexAILyriaTextToSpeechConfig()
                if VertexAILyriaTextToSpeechConfig.is_lyria_model(model)
                else VertexAITextToSpeechConfig()
            )

        # Cast to specific Vertex AI config type to access dispatch method
        vertex_config: Final = cast(VertexAITextToSpeechConfig, text_to_speech_provider_config)

        # Store Vertex AI specific params in litellm_params_dict
        litellm_params_dict.update(
            {
                "vertex_project": generic_optional_params.vertex_project,
                "vertex_location": generic_optional_params.vertex_location,
                "vertex_credentials": generic_optional_params.vertex_credentials,
            }
        )

        response = vertex_config.dispatch_text_to_speech(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params_dict=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=http_timeout,
            extra_headers=headers,
            base_llm_http_handler=base_llm_http_handler,
            aspeech=aspeech or False,
            api_base=api_base,
            api_key=None,  # Vertex AI uses OAuth, not API key
            **kwargs,
        )
    elif custom_llm_provider == "gemini":
        from .endpoints.speech.speech_to_completion_bridge.handler import (
            speech_to_completion_bridge_handler,
        )

        return speech_to_completion_bridge_handler.speech(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers or {},
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
        )
    elif custom_llm_provider == "runwayml":
        from litellm.llms.runwayml.text_to_speech.transformation import (
            RunwayMLTextToSpeechConfig,
        )

        # RunwayML Text-to-Speech
        if text_to_speech_provider_config is None:
            raise litellm.BadRequestError(
                message="RunwayML Text-to-Speech configuration not found",
                model=model,
                llm_provider=custom_llm_provider,
            )

        # Cast to specific RunwayML config type to access dispatch method
        runwayml_config: Final = cast(RunwayMLTextToSpeechConfig, text_to_speech_provider_config)

        response = runwayml_config.dispatch_text_to_speech(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params_dict=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=http_timeout,
            extra_headers=extra_headers,
            base_llm_http_handler=base_llm_http_handler,
            aspeech=aspeech or False,
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )
    elif custom_llm_provider == "minimax":
        from litellm.llms.minimax.text_to_speech.transformation import (
            MinimaxTextToSpeechConfig,
        )

        # MiniMax Text-to-Speech
        if text_to_speech_provider_config is None:
            text_to_speech_provider_config = MinimaxTextToSpeechConfig()

        minimax_config: Final = cast(MinimaxTextToSpeechConfig, text_to_speech_provider_config)

        if api_base is not None:
            litellm_params_dict["api_base"] = api_base
        if api_key is not None:
            litellm_params_dict["api_key"] = api_key

        # Convert voice to string if it's a dict (minimax handler expects Optional[str])
        voice_str: str | None = None
        if isinstance(voice, str):
            voice_str = voice
        elif isinstance(voice, dict):
            # Extract voice_id from dict if needed
            voice_str = voice.get("voice_id") or voice.get("id") or voice.get("name")

        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_str,
            text_to_speech_provider_config=minimax_config,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=http_timeout,
            extra_headers=extra_headers,
            client=client,
            _is_async=aspeech or False,
        )
    elif custom_llm_provider == "mistral":
        from litellm.llms.mistral.audio_speech.transformation import (
            MistralTextToSpeechConfig,
        )

        mistral_tts_config: Final = text_to_speech_provider_config or MistralTextToSpeechConfig()

        if api_base is not None:
            litellm_params_dict["api_base"] = api_base
        if api_key is not None:
            litellm_params_dict["api_key"] = api_key

        mistral_voice: Final[str | None] = voice if isinstance(voice, str) else None

        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=mistral_voice,
            text_to_speech_provider_config=mistral_tts_config,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=http_timeout,
            extra_headers=extra_headers,
            client=client,
            _is_async=aspeech or False,
        )
    elif custom_llm_provider == "aws_polly":
        from litellm.llms.aws_polly.text_to_speech.transformation import (
            AWSPollyTextToSpeechConfig,
        )

        # AWS Polly Text-to-Speech
        if text_to_speech_provider_config is None:
            text_to_speech_provider_config = AWSPollyTextToSpeechConfig()

        # Cast to specific AWS Polly config type to access dispatch method
        aws_polly_config: Final = cast(AWSPollyTextToSpeechConfig, text_to_speech_provider_config)

        response = aws_polly_config.dispatch_text_to_speech(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params_dict=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=http_timeout,
            extra_headers=extra_headers,
            base_llm_http_handler=base_llm_http_handler,
            aspeech=aspeech or False,
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )

    if response is None:
        raise Exception(
            f"Unable to map the custom llm provider={custom_llm_provider} to a known provider={litellm.provider_list}."
        )
    return response


##### Health Endpoints #######################


async def ahealth_check(
    model_params: dict,
    mode: str | None = None,
    prompt: str | None = None,
    input: list | None = None,
):
    """
    Support health checks for different providers. Return remaining rate limit, etc.

    Returns:
        {
            "x-ratelimit-remaining-requests": int,
            "x-ratelimit-remaining-tokens": int,
            "x-ms-region": str,
        }
    """
    from litellm.litellm_core_utils.cached_imports import get_litellm_logging_class
    from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers, default_health_check_mode
    from litellm.litellm_core_utils.health_check_utils import OPTIONAL_STR

    # Use cached import helper to lazy-load Logging class (only loads when function is called)
    Logging: Final = get_litellm_logging_class()

    # Map modes to their corresponding health check calls
    #########################################################
    # Init request with tracking information
    #########################################################
    litellm_logging_obj: Final = Logging(
        model="",
        messages=[],
        stream=False,
        call_type="acompletion",
        litellm_call_id=str(uuid.uuid4()),
        start_time=datetime.datetime.now(),
        function_id=str(uuid.uuid4()),
        log_raw_request_response=True,
    )
    model_params["litellm_logging_obj"] = litellm_logging_obj
    model_params = HealthCheckHelpers.update_model_params_with_health_check_tracking_information(
        model_params=model_params
    )
    #########################################################
    try:
        requested_model: Final = OPTIONAL_STR.validate_python(model_params.get("model", None))
        if requested_model is None:
            raise Exception("model not set")

        custom_llm_provider_from_params: Final = model_params.get("custom_llm_provider", None)
        api_base_from_params: Final = model_params.get("api_base", None)
        api_key_from_params: Final = model_params.get("api_key", None)

        model, custom_llm_provider, _, _ = get_llm_provider(
            model=requested_model,
            custom_llm_provider=custom_llm_provider_from_params,
            api_base=api_base_from_params,
            api_key=api_key_from_params,
        )

        model_params["cache"] = {"no-cache": True}  # don't used cached responses for making health check calls
        mode = mode or default_health_check_mode(
            requested_model=requested_model, model=model, custom_llm_provider=custom_llm_provider
        )
        if "*" in model:
            return await HealthCheckHelpers.ahealth_check_wildcard_models(
                model=model,
                custom_llm_provider=custom_llm_provider,
                model_params=model_params,
                litellm_logging_obj=litellm_logging_obj,
            )

        mode_handlers: Final = HealthCheckHelpers.get_mode_handlers(
            model=model,
            custom_llm_provider=custom_llm_provider,
            model_params=model_params,
            prompt=prompt,
            input=input,
        )

        handler_mode: Final = "decisions" if is_decisions_model_mode(mode) else mode
        if handler_mode in mode_handlers:
            _response: Final = await mode_handlers[handler_mode]()
            _response_headers: Final = cast(  # cast-ok: provider headers are stored as a string-keyed mapping
                Mapping[str, object], (get_hidden_params(_response) or {}).get("headers", {}) or {}
            )
            return create_health_check_response(_response_headers)
        else:
            raise Exception(f"Mode {mode} not supported. See modes here: https://docs.litellm.ai/docs/proxy/health")
    except Exception as e:
        stack_trace = redact_string(traceback.format_exc())
        if isinstance(stack_trace, str):
            stack_trace = stack_trace[:1000]

        error_to_return: Final = str(e) + "\nstack trace: " + stack_trace

        raw_request_typed_dict: Final = litellm_logging_obj.model_call_details.get("raw_request_typed_dict")

        return {
            "error": error_to_return,
            "raw_request_typed_dict": raw_request_typed_dict,
            "exception": e,
        }


####### HELPER FUNCTIONS ################
## Set verbose to true -> ```litellm.set_verbose = True```


def config_completion(**kwargs):
    if litellm.config_path is not None:
        config_args: Final = read_config_args(litellm.config_path)
        # overwrite any args passed in with config args
        return completion(**kwargs, **config_args)
    else:
        raise ValueError(
            "No config path set, please set a config path using `litellm.config_path = 'path/to/config.json'`"
        )


from litellm.litellm_core_utils.streaming_chunk_builder_utils import (  # noqa: E402  # re-export after the move
    stream_chunk_builder as stream_chunk_builder,  # noqa: PLC0414  # pyright: ignore[reportUnknownVariableType]  # chunks is an untyped list
)
from litellm.litellm_core_utils.streaming_chunk_builder_utils import (  # noqa: E402  # re-export after the move
    stream_chunk_builder_text_completion as stream_chunk_builder_text_completion,  # noqa: PLC0414  # pyright: ignore[reportUnknownVariableType]  # chunks is an untyped list
)

########## Token Counting API ##########


async def acount_tokens(
    model: str,
    messages: list[dict[str, object]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    system: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> "TokenCountResponse":
    """
    Count tokens for a given model and messages using provider-specific APIs.

    Routes to the appropriate provider's token counting API (OpenAI, Anthropic, etc.)
    for exact token counts. Falls back to local tiktoken-based counting for unsupported providers.

    Args:
        model: The model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022")
        messages: The messages to count tokens for (standard chat format)
        tools: Optional tools/functions to include in token count
        system: Optional system message/instructions
        api_key: Optional API key (falls back to environment variable)
        api_base: Optional custom API base URL

    Returns:
        TokenCountResponse with total_tokens and metadata
    """
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    from litellm.types.utils import LlmProviders, TokenCountResponse
    from litellm.utils import ProviderConfigManager

    # Determine provider from model string
    (
        resolved_model,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = get_llm_provider(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )

    # Use dynamic key/base if not explicitly provided
    if api_key is None:
        api_key = dynamic_api_key
    if api_base is None:
        api_base = dynamic_api_base

    # Build deployment dict for the token counter
    deployment: Final[dict[str, object]] = {
        "litellm_params": {
            "model": model,
            "api_key": api_key,
            "api_base": api_base,
        }
    }

    # Try to get provider-specific token counter
    try:
        llm_provider_enum: Final = LlmProviders(custom_llm_provider)
        provider_model_info = ProviderConfigManager.get_provider_model_info(model=model, provider=llm_provider_enum)

        if provider_model_info is not None:
            token_counter_instance: Final = provider_model_info.get_token_counter()
            if token_counter_instance is not None and token_counter_instance.should_use_token_counting_api(
                custom_llm_provider
            ):
                result: Final = await token_counter_instance.count_tokens(
                    model_to_use=resolved_model,
                    messages=messages,
                    contents=None,
                    deployment=deployment,
                    request_model=model,
                    tools=tools,
                    system=system,
                )
                if result is not None and not result.error:
                    return result
    except Exception as e:
        verbose_logger.debug("Provider token counting failed for model=%s, falling back to local: %s", model, e)

    # Fallback to local token counting
    fallback_messages = messages or []
    if system and fallback_messages:
        fallback_messages = [{"role": "system", "content": system}] + fallback_messages
    local_count: Final = await asyncify(litellm.token_counter)(
        model=model,
        messages=fallback_messages,
        tools=tools,
    )

    return TokenCountResponse(
        total_tokens=local_count,
        request_model=model,
        model_used=resolved_model,
        tokenizer_type="local_tokenizer",
    )


# Cache for encoding to avoid repeated __getattr__ calls
_encoding_cache: Tokenizer | None = None


def _load_module_encoding() -> Tokenizer:
    from litellm import main as litellm_main

    return litellm_main.encoding


def _get_encoding() -> Tokenizer:
    """Get encoding, loading it lazily if needed."""
    global _encoding_cache
    if _encoding_cache is None:
        # Access via module to trigger __getattr__ if not cached
        _encoding_cache = _load_module_encoding()
    return _encoding_cache


def _load_default_encoding() -> Tokenizer:
    from litellm._lazy_imports import get_default_encoding

    return get_default_encoding()


def __getattr__(name: str) -> Tokenizer:
    """Lazy import handler for main module"""
    if name == "encoding":
        _encoding: Final = _load_default_encoding()
        # Cache it in the module's __dict__ for subsequent accesses
        sys.modules[__name__].__dict__["encoding"] = _encoding
        global _encoding_cache
        _encoding_cache = _encoding
        return _encoding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
