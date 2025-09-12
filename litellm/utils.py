# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import ast
import asyncio
import base64
import binascii
import copy
import datetime
import hashlib
import inspect
import io
import itertools
import json
import logging
import os
import random  # type: ignore
import re
import struct
import subprocess

# What is this?
## Generic utils.py file. Problem-specific utils (e.g. 'cost calculation), should all be in `litellm_core_utils/`.
import sys
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from importlib import resources
from inspect import iscoroutine
from os.path import abspath, dirname, join

import aiohttp
import dotenv
import fastuuid as uuid
import httpx
import openai
import tiktoken
from httpx import Proxy
from httpx._utils import get_environment_proxies
from openai.lib import _parsing, _pydantic
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel
from tiktoken import Encoding
from tokenizers import Tokenizer

import litellm
import litellm._service_logger  # for storing API inputs, outputs, and metadata
import litellm.litellm_core_utils
import litellm.litellm_core_utils.audio_utils.utils
import litellm.litellm_core_utils.json_validation_rule
import litellm.llms
import litellm.llms.gemini
from litellm.caching._internal_lru_cache import lru_cache_wrapper
from litellm.caching.caching import DualCache
from litellm.caching.caching_handler import CachingHandlerResponse, LLMCachingHandler
from litellm.constants import (
    DEFAULT_CHAT_COMPLETION_PARAM_VALUES,
    DEFAULT_EMBEDDING_PARAM_VALUES,
    DEFAULT_MAX_LRU_CACHE_SIZE,
    DEFAULT_TRIM_RATIO,
    FUNCTION_DEFINITION_TOKEN_COUNT,
    INITIAL_RETRY_DELAY,
    JITTER,
    MAX_RETRY_DELAY,
    MAX_TOKEN_TRIMMING_ATTEMPTS,
    MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
    OPENAI_EMBEDDING_PARAMS,
    TOOL_CHOICE_OBJECT_TOKEN_COUNT,
)
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.vector_store_integrations.base_vector_store import (
    BaseVectorStore,
)
from litellm.litellm_core_utils.core_helpers import (
    map_finish_reason,
    process_response_headers,
)
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.default_encoding import encoding
from litellm.litellm_core_utils.exception_mapping_utils import (
    _get_response_headers,
    exception_type,
    get_error_message,
)
from litellm.litellm_core_utils.get_litellm_params import (
    _get_base_model_from_litellm_call_metadata,
    get_litellm_params,
)
from litellm.litellm_core_utils.get_llm_provider_logic import (
    _is_non_openai_azure_model,
    get_llm_provider,
)
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from litellm.litellm_core_utils.llm_request_utils import _ensure_extra_body_is_safe
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    LiteLLMResponseObjectHandler,
    _handle_invalid_parallel_tool_calls,
    _parse_content_for_reasoning,
    convert_to_model_response_object,
    convert_to_streaming_response,
    convert_to_streaming_response_async,
)
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.llm_response_utils.get_formatted_prompt import (
    get_formatted_prompt,
)
from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
)
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    ResponseMetadata,
)
from litellm.litellm_core_utils.redact_messages import (
    LiteLLMLoggingObject,
    redact_message_input_output_from_logging,
)
from litellm.litellm_core_utils.rules import Rules
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.litellm_core_utils.token_counter import get_modified_max_tokens
from litellm.llms.base_llm.google_genai.transformation import (
    BaseGoogleGenAIGenerateContentConfig,
)
from litellm.llms.bedrock.common_utils import BedrockModelInfo
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.router_utils.get_retry_from_policy import (
    get_num_retries_from_retry_policy,
    reset_retry_policy,
)
from litellm.secret_managers.main import get_secret
from litellm.types.llms.anthropic import (
    ANTHROPIC_API_ONLY_HEADERS,
    AnthropicThinkingParam,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    AllPromptValues,
    ChatCompletionAssistantToolCall,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    OpenAITextCompletionUserMessage,
    OpenAIWebSearchOptions,
)
from litellm.types.rerank import RerankResponse
from litellm.types.utils import FileTypes  # type: ignore
from litellm.types.utils import (
    OPENAI_RESPONSE_HEADERS,
    CallTypes,
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    CostPerToken,
    CredentialItem,
    CustomHuggingfaceTokenizer,
    Delta,
    Embedding,
    EmbeddingResponse,
    Function,
    ImageResponse,
    LlmProviders,
    LlmProvidersSet,
    LLMResponseTypes,
    Message,
    ModelInfo,
    ModelInfoBase,
    ModelResponse,
    ModelResponseStream,
    ProviderField,
    ProviderSpecificModelInfo,
    RawRequestTypedDict,
    SelectTokenizerResponse,
    StreamingChoices,
    TextChoices,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
    all_litellm_params,
)

try:
    # Python 3.9+
    with resources.files("litellm.litellm_core_utils.tokenizers").joinpath(
        "anthropic_tokenizer.json"
    ).open("r", encoding="utf-8") as f:
        json_data = json.load(f)
except (ImportError, AttributeError, TypeError):
    with resources.open_text(
        "litellm.litellm_core_utils.tokenizers", "anthropic_tokenizer.json"
    ) as f:
        json_data = json.load(f)

# Convert to str (if necessary)
claude_json_str = json.dumps(json_data)
import importlib.metadata
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
)

from openai import OpenAIError as OriginalError

from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.litellm_core_utils.token_counter import token_counter as token_counter_new
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.base_utils import (
    BaseLLMModelInfo,
    type_to_response_format_param,
)
from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.base_llm.completion.transformation import BaseTextCompletionConfig
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.files.transformation import BaseFilesConfig
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.base_llm.image_variations.transformation import (
    BaseImageVariationConfig,
)
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig

from ._logging import _is_debugging_on, verbose_logger
from .caching.caching import (
    AzureBlobCache,
    Cache,
    QdrantSemanticCache,
    RedisCache,
    RedisSemanticCache,
    S3Cache,
)
from .exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)
from .proxy._types import AllowedModelRegion, KeyManagementSystem
from .types.llms.openai import (
    ChatCompletionDeltaToolCallChunk,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from .types.router import LiteLLM_Params

####### ENVIRONMENT VARIABLES ####################
# Adjust to your specific application needs / system capabilities.
sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
posthog = None
slack_app = None
alerts_channel = None
heliconeLogger = None
athinaLogger = None
promptLayerLogger = None
langsmithLogger = None
logfireLogger = None
weightsBiasesLogger = None
customLogger = None
langFuseLogger = None
openMeterLogger = None
lagoLogger = None
dataDogLogger = None
prometheusLogger = None
dynamoLogger = None
s3Logger = None
greenscaleLogger = None
lunaryLogger = None
aispendLogger = None
supabaseClient = None
callback_list: Optional[List[str]] = []
user_logger_fn = None
additional_details: Optional[Dict[str, str]] = {}
local_cache: Optional[Dict[str, str]] = {}
last_fetched_at = None
last_fetched_at_keys = None
######## Model Response #########################

# All liteLLM Model responses will be in this format, Follows the OpenAI Format
# https://docs.litellm.ai/docs/completion/output
# {
#   'choices': [
#      {
#         'finish_reason': 'stop',
#         'index': 0,
#         'message': {
#            'role': 'assistant',
#             'content': " I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
#         }
#       }
#     ],
#  'created': 1691429984.3852863,
#  'model': 'claude-instant-1',
#  'usage': {'prompt_tokens': 18, 'completion_tokens': 23, 'total_tokens': 41}
# }


############################################################
def print_verbose(
    print_statement,
    logger_only: bool = False,
    log_level: Literal["DEBUG", "INFO", "ERROR"] = "DEBUG",
):
    try:
        if log_level == "DEBUG":
            verbose_logger.debug(print_statement)
        elif log_level == "INFO":
            verbose_logger.info(print_statement)
        elif log_level == "ERROR":
            verbose_logger.error(print_statement)
        if litellm.set_verbose is True and logger_only is False:
            print(print_statement)  # noqa
    except Exception:
        pass


####### CLIENT ###################
# make it easy to log if completion/embedding runs succeeded or failed + see what happened | Non-Blocking
def custom_llm_setup():
    """
    Add custom_llm provider to provider list
    """
    for custom_llm in litellm.custom_provider_map:
        if custom_llm["provider"] not in litellm.provider_list:
            litellm.provider_list.append(custom_llm["provider"])

        if custom_llm["provider"] not in litellm._custom_providers:
            litellm._custom_providers.append(custom_llm["provider"])


def _add_custom_logger_callback_to_specific_event(
    callback: str, logging_event: Literal["success", "failure"]
) -> None:
    """
    Add a custom logger callback to the specific event
    """
    from litellm import _custom_logger_compatible_callbacks_literal
    from litellm.litellm_core_utils.litellm_logging import (
        _init_custom_logger_compatible_class,
    )

    if callback not in litellm._known_custom_logger_compatible_callbacks:
        verbose_logger.debug(
            f"Callback {callback} is not a valid custom logger compatible callback. Known list - {litellm._known_custom_logger_compatible_callbacks}"
        )
        return

    callback_class = _init_custom_logger_compatible_class(
        cast(_custom_logger_compatible_callbacks_literal, callback),
        internal_usage_cache=None,
        llm_router=None,
    )

    if callback_class:
        if (
            logging_event == "success"
            and _custom_logger_class_exists_in_success_callbacks(callback_class)
            is False
        ):
            litellm.logging_callback_manager.add_litellm_success_callback(
                callback_class
            )
            litellm.logging_callback_manager.add_litellm_async_success_callback(
                callback_class
            )
            if callback in litellm.success_callback:
                litellm.success_callback.remove(
                    callback
                )  # remove the string from the callback list
            if callback in litellm._async_success_callback:
                litellm._async_success_callback.remove(
                    callback
                )  # remove the string from the callback list
        elif (
            logging_event == "failure"
            and _custom_logger_class_exists_in_failure_callbacks(callback_class)
            is False
        ):
            litellm.logging_callback_manager.add_litellm_failure_callback(
                callback_class
            )
            litellm.logging_callback_manager.add_litellm_async_failure_callback(
                callback_class
            )
            if callback in litellm.failure_callback:
                litellm.failure_callback.remove(
                    callback
                )  # remove the string from the callback list
            if callback in litellm._async_failure_callback:
                litellm._async_failure_callback.remove(
                    callback
                )  # remove the string from the callback list


def _custom_logger_class_exists_in_success_callbacks(
    callback_class: CustomLogger,
) -> bool:
    """
    Returns True if an instance of the custom logger exists in litellm.success_callback or litellm._async_success_callback

    e.g if `LangfusePromptManagement` is passed in, it will return True if an instance of `LangfusePromptManagement` exists in litellm.success_callback or litellm._async_success_callback

    Prevents double adding a custom logger callback to the litellm callbacks
    """
    return any(
        isinstance(cb, type(callback_class))
        for cb in litellm.success_callback + litellm._async_success_callback
    )


def _custom_logger_class_exists_in_failure_callbacks(
    callback_class: CustomLogger,
) -> bool:
    """
    Returns True if an instance of the custom logger exists in litellm.failure_callback or litellm._async_failure_callback

    e.g if `LangfusePromptManagement` is passed in, it will return True if an instance of `LangfusePromptManagement` exists in litellm.failure_callback or litellm._async_failure_callback

    Prevents double adding a custom logger callback to the litellm callbacks
    """
    return any(
        isinstance(cb, type(callback_class))
        for cb in litellm.failure_callback + litellm._async_failure_callback
    )


def get_request_guardrails(kwargs: Dict[str, Any]) -> List[str]:
    """
    Get the request guardrails from the kwargs
    """
    metadata = kwargs.get("metadata") or {}
    requester_metadata = metadata.get("requester_metadata") or {}
    applied_guardrails = requester_metadata.get("guardrails") or []
    return applied_guardrails


def get_applied_guardrails(kwargs: Dict[str, Any]) -> List[str]:
    """
    - Add 'default_on' guardrails to the list
    - Add request guardrails to the list
    """

    request_guardrails = get_request_guardrails(kwargs)
    applied_guardrails = []
    for callback in litellm.callbacks:
        if callback is not None and isinstance(callback, CustomGuardrail):
            if callback.guardrail_name is not None:
                if callback.default_on is True:
                    applied_guardrails.append(callback.guardrail_name)
                elif callback.guardrail_name in request_guardrails:
                    applied_guardrails.append(callback.guardrail_name)

    return applied_guardrails


def load_credentials_from_list(kwargs: dict):
    """
    Updates kwargs with the credentials if credential_name in kwarg
    """
    credential_name = kwargs.get("litellm_credential_name")
    if credential_name and litellm.credential_list:
        credential_accessor = CredentialAccessor.get_credential_values(credential_name)
        for key, value in credential_accessor.items():
            if key not in kwargs:
                kwargs[key] = value


def get_dynamic_callbacks(
    dynamic_callbacks: Optional[List[Union[str, Callable, CustomLogger]]],
) -> List:
    returned_callbacks = litellm.callbacks.copy()
    if dynamic_callbacks:
        returned_callbacks.extend(dynamic_callbacks)  # type: ignore
    return returned_callbacks


def function_setup(  # noqa: PLR0915
    original_function: str, rules_obj, start_time, *args, **kwargs
):  # just run once to check if user wants to send their data anywhere - PostHog/Sentry/Slack/etc.
    ### NOTICES ###
    from litellm import Logging as LiteLLMLogging
    from litellm.litellm_core_utils.litellm_logging import set_callbacks

    if litellm.set_verbose is True:
        verbose_logger.warning(
            "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
        )
    try:
        global callback_list, add_breadcrumb, user_logger_fn, Logging

        ## CUSTOM LLM SETUP ##
        custom_llm_setup()

        ## GET APPLIED GUARDRAILS
        applied_guardrails = get_applied_guardrails(kwargs)

        ## LOGGING SETUP
        function_id: Optional[str] = kwargs["id"] if "id" in kwargs else None

        ## DYNAMIC CALLBACKS ##
        dynamic_callbacks: Optional[List[Union[str, Callable, CustomLogger]]] = (
            kwargs.pop("callbacks", None)
        )
        all_callbacks = get_dynamic_callbacks(dynamic_callbacks=dynamic_callbacks)

        if len(all_callbacks) > 0:
            for callback in all_callbacks:
                # check if callback is a string - e.g. "lago", "openmeter"
                if isinstance(callback, str):
                    callback = litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class(  # type: ignore
                        callback, internal_usage_cache=None, llm_router=None  # type: ignore
                    )
                    if callback is None or any(
                        isinstance(cb, type(callback))
                        for cb in litellm._async_success_callback
                    ):  # don't double add a callback
                        continue
                if callback not in litellm.input_callback:
                    litellm.input_callback.append(callback)  # type: ignore
                if callback not in litellm.success_callback:
                    litellm.logging_callback_manager.add_litellm_success_callback(callback)  # type: ignore
                if callback not in litellm.failure_callback:
                    litellm.logging_callback_manager.add_litellm_failure_callback(callback)  # type: ignore
                if callback not in litellm._async_success_callback:
                    litellm.logging_callback_manager.add_litellm_async_success_callback(callback)  # type: ignore
                if callback not in litellm._async_failure_callback:
                    litellm.logging_callback_manager.add_litellm_async_failure_callback(callback)  # type: ignore
            print_verbose(
                f"Initialized litellm callbacks, Async Success Callbacks: {litellm._async_success_callback}"
            )

        if (
            len(litellm.input_callback) > 0
            or len(litellm.success_callback) > 0
            or len(litellm.failure_callback) > 0
        ) and len(
            callback_list  # type: ignore
        ) == 0:  # type: ignore
            callback_list = list(
                set(
                    litellm.input_callback  # type: ignore
                    + litellm.success_callback
                    + litellm.failure_callback
                )
            )
            set_callbacks(callback_list=callback_list, function_id=function_id)
        ## ASYNC CALLBACKS
        if len(litellm.input_callback) > 0:
            removed_async_items = []
            for index, callback in enumerate(litellm.input_callback):  # type: ignore
                if inspect.iscoroutinefunction(callback):
                    litellm._async_input_callback.append(callback)
                    removed_async_items.append(index)

            # Pop the async items from input_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                litellm.input_callback.pop(index)
        if len(litellm.success_callback) > 0:
            removed_async_items = []
            for index, callback in enumerate(litellm.success_callback):  # type: ignore
                if inspect.iscoroutinefunction(callback):
                    litellm.logging_callback_manager.add_litellm_async_success_callback(
                        callback
                    )
                    removed_async_items.append(index)
                elif callback == "dynamodb" or callback == "openmeter":
                    # dynamo is an async callback, it's used for the proxy and needs to be async
                    # we only support async dynamo db logging for acompletion/aembedding since that's used on proxy
                    litellm.logging_callback_manager.add_litellm_async_success_callback(
                        callback
                    )
                    removed_async_items.append(index)
                elif (
                    callback in litellm._known_custom_logger_compatible_callbacks
                    and isinstance(callback, str)
                ):
                    _add_custom_logger_callback_to_specific_event(callback, "success")

            # Pop the async items from success_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                litellm.success_callback.pop(index)

        if len(litellm.failure_callback) > 0:
            removed_async_items = []
            for index, callback in enumerate(litellm.failure_callback):  # type: ignore
                if inspect.iscoroutinefunction(callback):
                    litellm.logging_callback_manager.add_litellm_async_failure_callback(
                        callback
                    )
                    removed_async_items.append(index)
                elif (
                    callback in litellm._known_custom_logger_compatible_callbacks
                    and isinstance(callback, str)
                ):
                    _add_custom_logger_callback_to_specific_event(callback, "failure")

            # Pop the async items from failure_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                litellm.failure_callback.pop(index)
        ### DYNAMIC CALLBACKS ###
        dynamic_success_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None
        dynamic_async_success_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None
        dynamic_failure_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None
        dynamic_async_failure_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None
        if kwargs.get("success_callback", None) is not None and isinstance(
            kwargs["success_callback"], list
        ):
            removed_async_items = []
            for index, callback in enumerate(kwargs["success_callback"]):
                if (
                    inspect.iscoroutinefunction(callback)
                    or callback == "dynamodb"
                    or callback == "s3"
                ):
                    if dynamic_async_success_callbacks is not None and isinstance(
                        dynamic_async_success_callbacks, list
                    ):
                        dynamic_async_success_callbacks.append(callback)
                    else:
                        dynamic_async_success_callbacks = [callback]
                    removed_async_items.append(index)
            # Pop the async items from success_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                kwargs["success_callback"].pop(index)
            dynamic_success_callbacks = kwargs.pop("success_callback")
        if kwargs.get("failure_callback", None) is not None and isinstance(
            kwargs["failure_callback"], list
        ):
            dynamic_failure_callbacks = kwargs.pop("failure_callback")

        if add_breadcrumb:
            try:
                from litellm.litellm_core_utils.core_helpers import safe_deep_copy

                details_to_log = safe_deep_copy(kwargs)
            except Exception:
                details_to_log = kwargs

            if litellm.turn_off_message_logging:
                # make a copy of the _model_Call_details and log it
                details_to_log.pop("messages", None)
                details_to_log.pop("input", None)
                details_to_log.pop("prompt", None)
            add_breadcrumb(
                category="litellm.llm_call",
                message=f"Keyword Args: {details_to_log}",
                level="info",
            )
        if "logger_fn" in kwargs:
            user_logger_fn = kwargs["logger_fn"]
        # INIT LOGGER - for user-specified integrations
        model = args[0] if len(args) > 0 else kwargs.get("model", None)
        call_type = original_function
        if (
            call_type == CallTypes.completion.value
            or call_type == CallTypes.acompletion.value
            or call_type == CallTypes.anthropic_messages.value
        ):
            messages = None
            if len(args) > 1:
                messages = args[1]
            elif kwargs.get("messages", None):
                messages = kwargs["messages"]
            ### PRE-CALL RULES ###
            if (
                isinstance(messages, list)
                and len(messages) > 0
                and isinstance(messages[0], dict)
                and "content" in messages[0]
            ):
                rules_obj.pre_call_rules(
                    input="".join(
                        m.get("content", "")
                        for m in messages
                        if "content" in m and isinstance(m["content"], str)
                    ),
                    model=model,
                )
        elif (
            call_type == CallTypes.embedding.value
            or call_type == CallTypes.aembedding.value
        ):
            messages = args[1] if len(args) > 1 else kwargs.get("input", None)
        elif (
            call_type == CallTypes.image_generation.value
            or call_type == CallTypes.aimage_generation.value
        ):
            messages = args[0] if len(args) > 0 else kwargs["prompt"]
        elif (
            call_type == CallTypes.moderation.value
            or call_type == CallTypes.amoderation.value
        ):
            messages = args[1] if len(args) > 1 else kwargs["input"]
        elif (
            call_type == CallTypes.atext_completion.value
            or call_type == CallTypes.text_completion.value
        ):
            messages = args[0] if len(args) > 0 else kwargs["prompt"]
        elif (
            call_type == CallTypes.rerank.value or call_type == CallTypes.arerank.value
        ):
            messages = kwargs.get("query")
        elif (
            call_type == CallTypes.atranscription.value
            or call_type == CallTypes.transcription.value
        ):
            _file_obj: FileTypes = args[1] if len(args) > 1 else kwargs["file"]
            file_checksum = (
                litellm.litellm_core_utils.audio_utils.utils.get_audio_file_name(
                    file_obj=_file_obj
                )
            )
            if "metadata" in kwargs:
                kwargs["metadata"]["file_checksum"] = file_checksum
            else:
                kwargs["metadata"] = {"file_checksum": file_checksum}
            messages = file_checksum
        elif (
            call_type == CallTypes.aspeech.value or call_type == CallTypes.speech.value
        ):
            messages = kwargs.get("input", "speech")
        elif (
            call_type == CallTypes.aresponses.value
            or call_type == CallTypes.responses.value
        ):
            messages = args[0] if len(args) > 0 else kwargs["input"]
        else:
            messages = "default-message-value"
        stream = False
        if _is_streaming_request(
            kwargs=kwargs,
            call_type=call_type,
        ):
            stream = True
        logging_obj = LiteLLMLogging(
            model=model,  # type: ignore
            messages=messages,
            stream=stream,
            litellm_call_id=kwargs["litellm_call_id"],
            litellm_trace_id=kwargs.get("litellm_trace_id"),
            function_id=function_id or "",
            call_type=call_type,
            start_time=start_time,
            dynamic_success_callbacks=dynamic_success_callbacks,
            dynamic_failure_callbacks=dynamic_failure_callbacks,
            dynamic_async_success_callbacks=dynamic_async_success_callbacks,
            dynamic_async_failure_callbacks=dynamic_async_failure_callbacks,
            kwargs=kwargs,
            applied_guardrails=applied_guardrails,
        )

        ## check if metadata is passed in
        litellm_params: Dict[str, Any] = {"api_base": ""}
        if "metadata" in kwargs:
            litellm_params["metadata"] = kwargs["metadata"]

        logging_obj.update_environment_variables(
            model=model,
            user="",
            optional_params={},
            litellm_params=litellm_params,
            stream_options=kwargs.get("stream_options", None),
        )
        return logging_obj, kwargs
    except Exception as e:
        verbose_logger.exception(
            "litellm.utils.py::function_setup() - [Non-Blocking] Error in function_setup"
        )
        raise e


async def _client_async_logging_helper(
    logging_obj: LiteLLMLoggingObject,
    result,
    start_time,
    end_time,
    is_completion_with_fallbacks: bool,
):
    if (
        is_completion_with_fallbacks is False
    ):  # don't log the parent event litellm.completion_with_fallbacks as a 'log_success_event', this will lead to double logging the same call - https://github.com/BerriAI/litellm/issues/7477
        print_verbose(
            f"Async Wrapper: Completed Call, calling async_success_handler: {logging_obj.async_success_handler}"
        )
        ################################################
        # Async Logging Worker
        ################################################
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
            async_coroutine=logging_obj.async_success_handler(
                result=result, start_time=start_time, end_time=end_time
            )
        )

        ################################################
        # Sync Logging Worker
        ################################################
        logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=result,
            start_time=start_time,
            end_time=end_time,
        )


def _get_wrapper_num_retries(
    kwargs: Dict[str, Any], exception: Exception
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Get the number of retries from the kwargs and the retry policy.
    Used for the wrapper functions.
    """

    num_retries = kwargs.get("num_retries", None)
    if num_retries is None:
        num_retries = litellm.num_retries
    if kwargs.get("retry_policy", None):
        retry_policy_num_retries = get_num_retries_from_retry_policy(
            exception=exception,
            retry_policy=kwargs.get("retry_policy"),
        )
        kwargs["retry_policy"] = reset_retry_policy()
        if retry_policy_num_retries is not None:
            num_retries = retry_policy_num_retries

    return num_retries, kwargs


def _get_wrapper_timeout(
    kwargs: Dict[str, Any], exception: Exception
) -> Optional[Union[float, int, httpx.Timeout]]:
    """
    Get the timeout from the kwargs
    Used for the wrapper functions.
    """

    timeout = cast(
        Optional[Union[float, int, httpx.Timeout]], kwargs.get("timeout", None)
    )

    return timeout


def client(original_function):  # noqa: PLR0915
    rules_obj = Rules()

    def check_coroutine(value) -> bool:
        if inspect.iscoroutine(value):
            return True
        elif inspect.iscoroutinefunction(value):
            return True
        else:
            return False

    async def async_pre_call_deployment_hook(kwargs: Dict[str, Any], call_type: str):
        """
        Allow modifying the request just before it's sent to the deployment.

        Use this instead of 'async_pre_call_hook' when you need to modify the request AFTER a deployment is selected, but BEFORE the request is sent.
        """
        try:
            typed_call_type = CallTypes(call_type)
        except ValueError:
            typed_call_type = None  # unknown call type

        modified_kwargs = kwargs.copy()

        for callback in litellm.callbacks:
            if isinstance(callback, CustomLogger):
                result = await callback.async_pre_call_deployment_hook(
                    modified_kwargs, typed_call_type
                )
                if result is not None:
                    modified_kwargs = result

        return modified_kwargs

    async def async_post_call_success_deployment_hook(
        request_data: dict, response: Any, call_type: Optional[CallTypes]
    ) -> Optional[Any]:
        """
        Allow modifying / reviewing the response just after it's received from the deployment.
        """
        try:
            typed_call_type = CallTypes(call_type)
        except ValueError:
            typed_call_type = None  # unknown call type

        for callback in litellm.callbacks:
            if isinstance(callback, CustomLogger):
                result = await callback.async_post_call_success_deployment_hook(
                    request_data, cast(LLMResponseTypes, response), typed_call_type
                )
                if result is not None:
                    return result

        return response

    def post_call_processing(original_response, model, optional_params: Optional[dict]):
        try:
            if original_response is None:
                pass
            else:
                call_type = original_function.__name__
                if (
                    call_type == CallTypes.completion.value
                    or call_type == CallTypes.acompletion.value
                ):
                    is_coroutine = check_coroutine(original_response)
                    if is_coroutine is True:
                        pass
                    else:
                        if (
                            isinstance(original_response, ModelResponse)
                            and len(original_response.choices) > 0
                        ):
                            model_response: Optional[str] = original_response.choices[
                                0
                            ].message.content  # type: ignore
                            if model_response is not None:
                                ### POST-CALL RULES ###
                                rules_obj.post_call_rules(
                                    input=model_response, model=model
                                )
                                ### JSON SCHEMA VALIDATION ###
                                if litellm.enable_json_schema_validation is True:
                                    try:
                                        if (
                                            optional_params is not None
                                            and "response_format" in optional_params
                                            and optional_params["response_format"]
                                            is not None
                                        ):
                                            json_response_format: Optional[dict] = None
                                            if (
                                                isinstance(
                                                    optional_params["response_format"],
                                                    dict,
                                                )
                                                and optional_params[
                                                    "response_format"
                                                ].get("json_schema")
                                                is not None
                                            ):
                                                json_response_format = optional_params[
                                                    "response_format"
                                                ]
                                            elif _parsing._completions.is_basemodel_type(
                                                optional_params["response_format"]  # type: ignore
                                            ):
                                                json_response_format = (
                                                    type_to_response_format_param(
                                                        response_format=optional_params[
                                                            "response_format"
                                                        ]
                                                    )
                                                )
                                            if json_response_format is not None:
                                                litellm.litellm_core_utils.json_validation_rule.validate_schema(
                                                    schema=json_response_format[
                                                        "json_schema"
                                                    ]["schema"],
                                                    response=model_response,
                                                )
                                    except TypeError:
                                        pass
                                if (
                                    optional_params is not None
                                    and "response_format" in optional_params
                                    and isinstance(
                                        optional_params["response_format"], dict
                                    )
                                    and "type" in optional_params["response_format"]
                                    and optional_params["response_format"]["type"]
                                    == "json_object"
                                    and "response_schema"
                                    in optional_params["response_format"]
                                    and isinstance(
                                        optional_params["response_format"][
                                            "response_schema"
                                        ],
                                        dict,
                                    )
                                    and "enforce_validation"
                                    in optional_params["response_format"]
                                    and optional_params["response_format"][
                                        "enforce_validation"
                                    ]
                                    is True
                                ):
                                    # schema given, json response expected, and validation enforced
                                    litellm.litellm_core_utils.json_validation_rule.validate_schema(
                                        schema=optional_params["response_format"][
                                            "response_schema"
                                        ],
                                        response=model_response,
                                    )

        except Exception as e:
            raise e

    @wraps(original_function)
    def wrapper(*args, **kwargs):  # noqa: PLR0915
        # DO NOT MOVE THIS. It always needs to run first
        # Check if this is an async function. If so only execute the async function
        call_type = original_function.__name__
        if _is_async_request(kwargs):
            # [OPTIONAL] CHECK MAX RETRIES / REQUEST
            if litellm.num_retries_per_request is not None:
                # check if previous_models passed in as ['litellm_params']['metadata]['previous_models']
                previous_models = kwargs.get("metadata", {}).get(
                    "previous_models", None
                )
                if previous_models is not None:
                    if litellm.num_retries_per_request <= len(previous_models):
                        raise Exception("Max retries per request hit!")

            # MODEL CALL
            result = original_function(*args, **kwargs)
            if _is_streaming_request(
                kwargs=kwargs,
                call_type=call_type,
            ):
                if (
                    "complete_response" in kwargs
                    and kwargs["complete_response"] is True
                ):
                    chunks = []
                    for idx, chunk in enumerate(result):
                        chunks.append(chunk)
                    return litellm.stream_chunk_builder(
                        chunks, messages=kwargs.get("messages", None)
                    )
                else:
                    return result

            return result

        # Prints Exactly what was passed to litellm function - don't execute any logic here - it should just print
        print_args_passed_to_litellm(original_function, args, kwargs)
        start_time = datetime.datetime.now()
        result = None
        logging_obj: Optional[LiteLLMLoggingObject] = kwargs.get(
            "litellm_logging_obj", None
        )

        # only set litellm_call_id if its not in kwargs
        if "litellm_call_id" not in kwargs:
            kwargs["litellm_call_id"] = str(uuid.uuid4())

        model: Optional[str] = args[0] if len(args) > 0 else kwargs.get("model", None)

        try:
            if logging_obj is None:
                logging_obj, kwargs = function_setup(
                    original_function.__name__, rules_obj, start_time, *args, **kwargs
                )
            ## LOAD CREDENTIALS
            load_credentials_from_list(kwargs)
            kwargs["litellm_logging_obj"] = logging_obj
            _llm_caching_handler: LLMCachingHandler = LLMCachingHandler(
                original_function=original_function,
                request_kwargs=kwargs,
                start_time=start_time,
            )
            logging_obj._llm_caching_handler = _llm_caching_handler

            # CHECK FOR 'os.environ/' in kwargs
            for k, v in kwargs.items():
                if v is not None and isinstance(v, str) and v.startswith("os.environ/"):
                    kwargs[k] = litellm.get_secret(v)
            # [OPTIONAL] CHECK BUDGET
            if litellm.max_budget:
                if litellm._current_cost > litellm.max_budget:
                    raise BudgetExceededError(
                        current_cost=litellm._current_cost,
                        max_budget=litellm.max_budget,
                    )

            # [OPTIONAL] CHECK MAX RETRIES / REQUEST
            if litellm.num_retries_per_request is not None:
                # check if previous_models passed in as ['litellm_params']['metadata]['previous_models']
                previous_models = kwargs.get("metadata", {}).get(
                    "previous_models", None
                )
                if previous_models is not None:
                    if litellm.num_retries_per_request <= len(previous_models):
                        raise Exception("Max retries per request hit!")

            # [OPTIONAL] CHECK CACHE
            print_verbose(
                f"SYNC kwargs[caching]: {kwargs.get('caching', False)}; litellm.cache: {litellm.cache}; kwargs.get('cache')['no-cache']: {kwargs.get('cache', {}).get('no-cache', False)}"
            )
            # if caching is false or cache["no-cache"]==True, don't run this
            if (
                (
                    (
                        (
                            kwargs.get("caching", None) is None
                            and litellm.cache is not None
                        )
                        or kwargs.get("caching", False) is True
                    )
                    and kwargs.get("cache", {}).get("no-cache", False) is not True
                )
                and kwargs.get("aembedding", False) is not True
                and kwargs.get("atext_completion", False) is not True
                and kwargs.get("acompletion", False) is not True
                and kwargs.get("aimg_generation", False) is not True
                and kwargs.get("atranscription", False) is not True
                and kwargs.get("arerank", False) is not True
                and kwargs.get("_arealtime", False) is not True
            ):  # allow users to control returning cached responses from the completion function
                # checking cache
                verbose_logger.debug("INSIDE CHECKING SYNC CACHE")
                caching_handler_response: CachingHandlerResponse = (
                    _llm_caching_handler._sync_get_cache(
                        model=model or "",
                        original_function=original_function,
                        logging_obj=logging_obj,
                        start_time=start_time,
                        call_type=call_type,
                        kwargs=kwargs,
                        args=args,
                    )
                )

                if caching_handler_response.cached_result is not None:
                    verbose_logger.debug("Cache hit!")
                    return caching_handler_response.cached_result

            # CHECK MAX TOKENS
            if (
                kwargs.get("max_tokens", None) is not None
                and model is not None
                and litellm.modify_params
                is True  # user is okay with params being modified
                and (
                    call_type == CallTypes.acompletion.value
                    or call_type == CallTypes.completion.value
                    or call_type == CallTypes.anthropic_messages.value
                )
            ):
                try:
                    base_model = model
                    if kwargs.get("hf_model_name", None) is not None:
                        base_model = f"huggingface/{kwargs.get('hf_model_name')}"
                    messages = None
                    if len(args) > 1:
                        messages = args[1]
                    elif kwargs.get("messages", None):
                        messages = kwargs["messages"]
                    user_max_tokens = kwargs.get("max_tokens")
                    modified_max_tokens = get_modified_max_tokens(
                        model=model,
                        base_model=base_model,
                        messages=messages,
                        user_max_tokens=user_max_tokens,
                        buffer_num=None,
                        buffer_perc=None,
                    )
                    kwargs["max_tokens"] = modified_max_tokens
                except Exception as e:
                    print_verbose(f"Error while checking max token limit: {str(e)}")
            # MODEL CALL
            result = original_function(*args, **kwargs)
            end_time = datetime.datetime.now()
            if _is_streaming_request(
                kwargs=kwargs,
                call_type=call_type,
            ):
                if (
                    "complete_response" in kwargs
                    and kwargs["complete_response"] is True
                ):
                    chunks = []
                    for idx, chunk in enumerate(result):
                        chunks.append(chunk)
                    return litellm.stream_chunk_builder(
                        chunks, messages=kwargs.get("messages", None)
                    )
                else:
                    # RETURN RESULT
                    update_response_metadata(
                        result=result,
                        logging_obj=logging_obj,
                        model=model,
                        kwargs=kwargs,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    return result
            elif "acompletion" in kwargs and kwargs["acompletion"] is True:
                return result
            elif "aembedding" in kwargs and kwargs["aembedding"] is True:
                return result
            elif "aimg_generation" in kwargs and kwargs["aimg_generation"] is True:
                return result
            elif "atranscription" in kwargs and kwargs["atranscription"] is True:
                return result
            elif "aspeech" in kwargs and kwargs["aspeech"] is True:
                return result
            elif asyncio.iscoroutine(result):  # bubble up to relevant async function
                return result

            ### POST-CALL RULES ###
            post_call_processing(
                original_response=result,
                model=model or None,
                optional_params=kwargs,
            )

            # [OPTIONAL] ADD TO CACHE
            _llm_caching_handler.sync_set_cache(
                result=result,
                args=args,
                kwargs=kwargs,
            )

            # LOG SUCCESS - handle streaming success logging in the _next_ object, remove `handle_success` once it's deprecated
            verbose_logger.info("Wrapper: Completed Call, calling success_handler")
            executor.submit(
                logging_obj.success_handler,
                result,
                start_time,
                end_time,
            )
            # RETURN RESULT
            update_response_metadata(
                result=result,
                logging_obj=logging_obj,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )
            return result
        except Exception as e:
            call_type = original_function.__name__
            if call_type == CallTypes.completion.value:
                num_retries = (
                    kwargs.get("num_retries", None) or litellm.num_retries or None
                )
                if kwargs.get("retry_policy", None):
                    num_retries = get_num_retries_from_retry_policy(
                        exception=e,
                        retry_policy=kwargs.get("retry_policy"),
                    )
                    kwargs["retry_policy"] = (
                        reset_retry_policy()
                    )  # prevent infinite loops
                litellm.num_retries = (
                    None  # set retries to None to prevent infinite loops
                )
                context_window_fallback_dict = kwargs.get(
                    "context_window_fallback_dict", {}
                )

                _is_litellm_router_call = "model_group" in kwargs.get(
                    "metadata", {}
                )  # check if call from litellm.router/proxy
                if (
                    num_retries and not _is_litellm_router_call
                ):  # only enter this if call is not from litellm router/proxy. router has it's own logic for retrying
                    if (
                        isinstance(e, openai.APIError)
                        or isinstance(e, openai.Timeout)
                        or isinstance(e, openai.APIConnectionError)
                    ):
                        kwargs["num_retries"] = num_retries
                        return litellm.completion_with_retries(*args, **kwargs)
                elif (
                    isinstance(e, litellm.exceptions.ContextWindowExceededError)
                    and context_window_fallback_dict
                    and model in context_window_fallback_dict
                    and not _is_litellm_router_call
                ):
                    if len(args) > 0:
                        args[0] = context_window_fallback_dict[model]  # type: ignore
                    else:
                        kwargs["model"] = context_window_fallback_dict[model]
                    return original_function(*args, **kwargs)
            traceback_exception = traceback.format_exc()
            end_time = datetime.datetime.now()

            # LOG FAILURE - handle streaming failure logging in the _next_ object, remove `handle_failure` once it's deprecated
            if logging_obj:
                logging_obj.failure_handler(
                    e, traceback_exception, start_time, end_time
                )  # DO NOT MAKE THREADED - router retry fallback relies on this!
            raise e

    @wraps(original_function)
    async def wrapper_async(*args, **kwargs):  # noqa: PLR0915
        print_args_passed_to_litellm(original_function, args, kwargs)
        start_time = datetime.datetime.now()
        result = None
        logging_obj: Optional[LiteLLMLoggingObject] = kwargs.get(
            "litellm_logging_obj", None
        )
        _llm_caching_handler: LLMCachingHandler = LLMCachingHandler(
            original_function=original_function,
            request_kwargs=kwargs,
            start_time=start_time,
        )
        # only set litellm_call_id if its not in kwargs
        call_type = original_function.__name__
        if "litellm_call_id" not in kwargs:
            kwargs["litellm_call_id"] = str(uuid.uuid4())

        model: Optional[str] = args[0] if len(args) > 0 else kwargs.get("model", None)
        is_completion_with_fallbacks = kwargs.get("fallbacks") is not None

        try:
            if logging_obj is None:
                logging_obj, kwargs = function_setup(
                    original_function.__name__, rules_obj, start_time, *args, **kwargs
                )

            modified_kwargs = await async_pre_call_deployment_hook(kwargs, call_type)
            if modified_kwargs is not None:
                kwargs = modified_kwargs

            kwargs["litellm_logging_obj"] = logging_obj
            ## LOAD CREDENTIALS
            load_credentials_from_list(kwargs)
            logging_obj._llm_caching_handler = _llm_caching_handler
            # [OPTIONAL] CHECK BUDGET
            if litellm.max_budget:
                if litellm._current_cost > litellm.max_budget:
                    raise BudgetExceededError(
                        current_cost=litellm._current_cost,
                        max_budget=litellm.max_budget,
                    )

            # [OPTIONAL] CHECK CACHE
            print_verbose(
                f"ASYNC kwargs[caching]: {kwargs.get('caching', False)}; litellm.cache: {litellm.cache}; kwargs.get('cache'): {kwargs.get('cache', None)}"
            )
            _caching_handler_response: CachingHandlerResponse = (
                await _llm_caching_handler._async_get_cache(
                    model=model or "",
                    original_function=original_function,
                    logging_obj=logging_obj,
                    start_time=start_time,
                    call_type=call_type,
                    kwargs=kwargs,
                    args=args,
                )
            )

            if (
                _caching_handler_response.cached_result is not None
                and _caching_handler_response.final_embedding_cached_response is None
            ):
                return _caching_handler_response.cached_result

            elif _caching_handler_response.embedding_all_elements_cache_hit is True:
                return _caching_handler_response.final_embedding_cached_response

            # CHECK MAX TOKENS
            if (
                kwargs.get("max_tokens", None) is not None
                and model is not None
                and litellm.modify_params
                is True  # user is okay with params being modified
                and (
                    call_type == CallTypes.acompletion.value
                    or call_type == CallTypes.completion.value
                    or call_type == CallTypes.anthropic_messages.value
                )
            ):
                try:
                    base_model = model
                    if kwargs.get("hf_model_name", None) is not None:
                        base_model = f"huggingface/{kwargs.get('hf_model_name')}"
                    messages = None
                    if len(args) > 1:
                        messages = args[1]
                    elif kwargs.get("messages", None):
                        messages = kwargs["messages"]
                    user_max_tokens = kwargs.get("max_tokens")
                    modified_max_tokens = get_modified_max_tokens(
                        model=model,
                        base_model=base_model,
                        messages=messages,
                        user_max_tokens=user_max_tokens,
                        buffer_num=None,
                        buffer_perc=None,
                    )
                    kwargs["max_tokens"] = modified_max_tokens
                except Exception as e:
                    print_verbose(f"Error while checking max token limit: {str(e)}")

            # MODEL CALL
            result = await original_function(*args, **kwargs)
            end_time = datetime.datetime.now()
            if _is_streaming_request(
                kwargs=kwargs,
                call_type=call_type,
            ):
                if (
                    "complete_response" in kwargs
                    and kwargs["complete_response"] is True
                ):
                    chunks = []
                    for idx, chunk in enumerate(result):
                        chunks.append(chunk)
                    return litellm.stream_chunk_builder(
                        chunks, messages=kwargs.get("messages", None)
                    )
                else:
                    update_response_metadata(
                        result=result,
                        logging_obj=logging_obj,
                        model=model,
                        kwargs=kwargs,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    return result
            elif call_type == CallTypes.arealtime.value:
                return result
            ### POST-CALL RULES ###
            post_call_processing(
                original_response=result, model=model, optional_params=kwargs
            )
            # Only run if call_type is a valid value in CallTypes
            if call_type in [ct.value for ct in CallTypes]:
                await async_post_call_success_deployment_hook(
                    request_data=kwargs,
                    response=result,
                    call_type=CallTypes(call_type),
                )

            ## Add response to cache
            await _llm_caching_handler.async_set_cache(
                result=result,
                original_function=original_function,
                kwargs=kwargs,
                args=args,
            )

            # LOG SUCCESS - handle streaming success logging in the _next_ object
            asyncio.create_task(
                _client_async_logging_helper(
                    logging_obj=logging_obj,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    is_completion_with_fallbacks=is_completion_with_fallbacks,
                )
            )
            logging_obj.handle_sync_success_callbacks_for_async_calls(
                result=result,
                start_time=start_time,
                end_time=end_time,
            )
            # REBUILD EMBEDDING CACHING
            if (
                isinstance(result, EmbeddingResponse)
                and _caching_handler_response.final_embedding_cached_response
                is not None
            ):
                return _llm_caching_handler._combine_cached_embedding_response_with_api_result(
                    _caching_handler_response=_caching_handler_response,
                    embedding_response=result,
                    start_time=start_time,
                    end_time=end_time,
                )

            update_response_metadata(
                result=result,
                logging_obj=logging_obj,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )

            return result
        except Exception as e:
            traceback_exception = traceback.format_exc()
            end_time = datetime.datetime.now()
            if logging_obj:
                try:
                    logging_obj.failure_handler(
                        e, traceback_exception, start_time, end_time
                    )  # DO NOT MAKE THREADED - router retry fallback relies on this!
                except Exception as e:
                    raise e
                try:
                    await logging_obj.async_failure_handler(
                        e, traceback_exception, start_time, end_time
                    )
                except Exception as e:
                    raise e

            call_type = original_function.__name__
            num_retries, kwargs = _get_wrapper_num_retries(kwargs=kwargs, exception=e)
            if call_type == CallTypes.acompletion.value:
                context_window_fallback_dict = kwargs.get(
                    "context_window_fallback_dict", {}
                )

                _is_litellm_router_call = "model_group" in kwargs.get(
                    "metadata", {}
                )  # check if call from litellm.router/proxy

                if (
                    num_retries and not _is_litellm_router_call
                ):  # only enter this if call is not from litellm router/proxy. router has it's own logic for retrying
                    try:
                        litellm.num_retries = (
                            None  # set retries to None to prevent infinite loops
                        )
                        kwargs["num_retries"] = num_retries
                        kwargs["original_function"] = original_function
                        if isinstance(
                            e, openai.RateLimitError
                        ):  # rate limiting specific error
                            kwargs["retry_strategy"] = "exponential_backoff_retry"
                        elif isinstance(e, openai.APIError):  # generic api error
                            kwargs["retry_strategy"] = "constant_retry"
                        return await litellm.acompletion_with_retries(*args, **kwargs)
                    except Exception:
                        pass
                elif (
                    isinstance(e, litellm.exceptions.ContextWindowExceededError)
                    and context_window_fallback_dict
                    and model in context_window_fallback_dict
                ):
                    if len(args) > 0:
                        args[0] = context_window_fallback_dict[model]  # type: ignore
                    else:
                        kwargs["model"] = context_window_fallback_dict[model]
                    return await original_function(*args, **kwargs)

            setattr(
                e, "num_retries", num_retries
            )  ## IMPORTANT: returns the deployment's num_retries to the router

            timeout = _get_wrapper_timeout(kwargs=kwargs, exception=e)
            setattr(e, "timeout", timeout)
            raise e

    is_coroutine = inspect.iscoroutinefunction(original_function)

    # Return the appropriate wrapper based on the original function type
    if is_coroutine:
        return wrapper_async
    else:
        return wrapper


def _is_async_request(
    kwargs: Optional[dict],
    is_pass_through: bool = False,
) -> bool:
    """
    Returns True if the call type is an internal async request.

    eg. litellm.acompletion, litellm.aimage_generation, litellm.acreate_batch, litellm._arealtime

    Args:
        kwargs (dict): The kwargs passed to the litellm function
        is_pass_through (bool): Whether the call is a pass-through call. By default all pass through calls are async.
    """
    if kwargs is None:
        return False
    if (
        kwargs.get("acompletion", False) is True
        or kwargs.get("aembedding", False) is True
        or kwargs.get("aimg_generation", False) is True
        or kwargs.get("amoderation", False) is True
        or kwargs.get("atext_completion", False) is True
        or kwargs.get("atranscription", False) is True
        or kwargs.get("arerank", False) is True
        or kwargs.get("_arealtime", False) is True
        or kwargs.get("acreate_batch", False) is True
        or kwargs.get("acreate_fine_tuning_job", False) is True
        or is_pass_through is True
    ):
        return True
    return False


def _is_streaming_request(
    kwargs: Dict[str, Any],
    call_type: Union[CallTypes, str],
) -> bool:
    """
    Returns True if the call type is a streaming request.
    Returns True if:
        - if "stream=True" in kwargs  (litellm chat completion, litellm text completion, litellm messages)
        - if call_type is generate_content_stream or agenerate_content_stream (litellm google genai)
    """
    if "stream" in kwargs and kwargs["stream"] is True:
        return True

    #########################################################
    # Check if it's a google genai streaming request
    if isinstance(call_type, str):
        # check if it can be casted to CallTypes
        try:
            call_type = CallTypes(call_type)
        except ValueError:
            return False

    if (
        call_type == CallTypes.generate_content_stream
        or call_type == CallTypes.agenerate_content_stream
    ):
        return True
    #########################################################
    return False


def update_response_metadata(
    result: Any,
    logging_obj: LiteLLMLoggingObject,
    model: Optional[str],
    kwargs: dict,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    """
    Updates response metadata, adds the following:
        - response._hidden_params
        - response._hidden_params["litellm_overhead_time_ms"]
        - response.response_time_ms
    """
    if result is None:
        return

    metadata = ResponseMetadata(result)
    metadata.set_hidden_params(logging_obj=logging_obj, model=model, kwargs=kwargs)
    metadata.set_timing_metrics(
        start_time=start_time, end_time=end_time, logging_obj=logging_obj
    )
    metadata.apply()


def _select_tokenizer(
    model: str, custom_tokenizer: Optional[CustomHuggingfaceTokenizer] = None
):
    if custom_tokenizer is not None:
        _tokenizer = create_pretrained_tokenizer(
            identifier=custom_tokenizer["identifier"],
            revision=custom_tokenizer["revision"],
            auth_token=custom_tokenizer["auth_token"],
        )
        return _tokenizer
    return _select_tokenizer_helper(model=model)


@lru_cache(maxsize=DEFAULT_MAX_LRU_CACHE_SIZE)
def _select_tokenizer_helper(model: str) -> SelectTokenizerResponse:
    if litellm.disable_hf_tokenizer_download is True:
        return _return_openai_tokenizer(model)

    try:
        result = _return_huggingface_tokenizer(model)
        if result is not None:
            return result
    except Exception as e:
        verbose_logger.debug(f"Error selecting tokenizer: {e}")

    # default - tiktoken
    return _return_openai_tokenizer(model)


def _return_openai_tokenizer(model: str) -> SelectTokenizerResponse:
    return {"type": "openai_tokenizer", "tokenizer": encoding}


def _return_huggingface_tokenizer(model: str) -> Optional[SelectTokenizerResponse]:
    if model in litellm.cohere_models and "command-r" in model:
        # cohere
        cohere_tokenizer = Tokenizer.from_pretrained(
            "Xenova/c4ai-command-r-v01-tokenizer"
        )
        return {"type": "huggingface_tokenizer", "tokenizer": cohere_tokenizer}
    # anthropic
    elif model in litellm.anthropic_models and "claude-3" not in model:
        claude_tokenizer = Tokenizer.from_str(claude_json_str)
        return {"type": "huggingface_tokenizer", "tokenizer": claude_tokenizer}
    # llama2
    elif "llama-2" in model.lower() or "replicate" in model.lower():
        tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}
    # llama3
    elif "llama-3" in model.lower():
        tokenizer = Tokenizer.from_pretrained("Xenova/llama-3-tokenizer")
        return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}
    else:
        return None


def encode(model="", text="", custom_tokenizer: Optional[dict] = None):
    """
    Encodes the given text using the specified model.

    Args:
        model (str): The name of the model to use for tokenization.
        custom_tokenizer (Optional[dict]): A custom tokenizer created with the `create_pretrained_tokenizer` or `create_tokenizer` method. Must be a dictionary with a string value for `type` and Tokenizer for `tokenizer`. Default is None.
        text (str): The text to be encoded.

    Returns:
        enc: The encoded text.
    """
    tokenizer_json = custom_tokenizer or _select_tokenizer(model=model)
    if isinstance(tokenizer_json["tokenizer"], Encoding):
        enc = tokenizer_json["tokenizer"].encode(text, disallowed_special=())
    else:
        enc = tokenizer_json["tokenizer"].encode(text)
    return enc


def decode(model="", tokens: List[int] = [], custom_tokenizer: Optional[dict] = None):
    tokenizer_json = custom_tokenizer or _select_tokenizer(model=model)
    dec = tokenizer_json["tokenizer"].decode(tokens)
    return dec


def create_pretrained_tokenizer(
    identifier: str, revision="main", auth_token: Optional[str] = None
):
    """
    Creates a tokenizer from an existing file on a HuggingFace repository to be used with `token_counter`.

    Args:
    identifier (str): The identifier of a Model on the Hugging Face Hub, that contains a tokenizer.json file
    revision (str, defaults to main): A branch or commit id
    auth_token (str, optional, defaults to None): An optional auth token used to access private repositories on the Hugging Face Hub

    Returns:
    dict: A dictionary with the tokenizer and its type.
    """

    try:
        tokenizer = Tokenizer.from_pretrained(
            identifier, revision=revision, auth_token=auth_token  # type: ignore
        )
    except Exception as e:
        verbose_logger.error(
            f"Error creating pretrained tokenizer: {e}. Defaulting to version without 'auth_token'."
        )
        tokenizer = Tokenizer.from_pretrained(identifier, revision=revision)
    return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}


def create_tokenizer(json: str):
    """
    Creates a tokenizer from a valid JSON string for use with `token_counter`.

    Args:
    json (str): A valid JSON string representing a previously serialized tokenizer

    Returns:
    dict: A dictionary with the tokenizer and its type.
    """

    tokenizer = Tokenizer.from_str(json)
    return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}


def token_counter(
    model="",
    custom_tokenizer: Optional[Union[dict, SelectTokenizerResponse]] = None,
    text: Optional[Union[str, List[str]]] = None,
    messages: Optional[List] = None,
    count_response_tokens: Optional[bool] = False,
    tools: Optional[List[ChatCompletionToolParam]] = None,
    tool_choice: Optional[ChatCompletionNamedToolChoiceParam] = None,
    use_default_image_token_count: Optional[bool] = False,
    default_token_count: Optional[int] = None,
) -> int:
    """
    The same as `litellm.litellm_core_utils.token_counter`.

    Kept for backwards compatibility.
    """

    #########################################################
    # Flag to disable token counter
    # We've gotten reports of this consuming CPU cycles,
    # exposing this flag to allow users to disable
    # it to confirm if this is indeed the issue
    #########################################################
    if litellm.disable_token_counter is True:
        return 0

    return token_counter_new(
        model,
        custom_tokenizer,
        text,
        messages,
        count_response_tokens,
        tools,
        tool_choice,
        use_default_image_token_count,
        default_token_count,
    )


def supports_httpx_timeout(custom_llm_provider: str) -> bool:
    """
    Helper function to know if a provider implementation supports httpx timeout
    """
    supported_providers = ["openai", "azure", "bedrock"]

    if custom_llm_provider in supported_providers:
        return True

    return False


def supports_system_messages(model: str, custom_llm_provider: Optional[str]) -> bool:
    """
    Check if the given model supports system messages and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (str): The provider to be checked.

    Returns:
    bool: True if the model supports system messages, False otherwise.

    Raises:
    Exception: If the given model is not found in model_prices_and_context_window.json.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_system_messages",
    )


def supports_web_search(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """
    Check if the given model supports web search and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (str): The provider to be checked.

    Returns:
    bool: True if the model supports web search, False otherwise.

    Raises:
    Exception: If the given model is not found in model_prices_and_context_window.json.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_web_search",
    )


def supports_url_context(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """
    Check if the given model supports URL context and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (str): The provider to be checked.

    Returns:
    bool: True if the model supports URL context, False otherwise.

    Raises:
    Exception: If the given model is not found in model_prices_and_context_window.json.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_url_context",
    )


def supports_native_streaming(model: str, custom_llm_provider: Optional[str]) -> bool:
    """
    Check if the given model supports native streaming and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (str): The provider to be checked.

    Returns:
    bool: True if the model supports native streaming, False otherwise.

    Raises:
    Exception: If the given model is not found in model_prices_and_context_window.json.
    """
    try:
        model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider
        )

        model_info = _get_model_info_helper(
            model=model, custom_llm_provider=custom_llm_provider
        )
        supports_native_streaming = model_info.get("supports_native_streaming", True)
        if supports_native_streaming is None:
            supports_native_streaming = True
        return supports_native_streaming
    except Exception as e:
        verbose_logger.debug(
            f"Model not found or error in checking supports_native_streaming support. You passed model={model}, custom_llm_provider={custom_llm_provider}. Error: {str(e)}"
        )
        return False


def supports_response_schema(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """
    Check if the given model + provider supports 'response_schema' as a param.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (str): The provider to be checked.

    Returns:
    bool: True if the model supports response_schema, False otherwise.

    Does not raise error. Defaults to 'False'. Outputs logging.error.
    """
    ## GET LLM PROVIDER ##
    try:
        model, custom_llm_provider, _, _ = get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider
        )
    except Exception as e:
        verbose_logger.debug(
            f"Model not found or error in checking response schema support. You passed model={model}, custom_llm_provider={custom_llm_provider}. Error: {str(e)}"
        )
        return False

    # providers that globally support response schema
    PROVIDERS_GLOBALLY_SUPPORT_RESPONSE_SCHEMA = [
        litellm.LlmProviders.PREDIBASE,
        litellm.LlmProviders.FIREWORKS_AI,
        litellm.LlmProviders.LM_STUDIO,
        litellm.LlmProviders.NEBIUS,
    ]

    if custom_llm_provider in PROVIDERS_GLOBALLY_SUPPORT_RESPONSE_SCHEMA:
        return True
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_response_schema",
    )


def supports_parallel_function_calling(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """
    Check if the given model supports parallel tool calls and return a boolean value.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_parallel_function_calling",
    )


def supports_function_calling(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """
    Check if the given model supports function calling and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (Optional[str]): The provider to be checked.

    Returns:
    bool: True if the model supports function calling, False otherwise.

    Raises:
    Exception: If the given model is not found or there's an error in retrieval.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_function_calling",
    )


def supports_tool_choice(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """
    Check if the given model supports `tool_choice` and return a boolean value.
    """
    return _supports_factory(
        model=model, custom_llm_provider=custom_llm_provider, key="supports_tool_choice"
    )


def _supports_provider_info_factory(
    model: str, custom_llm_provider: Optional[str], key: str
) -> Optional[Literal[True]]:
    """
    Check if the given model supports a provider specific model info and return a boolean value.
    """

    provider_info = get_provider_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    if provider_info is not None and provider_info.get(key, False) is True:
        return True
    return None


def _supports_factory(model: str, custom_llm_provider: Optional[str], key: str) -> bool:
    """
    Check if the given model supports function calling and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (Optional[str]): The provider to be checked.

    Returns:
    bool: True if the model supports function calling, False otherwise.

    Raises:
    Exception: If the given model is not found or there's an error in retrieval.
    """
    try:
        model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider
        )

        model_info = _get_model_info_helper(
            model=model, custom_llm_provider=custom_llm_provider
        )

        if model_info.get(key, False) is True:
            return True
        elif model_info.get(key) is None:  # don't check if 'False' explicitly set
            supported_by_provider = _supports_provider_info_factory(
                model, custom_llm_provider, key
            )
            if supported_by_provider is not None:
                return supported_by_provider

        return False
    except Exception as e:
        verbose_logger.debug(
            f"Model not found or error in checking {key} support. You passed model={model}, custom_llm_provider={custom_llm_provider}. Error: {str(e)}"
        )

        supported_by_provider = _supports_provider_info_factory(
            model, custom_llm_provider, key
        )
        if supported_by_provider is not None:
            return supported_by_provider

        return False


def supports_audio_input(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """Check if a given model supports audio input in a chat completion call"""
    return _supports_factory(
        model=model, custom_llm_provider=custom_llm_provider, key="supports_audio_input"
    )


def supports_pdf_input(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """Check if a given model supports pdf input in a chat completion call"""
    return _supports_factory(
        model=model, custom_llm_provider=custom_llm_provider, key="supports_pdf_input"
    )


def supports_audio_output(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Check if a given model supports audio output in a chat completion call"""
    return _supports_factory(
        model=model, custom_llm_provider=custom_llm_provider, key="supports_audio_input"
    )


def supports_prompt_caching(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """
    Check if the given model supports prompt caching and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (Optional[str]): The provider to be checked.

    Returns:
    bool: True if the model supports prompt caching, False otherwise.

    Raises:
    Exception: If the given model is not found or there's an error in retrieval.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_prompt_caching",
    )


def supports_computer_use(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """
    Check if the given model supports computer use and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (Optional[str]): The provider to be checked.

    Returns:
    bool: True if the model supports computer use, False otherwise.

    Raises:
    Exception: If the given model is not found or there's an error in retrieval.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_computer_use",
    )


def supports_vision(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """
    Check if the given model supports vision and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (Optional[str]): The provider to be checked.

    Returns:
    bool: True if the model supports vision, False otherwise.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_vision",
    )


def supports_reasoning(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """
    Check if the given model supports reasoning and return a boolean value.
    """
    return _supports_factory(
        model=model, custom_llm_provider=custom_llm_provider, key="supports_reasoning"
    )


def get_supported_regions(
    model: str, custom_llm_provider: Optional[str] = None
) -> Optional[List[str]]:
    """
    Get a list of supported regions for a given model and provider.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (Optional[str]): The provider to be checked.
    """
    try:
        model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider
        )

        model_info = _get_model_info_helper(
            model=model, custom_llm_provider=custom_llm_provider
        )

        supported_regions = model_info.get("supported_regions", None)
        if supported_regions is None:
            return None

        #########################################################
        # Ensure only list supported regions are returned
        #########################################################
        if isinstance(supported_regions, list):
            return supported_regions
        else:
            return None
    except Exception as e:
        verbose_logger.debug(
            f"Model not found or error in checking supported_regions support. You passed model={model}, custom_llm_provider={custom_llm_provider}. Error: {str(e)}"
        )
        return None


def supports_embedding_image_input(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """
    Check if the given model supports embedding image input and return a boolean value.
    """
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_embedding_image_input",
    )


####### HELPER FUNCTIONS ################
def _update_dictionary(existing_dict: Dict, new_dict: dict) -> dict:
    for k, v in new_dict.items():
        if v is not None:
            # Convert stringified numbers to appropriate numeric types
            if isinstance(v, str):
                existing_dict[k] = _convert_stringified_numbers(v)
            elif isinstance(v, dict):
                existing_nested_dict = existing_dict.get(k)
                if isinstance(existing_nested_dict, dict):
                    existing_nested_dict.update(v)
                    existing_dict[k] = existing_nested_dict
                else:
                    existing_dict[k] = v
            else:
                existing_dict[k] = v

    return existing_dict


def _convert_stringified_numbers(value):
    """Convert stringified numbers (including scientific notation) to appropriate numeric types."""
    if isinstance(value, str):
        try:
            # Try to convert to float first to handle scientific notation like "3e-07"
            if "e" in value.lower() or "." in value:
                return float(value)
            # Try to convert to int for whole numbers like "8192"
            else:
                return int(value)
        except (ValueError, TypeError):
            # If conversion fails, return the original string
            return value
    return value


def register_model(model_cost: Union[str, dict]):  # noqa: PLR0915
    """
    Register new / Override existing models (and their pricing) to specific providers.
    Provide EITHER a model cost dictionary or a url to a hosted json blob
    Example usage:
    model_cost_dict = {
        "gpt-4": {
            "max_tokens": 8192,
            "input_cost_per_token": 0.00003,
            "output_cost_per_token": 0.00006,
            "litellm_provider": "openai",
            "mode": "chat"
        },
    }
    """

    loaded_model_cost = {}
    if isinstance(model_cost, dict):
        # Convert stringified numbers to appropriate numeric types
        loaded_model_cost = model_cost
    elif isinstance(model_cost, str):
        loaded_model_cost = litellm.get_model_cost_map(url=model_cost)

    for key, value in loaded_model_cost.items():
        ## get model info ##
        try:
            existing_model: dict = cast(dict, get_model_info(model=key))
            model_cost_key = existing_model["key"]
        except Exception:
            existing_model = {}
            model_cost_key = key
        ## override / add new keys to the existing model cost dictionary
        updated_dictionary = _update_dictionary(existing_model, value)
        litellm.model_cost.setdefault(model_cost_key, {}).update(updated_dictionary)
        verbose_logger.debug(
            f"added/updated model={model_cost_key} in litellm.model_cost: {model_cost_key}"
        )
        # add new model names to provider lists
        if value.get("litellm_provider") == "openai":
            if key not in litellm.open_ai_chat_completion_models:
                litellm.open_ai_chat_completion_models.add(key)
        elif value.get("litellm_provider") == "text-completion-openai":
            if key not in litellm.open_ai_text_completion_models:
                litellm.open_ai_text_completion_models.add(key)
        elif value.get("litellm_provider") == "cohere":
            if key not in litellm.cohere_models:
                litellm.cohere_models.add(key)
        elif value.get("litellm_provider") == "anthropic":
            if key not in litellm.anthropic_models:
                litellm.anthropic_models.add(key)
        elif value.get("litellm_provider") == "openrouter":
            split_string = key.split("/", 1)
            if key not in litellm.openrouter_models:
                litellm.openrouter_models.add(split_string[1])
        elif value.get("litellm_provider") == "vercel_ai_gateway":
            if key not in litellm.vercel_ai_gateway_models:
                litellm.vercel_ai_gateway_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-text-models":
            if key not in litellm.vertex_text_models:
                litellm.vertex_text_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-code-text-models":
            if key not in litellm.vertex_code_text_models:
                litellm.vertex_code_text_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-chat-models":
            if key not in litellm.vertex_chat_models:
                litellm.vertex_chat_models.add(key)
        elif value.get("litellm_provider") == "vertex_ai-code-chat-models":
            if key not in litellm.vertex_code_chat_models:
                litellm.vertex_code_chat_models.add(key)
        elif value.get("litellm_provider") == "ai21":
            if key not in litellm.ai21_models:
                litellm.ai21_models.add(key)
        elif value.get("litellm_provider") == "nlp_cloud":
            if key not in litellm.nlp_cloud_models:
                litellm.nlp_cloud_models.add(key)
        elif value.get("litellm_provider") == "aleph_alpha":
            if key not in litellm.aleph_alpha_models:
                litellm.aleph_alpha_models.add(key)
        elif value.get("litellm_provider") == "bedrock":
            if key not in litellm.bedrock_models:
                litellm.bedrock_models.add(key)
        elif value.get("litellm_provider") == "novita":
            if key not in litellm.novita_models:
                litellm.novita_models.add(key)
    return model_cost


def _should_drop_param(k, additional_drop_params) -> bool:
    if (
        additional_drop_params is not None
        and isinstance(additional_drop_params, list)
        and k in additional_drop_params
    ):
        return True  # allow user to drop specific params for a model - e.g. vllm - logit bias

    return False


def _get_non_default_params(
    passed_params: dict, default_params: dict, additional_drop_params: Optional[list]
) -> dict:
    non_default_params = {}
    for k, v in passed_params.items():
        if (
            k in default_params
            and v != default_params[k]
            and _should_drop_param(k=k, additional_drop_params=additional_drop_params)
            is False
        ):
            non_default_params[k] = v

    return non_default_params


def get_optional_params_transcription(
    model: str,
    custom_llm_provider: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    temperature: Optional[int] = None,
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None,
    drop_params: Optional[bool] = None,
    **kwargs,
):
    from litellm.constants import OPENAI_TRANSCRIPTION_PARAMS

    # retrieve all parameters passed to the function
    passed_params = locals()
    custom_llm_provider = passed_params.pop("custom_llm_provider")
    drop_params = passed_params.pop("drop_params")
    special_params = passed_params.pop("kwargs")
    for k, v in special_params.items():
        passed_params[k] = v

    default_params = {
        "language": None,
        "prompt": None,
        "response_format": None,
        "temperature": None,  # openai defaults this to 0
        "timestamp_granularities": None
    }

    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (k in default_params and v != default_params[k])
    }
    optional_params = {}

    ## raise exception if non-default value passed for non-openai/azure embedding calls
    def _check_valid_arg(supported_params):
        if len(non_default_params.keys()) > 0:
            keys = list(non_default_params.keys())
            for k in keys:
                if (
                    drop_params is True or litellm.drop_params is True
                ) and k not in supported_params:  # drop the unsupported non-default values
                    non_default_params.pop(k, None)
                elif k not in supported_params:
                    raise UnsupportedParamsError(
                        status_code=500,
                        message=f"Setting user/encoding format is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
                    )
            return non_default_params

    provider_config: Optional[BaseAudioTranscriptionConfig] = None
    if custom_llm_provider is not None:
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model=model,
            provider=LlmProviders(custom_llm_provider),
        )

    if custom_llm_provider == "openai" or custom_llm_provider == "azure":
        optional_params = non_default_params
    elif custom_llm_provider == "groq":
        supported_params = litellm.GroqSTTConfig().get_supported_openai_params_stt()
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.GroqSTTConfig().map_openai_params_stt(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )
    elif provider_config is not None:  # handles fireworks ai, and any future providers
        supported_params = provider_config.get_supported_openai_params(model=model)
        _check_valid_arg(supported_params=supported_params)
        optional_params = provider_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )
    optional_params = add_provider_specific_params_to_optional_params(
        optional_params=optional_params,
        passed_params=passed_params,
        custom_llm_provider=custom_llm_provider,
        openai_params=OPENAI_TRANSCRIPTION_PARAMS,
        additional_drop_params=kwargs.get("additional_drop_params", None),
    )

    return optional_params


def _map_openai_size_to_vertex_ai_aspect_ratio(size: Optional[str]) -> str:
    """Map OpenAI size parameter to Vertex AI aspectRatio."""
    if size is None:
        return "1:1"
    
    # Map OpenAI size strings to Vertex AI aspect ratio strings
    # Vertex AI accepts: "1:1", "9:16", "16:9", "4:3", "3:4"
    size_to_aspect_ratio = {
        "256x256": "1:1",  # Square
        "512x512": "1:1",  # Square
        "1024x1024": "1:1",  # Square (default)
        "1792x1024": "16:9",  # Landscape
        "1024x1792": "9:16",  # Portrait
    }
    return size_to_aspect_ratio.get(size, "1:1")  # Default to square if size not recognized


def get_optional_params_image_gen(
    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[str] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    additional_drop_params: Optional[list] = None,
    provider_config: Optional[BaseImageGenerationConfig] = None,
    drop_params: Optional[bool] = None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals()
    model = passed_params.pop("model", None)
    custom_llm_provider = passed_params.pop("custom_llm_provider")
    provider_config = passed_params.pop("provider_config", None)
    drop_params = passed_params.pop("drop_params", None)
    additional_drop_params = passed_params.pop("additional_drop_params", None)
    special_params = passed_params.pop("kwargs")
    for k, v in special_params.items():
        if k.startswith("aws_") and (
            custom_llm_provider != "bedrock" and custom_llm_provider != "sagemaker"
        ):  # allow dynamically setting boto3 init logic
            continue
        elif k == "hf_model_name" and custom_llm_provider != "sagemaker":
            continue
        elif (
            k.startswith("vertex_")
            and custom_llm_provider != "vertex_ai"
            and custom_llm_provider != "vertex_ai_beta"
        ):  # allow dynamically setting vertex ai init logic
            continue
        passed_params[k] = v

    default_params = {
        "n": None,
        "quality": None,
        "response_format": None,
        "size": None,
        "style": None,
        "user": None,
    }

    non_default_params = _get_non_default_params(
        passed_params=passed_params,
        default_params=default_params,
        additional_drop_params=additional_drop_params,
    )
    optional_params: Dict[str, Any] = {}

    ## raise exception if non-default value passed for non-openai/azure embedding calls
    def _check_valid_arg(supported_params):
        if len(non_default_params.keys()) > 0:
            keys = list(non_default_params.keys())
            for k in keys:
                if (
                    litellm.drop_params is True or drop_params is True
                ) and k not in supported_params:  # drop the unsupported non-default values
                    non_default_params.pop(k, None)
                elif k not in supported_params:
                    raise UnsupportedParamsError(
                        status_code=500,
                        message=f"Setting `{k}` is not supported by {custom_llm_provider}, {model}. To drop it from the call, set `litellm.drop_params = True`.",
                    )
            return non_default_params

    if provider_config is not None:
        supported_params = provider_config.get_supported_openai_params(
            model=model or ""
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = provider_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model or "",
            drop_params=drop_params if drop_params is not None else False,
        )
    elif (
        custom_llm_provider == "openai"
        or custom_llm_provider == "azure"
        or custom_llm_provider in litellm.openai_compatible_providers
    ):
        optional_params = non_default_params
    elif custom_llm_provider == "bedrock":
        # use stability3 config class if model is a stability3 model
        config_class = (
            litellm.AmazonStability3Config
            if litellm.AmazonStability3Config._is_stability_3_model(model=model)
            else (
                litellm.AmazonNovaCanvasConfig
                if litellm.AmazonNovaCanvasConfig._is_nova_model(model=model)
                else litellm.AmazonStabilityConfig
            )
        )
        supported_params = config_class.get_supported_openai_params(model=model)
        _check_valid_arg(supported_params=supported_params)
        optional_params = config_class.map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
    elif custom_llm_provider == "vertex_ai":
        supported_params = ["n", "size"]
        """
        All params here: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration?project=adroit-crow-413218
        """
        _check_valid_arg(supported_params=supported_params)
        if n is not None:
            optional_params["sampleCount"] = int(n)

        # Map OpenAI size parameter to Vertex AI aspectRatio
        if size is not None:
            optional_params["aspectRatio"] = _map_openai_size_to_vertex_ai_aspect_ratio(size)

    openai_params: list[str] = list(default_params.keys())
    if provider_config is not None:
        supported_params = provider_config.get_supported_openai_params(
            model=model or ""
        )
        openai_params = list(supported_params)

    optional_params = add_provider_specific_params_to_optional_params(
        optional_params=optional_params,
        passed_params=passed_params,
        custom_llm_provider=custom_llm_provider or "",
        openai_params=openai_params,
        additional_drop_params=additional_drop_params,
    )
    # remove keys with None or empty dict/list values to avoid sending empty payloads
    optional_params = {
        k: v
        for k, v in optional_params.items()
        if v is not None and (not isinstance(v, (dict, list)) or len(v) > 0)
    }
    return optional_params


def get_optional_params_embeddings(  # noqa: PLR0915
    # 2 optional params
    model: str,
    user: Optional[str] = None,
    encoding_format: Optional[str] = None,
    dimensions: Optional[int] = None,
    custom_llm_provider="",
    drop_params: Optional[bool] = None,
    additional_drop_params: Optional[List[str]] = None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals()
    custom_llm_provider = passed_params.pop("custom_llm_provider", None)
    special_params = passed_params.pop("kwargs")

    drop_params = passed_params.pop("drop_params", None)
    additional_drop_params = passed_params.pop("additional_drop_params", None)

    def _check_valid_arg(supported_params: Optional[list]):
        if supported_params is None:
            return
        unsupported_params = {}
        for k in non_default_params.keys():
            if k not in supported_params:
                unsupported_params[k] = non_default_params[k]
        if unsupported_params:
            if litellm.drop_params is True or (
                drop_params is not None and drop_params is True
            ):
                pass
            else:
                raise UnsupportedParamsError(
                    status_code=500,
                    message=f"{custom_llm_provider} does not support parameters: {unsupported_params}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n",
                )

    non_default_params = (
        PreProcessNonDefaultParams.embedding_pre_process_non_default_params(
            passed_params=passed_params,
            special_params=special_params,
            custom_llm_provider=custom_llm_provider,
            additional_drop_params=additional_drop_params,
            model=model,
        )
    )

    provider_config: Optional[BaseEmbeddingConfig] = None

    optional_params = {}
    if (
        custom_llm_provider is not None
        and custom_llm_provider in LlmProviders._member_map_.values()
    ):
        provider_config = ProviderConfigManager.get_provider_embedding_config(
            model=model,
            provider=LlmProviders(custom_llm_provider),
        )

    if provider_config is not None:
        supported_params: Optional[list] = provider_config.get_supported_openai_params(
            model=model
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = provider_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )
    ## raise exception if non-default value passed for non-openai/azure embedding calls
    elif custom_llm_provider == "openai":
        # 'dimensions` is only supported in `text-embedding-3` and later models

        if (
            model is not None
            and "text-embedding-3" not in model
            and "dimensions" in non_default_params.keys()
        ):
            raise UnsupportedParamsError(
                status_code=500,
                message="Setting dimensions is not supported for OpenAI `text-embedding-3` and later models. To drop it from the call, set `litellm.drop_params = True`.",
            )
        else:
            optional_params = non_default_params
    elif custom_llm_provider == "triton":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider=custom_llm_provider,
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.TritonEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )
    elif custom_llm_provider == "databricks":
        supported_params = get_supported_openai_params(
            model=model or "",
            custom_llm_provider="databricks",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.DatabricksEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )

    elif custom_llm_provider == "nvidia_nim":
        supported_params = get_supported_openai_params(
            model=model or "",
            custom_llm_provider="nvidia_nim",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.nvidiaNimEmbeddingConfig.map_openai_params(
            non_default_params=non_default_params, optional_params={}, kwargs=kwargs
        )
    elif custom_llm_provider == "vertex_ai" or custom_llm_provider == "gemini":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="vertex_ai",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        (
            optional_params,
            kwargs,
        ) = litellm.VertexAITextEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}, kwargs=kwargs
        )
    elif custom_llm_provider == "lm_studio":
        supported_params = (
            litellm.LmStudioEmbeddingConfig().get_supported_openai_params()
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.LmStudioEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
    elif custom_llm_provider == "bedrock":
        # if dimensions is in non_default_params -> pass it for model=bedrock/amazon.titan-embed-text-v2
        if "amazon.titan-embed-text-v1" in model:
            object: Any = litellm.AmazonTitanG1Config()
        elif "amazon.titan-embed-image-v1" in model:
            object = litellm.AmazonTitanMultimodalEmbeddingG1Config()
        elif "amazon.titan-embed-text-v2:0" in model:
            object = litellm.AmazonTitanV2Config()
        elif "cohere.embed-multilingual-v3" in model:
            object = litellm.BedrockCohereEmbeddingConfig()
        else:  # unmapped model
            supported_params = []
            _check_valid_arg(supported_params=supported_params)
            final_params = {**kwargs}
            return final_params

        supported_params = object.get_supported_openai_params()
        _check_valid_arg(supported_params=supported_params)
        optional_params = object.map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
    elif custom_llm_provider == "mistral":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="mistral",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.MistralEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
    elif custom_llm_provider == "jina_ai":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="jina_ai",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.JinaAIEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )
    elif custom_llm_provider == "voyage":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="voyage",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        if litellm.VoyageContextualEmbeddingConfig.is_contextualized_embeddings(model):
            optional_params = (
                litellm.VoyageContextualEmbeddingConfig().map_openai_params(
                    non_default_params=non_default_params,
                    optional_params={},
                    model=model,
                    drop_params=drop_params if drop_params is not None else False,
                )
            )
        else:
            optional_params = litellm.VoyageEmbeddingConfig().map_openai_params(
                non_default_params=non_default_params,
                optional_params={},
                model=model,
                drop_params=drop_params if drop_params is not None else False,
            )
    elif custom_llm_provider == "infinity":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="infinity",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.InfinityEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )
    elif custom_llm_provider == "fireworks_ai":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="fireworks_ai",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.FireworksAIEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}, model=model
        )
    elif custom_llm_provider == "sambanova":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="sambanova",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.SambaNovaEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )

    elif (
        custom_llm_provider != "openai"
        and custom_llm_provider != "azure"
        and custom_llm_provider not in litellm.openai_compatible_providers
    ):
        if len(non_default_params.keys()) > 0:
            if (
                litellm.drop_params is True or drop_params is True
            ):  # drop the unsupported non-default values
                keys = list(non_default_params.keys())
                for k in keys:
                    non_default_params.pop(k, None)
            else:
                raise UnsupportedParamsError(
                    status_code=500,
                    message=f"Setting {non_default_params} is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
                )
        else:
            optional_params = non_default_params
    else:
        optional_params = non_default_params

    final_params = add_provider_specific_params_to_optional_params(
        optional_params=optional_params,
        passed_params=passed_params,
        custom_llm_provider=custom_llm_provider,
        openai_params=list(DEFAULT_EMBEDDING_PARAM_VALUES.keys()),
        additional_drop_params=kwargs.get("additional_drop_params", None),
    )

    if "extra_body" in final_params and len(final_params["extra_body"]) == 0:
        final_params.pop("extra_body", None)

    return final_params


def _remove_additional_properties(schema):
    """
    clean out 'additionalProperties = False'. Causes vertexai/gemini OpenAI API Schema errors - https://github.com/langchain-ai/langchainjs/issues/5240

    Relevant Issues: https://github.com/BerriAI/litellm/issues/6136, https://github.com/BerriAI/litellm/issues/6088
    """
    if isinstance(schema, dict):
        # Remove the 'additionalProperties' key if it exists and is set to False
        if "additionalProperties" in schema and schema["additionalProperties"] is False:
            del schema["additionalProperties"]

        # Recursively process all dictionary values
        for key, value in schema.items():
            _remove_additional_properties(value)

    elif isinstance(schema, list):
        # Recursively process all items in the list
        for item in schema:
            _remove_additional_properties(item)

    return schema


def _remove_strict_from_schema(schema):
    """
    Relevant Issues: https://github.com/BerriAI/litellm/issues/6136, https://github.com/BerriAI/litellm/issues/6088
    """
    if isinstance(schema, dict):
        # Remove the 'additionalProperties' key if it exists and is set to False
        if "strict" in schema:
            del schema["strict"]

        # Recursively process all dictionary values
        for key, value in schema.items():
            _remove_strict_from_schema(value)

    elif isinstance(schema, list):
        # Recursively process all items in the list
        for item in schema:
            _remove_strict_from_schema(item)

    return schema


def _remove_json_schema_refs(schema, max_depth=10):
    """
    Remove JSON schema reference fields like '$id' and '$schema' that can cause issues with some providers.

    These fields are used for schema validation but can cause problems when the schema references
    are not accessible to the provider's validation system.

    Args:
        schema: The schema object to clean (dict, list, or other)
        max_depth: Maximum recursion depth to prevent infinite loops (default: 10)

    Relevant Issues: Mistral API grammar validation fails when schema contains $id and $schema references
    """
    if max_depth <= 0:
        return schema

    if isinstance(schema, dict):
        # Remove JSON schema reference fields
        schema.pop("$id", None)
        schema.pop("$schema", None)

        # Recursively process all dictionary values
        for key, value in schema.items():
            _remove_json_schema_refs(value, max_depth - 1)

    elif isinstance(schema, list):
        # Recursively process all items in the list
        for item in schema:
            _remove_json_schema_refs(item, max_depth - 1)

    return schema


def _remove_unsupported_params(
    non_default_params: dict, supported_openai_params: Optional[List[str]]
) -> dict:
    """
    Remove unsupported params from non_default_params
    """
    remove_keys = []
    if supported_openai_params is None:
        return {}  # no supported params, so no optional openai params to send
    for param in non_default_params.keys():
        if param not in supported_openai_params:
            remove_keys.append(param)
    for key in remove_keys:
        non_default_params.pop(key, None)
    return non_default_params


class PreProcessNonDefaultParams:
    @staticmethod
    def base_pre_process_non_default_params(
        passed_params: dict,
        special_params: dict,
        custom_llm_provider: str,
        additional_drop_params: Optional[List[str]],
        default_param_values: dict,
        additional_endpoint_specific_params: List[str],
    ) -> dict:
        for k, v in special_params.items():
            if k.startswith("aws_") and (
                custom_llm_provider != "bedrock"
                and not custom_llm_provider.startswith("sagemaker")
            ):  # allow dynamically setting boto3 init logic
                continue
            elif k == "hf_model_name" and custom_llm_provider != "sagemaker":
                continue
            elif (
                k.startswith("vertex_")
                and custom_llm_provider != "vertex_ai"
                and custom_llm_provider != "vertex_ai_beta"
            ):  # allow dynamically setting vertex ai init logic
                continue
            passed_params[k] = v

        # filter out those parameters that were passed with non-default values
        non_default_params = {
            k: v
            for k, v in passed_params.items()
            if (
                k != "model"
                and k != "custom_llm_provider"
                and k != "api_version"
                and k != "drop_params"
                and k != "allowed_openai_params"
                and k != "additional_drop_params"
                and k not in additional_endpoint_specific_params
                and k in default_param_values
                and v != default_param_values[k]
                and _should_drop_param(
                    k=k, additional_drop_params=additional_drop_params
                )
                is False
            )
        }

        return non_default_params

    @staticmethod
    def embedding_pre_process_non_default_params(
        passed_params: dict,
        special_params: dict,
        custom_llm_provider: str,
        additional_drop_params: Optional[List[str]],
        model: str,
        remove_sensitive_keys: bool = False,
        add_provider_specific_params: bool = False,
    ) -> dict:
        non_default_params = (
            PreProcessNonDefaultParams.base_pre_process_non_default_params(
                passed_params=passed_params,
                special_params=special_params,
                custom_llm_provider=custom_llm_provider,
                additional_drop_params=additional_drop_params,
                default_param_values={k: None for k in OPENAI_EMBEDDING_PARAMS},
                additional_endpoint_specific_params=["input"],
            )
        )

        return non_default_params


def pre_process_non_default_params(
    passed_params: dict,
    special_params: dict,
    custom_llm_provider: str,
    additional_drop_params: Optional[List[str]],
    model: str,
    remove_sensitive_keys: bool = False,
    add_provider_specific_params: bool = False,
    provider_config: Optional[BaseConfig] = None,
) -> dict:
    """
    Pre-process non-default params to a standardized format
    """
    # retrieve all parameters passed to the function

    non_default_params = PreProcessNonDefaultParams.base_pre_process_non_default_params(
        passed_params=passed_params,
        special_params=special_params,
        custom_llm_provider=custom_llm_provider,
        additional_drop_params=additional_drop_params,
        default_param_values=DEFAULT_CHAT_COMPLETION_PARAM_VALUES,
        additional_endpoint_specific_params=["messages"],
    )

    if "response_format" in non_default_params:
        if provider_config is not None:
            non_default_params["response_format"] = (
                provider_config.get_json_schema_from_pydantic_object(
                    response_format=non_default_params["response_format"]
                )
            )
        else:
            non_default_params["response_format"] = type_to_response_format_param(
                response_format=non_default_params["response_format"]
            )

    if "tools" in non_default_params and isinstance(
        non_default_params, list
    ):  # fixes https://github.com/BerriAI/litellm/issues/4933
        tools = non_default_params["tools"]
        for (
            tool
        ) in (
            tools
        ):  # clean out 'additionalProperties = False'. Causes vertexai/gemini OpenAI API Schema errors - https://github.com/langchain-ai/langchainjs/issues/5240
            tool_function = tool.get("function", {})
            parameters = tool_function.get("parameters", None)
            if parameters is not None:
                new_parameters = copy.deepcopy(parameters)
                if (
                    "additionalProperties" in new_parameters
                    and new_parameters["additionalProperties"] is False
                ):
                    new_parameters.pop("additionalProperties", None)
                tool_function["parameters"] = new_parameters

    if add_provider_specific_params:
        non_default_params = add_provider_specific_params_to_optional_params(
            optional_params=non_default_params,
            passed_params=passed_params,
            custom_llm_provider=custom_llm_provider,
            openai_params=list(DEFAULT_CHAT_COMPLETION_PARAM_VALUES.keys()),
            additional_drop_params=additional_drop_params,
        )

    if remove_sensitive_keys:
        non_default_params = remove_sensitive_keys_from_dict(non_default_params)
    return non_default_params


def remove_sensitive_keys_from_dict(d: dict) -> dict:
    """
    Remove sensitive keys from a dictionary
    """
    sensitive_key_phrases = ["key", "secret", "access", "credential"]
    remove_keys = []
    for key in d.keys():
        if any(phrase in key.lower() for phrase in sensitive_key_phrases):
            remove_keys.append(key)
    for key in remove_keys:
        d.pop(key)
    return d


def pre_process_optional_params(
    passed_params: dict, non_default_params: dict, custom_llm_provider: str
) -> dict:
    """For .completion(), preprocess optional params"""
    optional_params: Dict = {}

    common_auth_dict = litellm.common_cloud_provider_auth_params
    if custom_llm_provider in common_auth_dict["providers"]:
        """
        Check if params = ["project", "region_name", "token"]
        and correctly translate for = ["azure", "vertex_ai", "watsonx", "aws"]
        """
        if custom_llm_provider == "azure":
            optional_params = litellm.AzureOpenAIConfig().map_special_auth_params(
                non_default_params=passed_params, optional_params=optional_params
            )
        elif custom_llm_provider == "bedrock":
            optional_params = (
                litellm.AmazonBedrockGlobalConfig().map_special_auth_params(
                    non_default_params=passed_params, optional_params=optional_params
                )
            )
        elif (
            custom_llm_provider == "vertex_ai"
            or custom_llm_provider == "vertex_ai_beta"
        ):
            optional_params = litellm.VertexAIConfig().map_special_auth_params(
                non_default_params=passed_params, optional_params=optional_params
            )
        elif custom_llm_provider == "watsonx":
            optional_params = litellm.IBMWatsonXAIConfig().map_special_auth_params(
                non_default_params=passed_params, optional_params=optional_params
            )

    ## raise exception if function calling passed in for a provider that doesn't support it
    if (
        "functions" in non_default_params
        or "function_call" in non_default_params
        or "tools" in non_default_params
    ):
        if (
            custom_llm_provider == "ollama"
            and custom_llm_provider != "text-completion-openai"
            and custom_llm_provider != "azure"
            and custom_llm_provider != "vertex_ai"
            and custom_llm_provider != "anyscale"
            and custom_llm_provider != "together_ai"
            and custom_llm_provider != "groq"
            and custom_llm_provider != "nvidia_nim"
            and custom_llm_provider != "cerebras"
            and custom_llm_provider != "xai"
            and custom_llm_provider != "ai21_chat"
            and custom_llm_provider != "volcengine"
            and custom_llm_provider != "deepseek"
            and custom_llm_provider != "codestral"
            and custom_llm_provider != "mistral"
            and custom_llm_provider != "anthropic"
            and custom_llm_provider != "cohere_chat"
            and custom_llm_provider != "cohere"
            and custom_llm_provider != "bedrock"
            and custom_llm_provider != "ollama_chat"
            and custom_llm_provider != "openrouter"
            and custom_llm_provider != "vercel_ai_gateway"
            and custom_llm_provider != "nebius"
            and custom_llm_provider not in litellm.openai_compatible_providers
        ):
            if custom_llm_provider == "ollama":
                # ollama actually supports json output
                optional_params["format"] = "json"
                litellm.add_function_to_prompt = (
                    True  # so that main.py adds the function call to the prompt
                )
                if "tools" in non_default_params:
                    optional_params["functions_unsupported_model"] = (
                        non_default_params.pop("tools")
                    )
                    non_default_params.pop(
                        "tool_choice", None
                    )  # causes ollama requests to hang
                elif "functions" in non_default_params:
                    optional_params["functions_unsupported_model"] = (
                        non_default_params.pop("functions")
                    )
            elif (
                litellm.add_function_to_prompt
            ):  # if user opts to add it to prompt instead
                optional_params["functions_unsupported_model"] = non_default_params.pop(
                    "tools", non_default_params.pop("functions", None)
                )
            else:
                raise UnsupportedParamsError(
                    status_code=500,
                    message=f"Function calling is not supported by {custom_llm_provider}.",
                )

    return optional_params


def get_optional_params(  # noqa: PLR0915
    # use the openai defaults
    # https://platform.openai.com/docs/api-reference/chat/create
    model: str,
    functions=None,
    function_call=None,
    temperature=None,
    top_p=None,
    n=None,
    stream=False,
    stream_options=None,
    stop=None,
    max_tokens=None,
    max_completion_tokens=None,
    modalities=None,
    prediction=None,
    audio=None,
    presence_penalty=None,
    frequency_penalty=None,
    logit_bias=None,
    user=None,
    custom_llm_provider="",
    response_format=None,
    seed=None,
    tools=None,
    tool_choice=None,
    max_retries=None,
    logprobs=None,
    top_logprobs=None,
    extra_headers=None,
    api_version=None,
    parallel_tool_calls=None,
    drop_params=None,
    allowed_openai_params: Optional[List[str]] = None,
    reasoning_effort=None,
    additional_drop_params=None,
    messages: Optional[List[AllMessageValues]] = None,
    thinking: Optional[AnthropicThinkingParam] = None,
    web_search_options: Optional[OpenAIWebSearchOptions] = None,
    safety_identifier: Optional[str] = None,
    **kwargs,
):
    passed_params = locals().copy()
    special_params = passed_params.pop("kwargs")
    non_default_params = pre_process_non_default_params(
        passed_params=passed_params,
        special_params=special_params,
        custom_llm_provider=custom_llm_provider,
        additional_drop_params=additional_drop_params,
        model=model,
    )
    optional_params = pre_process_optional_params(
        passed_params=passed_params,
        non_default_params=non_default_params,
        custom_llm_provider=custom_llm_provider,
    )
    provider_config: Optional[BaseConfig] = None
    if custom_llm_provider is not None and custom_llm_provider in [
        provider.value for provider in LlmProviders
    ]:
        provider_config = ProviderConfigManager.get_provider_chat_config(
            model=model, provider=LlmProviders(custom_llm_provider)
        )

    def _check_valid_arg(supported_params: List[str]):
        """
        Check if the params passed to completion() are supported by the provider

        Args:
            supported_params: List[str] - supported params from the litellm config
        """
        verbose_logger.info(
            f"\nLiteLLM completion() model= {model}; provider = {custom_llm_provider}"
        )
        verbose_logger.debug(
            f"\nLiteLLM: Params passed to completion() {passed_params}"
        )
        verbose_logger.debug(
            f"\nLiteLLM: Non-Default params passed to completion() {non_default_params}"
        )
        unsupported_params = {}
        for k in non_default_params.keys():
            if k not in supported_params:
                if k == "user" or k == "stream_options" or k == "stream":
                    continue
                if k == "n" and n == 1:  # langchain sends n=1 as a default value
                    continue  # skip this param
                if (
                    k == "max_retries"
                ):  # TODO: This is a patch. We support max retries for OpenAI, Azure. For non OpenAI LLMs we need to add support for max retries
                    continue  # skip this param
                # Always keeps this in elif code blocks
                else:
                    unsupported_params[k] = non_default_params[k]

        if unsupported_params:
            if litellm.drop_params is True or (
                drop_params is not None and drop_params is True
            ):
                for k in unsupported_params.keys():
                    non_default_params.pop(k, None)
            else:
                raise UnsupportedParamsError(
                    status_code=500,
                    message=f"{custom_llm_provider} does not support parameters: {list(unsupported_params.keys())}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. \n If you want to use these params dynamically send allowed_openai_params={list(unsupported_params.keys())} in your request.",
                )

    supported_params = get_supported_openai_params(
        model=model, custom_llm_provider=custom_llm_provider
    )
    if supported_params is None:
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider="openai"
        )

    supported_params = supported_params or []
    allowed_openai_params = allowed_openai_params or []
    supported_params.extend(allowed_openai_params)

    _check_valid_arg(
        supported_params=supported_params or [],
    )
    ## raise exception if provider doesn't support passed in param
    if custom_llm_provider == "anthropic":
        ## check if unsupported param passed in
        optional_params = litellm.AnthropicConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "anthropic_text":
        optional_params = litellm.AnthropicTextConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
        optional_params = litellm.AnthropicTextConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )

    elif custom_llm_provider == "cohere":
        ## check if unsupported param passed in
        # handle cohere params
        optional_params = litellm.CohereConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "cohere_chat":
        # handle cohere params
        optional_params = litellm.CohereChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "triton":
        optional_params = litellm.TritonConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )

    elif custom_llm_provider == "maritalk":
        optional_params = litellm.MaritalkConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "replicate":
        optional_params = litellm.ReplicateConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "predibase":
        optional_params = litellm.PredibaseConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "huggingface":
        optional_params = litellm.HuggingFaceChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "together_ai":
        optional_params = litellm.TogetherAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "vertex_ai" and (
        model in litellm.vertex_chat_models
        or model in litellm.vertex_code_chat_models
        or model in litellm.vertex_text_models
        or model in litellm.vertex_code_text_models
        or model in litellm.vertex_language_models
        or model in litellm.vertex_vision_models
    ):
        optional_params = litellm.VertexGeminiConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )

    elif custom_llm_provider == "gemini":
        optional_params = litellm.GoogleAIStudioGeminiConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "vertex_ai_beta" or (
        custom_llm_provider == "vertex_ai" and "gemini" in model
    ):
        optional_params = litellm.VertexGeminiConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif litellm.VertexAIAnthropicConfig.is_supported_model(
        model=model, custom_llm_provider=custom_llm_provider
    ):
        optional_params = litellm.VertexAIAnthropicConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "vertex_ai":
        if model in litellm.vertex_mistral_models:
            if "codestral" in model:
                optional_params = (
                    litellm.CodestralTextCompletionConfig().map_openai_params(
                        model=model,
                        non_default_params=non_default_params,
                        optional_params=optional_params,
                        drop_params=(
                            drop_params
                            if drop_params is not None and isinstance(drop_params, bool)
                            else False
                        ),
                    )
                )
            else:
                optional_params = litellm.MistralConfig().map_openai_params(
                    model=model,
                    non_default_params=non_default_params,
                    optional_params=optional_params,
                    drop_params=(
                        drop_params
                        if drop_params is not None and isinstance(drop_params, bool)
                        else False
                    ),
                )
        elif model in litellm.vertex_ai_ai21_models:
            optional_params = litellm.VertexAIAi21Config().map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )
        elif provider_config is not None:
            optional_params = provider_config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )
        else:  # use generic openai-like param mapping
            optional_params = litellm.VertexAILlama3Config().map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )

    elif custom_llm_provider == "sagemaker":
        # temperature, top_p, n, stream, stop, max_tokens, n, presence_penalty default to None
        optional_params = litellm.SagemakerConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "bedrock":
        bedrock_route = BedrockModelInfo.get_bedrock_route(model)
        bedrock_base_model = BedrockModelInfo.get_base_model(model)
        if bedrock_route == "converse" or bedrock_route == "converse_like":
            optional_params = litellm.AmazonConverseConfig().map_openai_params(
                model=model,
                non_default_params=non_default_params,
                optional_params=optional_params,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )

        elif "anthropic" in bedrock_base_model and bedrock_route == "invoke":
            if bedrock_base_model.startswith("anthropic.claude-3"):
                optional_params = (
                    litellm.AmazonAnthropicClaudeConfig().map_openai_params(
                        non_default_params=non_default_params,
                        optional_params=optional_params,
                        model=model,
                        drop_params=(
                            drop_params
                            if drop_params is not None and isinstance(drop_params, bool)
                            else False
                        ),
                    )
                )

            else:
                optional_params = litellm.AmazonAnthropicConfig().map_openai_params(
                    non_default_params=non_default_params,
                    optional_params=optional_params,
                    model=model,
                    drop_params=(
                        drop_params
                        if drop_params is not None and isinstance(drop_params, bool)
                        else False
                    ),
                )
        elif provider_config is not None:
            optional_params = provider_config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )
    elif custom_llm_provider == "cloudflare":
        optional_params = litellm.CloudflareChatConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "ollama":
        optional_params = litellm.OllamaConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "ollama_chat":
        optional_params = litellm.OllamaChatConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "nlp_cloud":
        optional_params = litellm.NLPCloudConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )

    elif custom_llm_provider == "petals":
        optional_params = litellm.PetalsConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "deepinfra":
        optional_params = litellm.DeepInfraConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "perplexity" and provider_config is not None:
        optional_params = provider_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "mistral" or custom_llm_provider == "codestral":
        optional_params = litellm.MistralConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "text-completion-codestral":
        optional_params = litellm.CodestralTextCompletionConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )

    elif custom_llm_provider == "databricks":
        optional_params = litellm.DatabricksConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "nvidia_nim":
        optional_params = litellm.NvidiaNimConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "cerebras":
        optional_params = litellm.CerebrasConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "xai":
        optional_params = litellm.XAIChatConfig().map_openai_params(
            model=model,
            non_default_params=non_default_params,
            optional_params=optional_params,
        )
    elif custom_llm_provider == "ai21_chat" or custom_llm_provider == "ai21":
        optional_params = litellm.AI21ChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "fireworks_ai":
        optional_params = litellm.FireworksAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "volcengine":
        optional_params = litellm.VolcEngineConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "hosted_vllm":
        optional_params = litellm.HostedVLLMChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "vllm":
        optional_params = litellm.VLLMConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "groq":
        optional_params = litellm.GroqChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "deepseek":
        optional_params = litellm.OpenAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "openrouter":
        optional_params = litellm.OpenrouterConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "watsonx":
        optional_params = litellm.IBMWatsonXChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
        # WatsonX-text param check
        for param in passed_params.keys():
            if litellm.IBMWatsonXAIConfig().is_watsonx_text_param(param):
                raise ValueError(
                    f"LiteLLM now defaults to Watsonx's `/text/chat` endpoint. Please use the `watsonx_text` provider instead, to call the `/text/generation` endpoint. Param: {param}"
                )
    elif custom_llm_provider == "watsonx_text":
        optional_params = litellm.IBMWatsonXAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "openai":
        optional_params = litellm.OpenAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "nebius":
        optional_params = litellm.NebiusConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    elif custom_llm_provider == "azure":
        if litellm.AzureOpenAIO1Config().is_o_series_model(model=model):
            optional_params = litellm.AzureOpenAIO1Config().map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )
        elif litellm.AzureOpenAIGPT5Config.is_model_gpt_5_model(model=model):
            optional_params = litellm.AzureOpenAIGPT5Config().map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )
        else:
            verbose_logger.debug(
                "Azure optional params - api_version: api_version={}, litellm.api_version={}, os.environ['AZURE_API_VERSION']={}".format(
                    api_version, litellm.api_version, get_secret("AZURE_API_VERSION")
                )
            )
            api_version = (
                api_version
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
                or litellm.AZURE_DEFAULT_API_VERSION
            )
            optional_params = litellm.AzureOpenAIConfig().map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                api_version=api_version,  # type: ignore
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
            )
    elif provider_config is not None:
        optional_params = provider_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    else:  # assume passing in params for openai-like api
        optional_params = litellm.OpenAILikeChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=(
                drop_params
                if drop_params is not None and isinstance(drop_params, bool)
                else False
            ),
        )
    # if user passed in non-default kwargs for specific providers/models, pass them along
    optional_params = add_provider_specific_params_to_optional_params(
        optional_params=optional_params,
        passed_params=passed_params,
        custom_llm_provider=custom_llm_provider,
        openai_params=list(DEFAULT_CHAT_COMPLETION_PARAM_VALUES.keys()),
        additional_drop_params=additional_drop_params,
    )
    print_verbose(f"Final returned optional params: {optional_params}")
    optional_params = _apply_openai_param_overrides(
        optional_params=optional_params,
        non_default_params=non_default_params,
        allowed_openai_params=allowed_openai_params,
    )
    return optional_params


def add_provider_specific_params_to_optional_params(
    optional_params: dict,
    passed_params: dict,
    custom_llm_provider: str,
    openai_params: List[str],
    additional_drop_params: Optional[list] = None,
) -> dict:
    """
    Add provider specific params to optional_params
    """
    if (
        custom_llm_provider
        in ["openai", "azure", "text-completion-openai"]
        + litellm.openai_compatible_providers
    ):
        # for openai, azure we should pass the extra/passed params within `extra_body` https://github.com/openai/openai-python/blob/ac33853ba10d13ac149b1fa3ca6dba7d613065c9/src/openai/resources/models.py#L46
        if (
            _should_drop_param(
                k="extra_body", additional_drop_params=additional_drop_params
            )
            is False
        ):
            extra_body = passed_params.pop("extra_body", {})
            for k in passed_params.keys():
                if k not in openai_params and passed_params[k] is not None:
                    extra_body[k] = passed_params[k]
            optional_params.setdefault("extra_body", {})
            initial_extra_body = {
                **optional_params["extra_body"],
                **extra_body,
            }

            if additional_drop_params is not None:
                processed_extra_body = {
                    k: v
                    for k, v in initial_extra_body.items()
                    if k not in additional_drop_params
                }
            else:
                processed_extra_body = initial_extra_body

            optional_params["extra_body"] = _ensure_extra_body_is_safe(
                extra_body=processed_extra_body
            )
    else:
        for k in passed_params.keys():
            if k not in openai_params and passed_params[k] is not None:
                optional_params[k] = passed_params[k]
    return optional_params


def _apply_openai_param_overrides(
    optional_params: dict, non_default_params: dict, allowed_openai_params: list
):
    """
    If user passes in allowed_openai_params, apply them to optional_params

    These params will get passed as is to the LLM API since the user opted in to passing them in the request
    """
    if allowed_openai_params:
        for param in allowed_openai_params:
            if param not in optional_params:
                optional_params[param] = non_default_params.pop(param, None)
    return optional_params


def get_non_default_params(passed_params: dict) -> dict:
    # filter out those parameters that were passed with non-default values
    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (
            k != "model"
            and k != "custom_llm_provider"
            and k in DEFAULT_CHAT_COMPLETION_PARAM_VALUES
            and v != DEFAULT_CHAT_COMPLETION_PARAM_VALUES[k]
        )
    }

    return non_default_params


def calculate_max_parallel_requests(
    max_parallel_requests: Optional[int],
    rpm: Optional[int],
    tpm: Optional[int],
    default_max_parallel_requests: Optional[int],
) -> Optional[int]:
    """
    Returns the max parallel requests to send to a deployment.

    Used in semaphore for async requests on router.

    Parameters:
    - max_parallel_requests - Optional[int] - max_parallel_requests allowed for that deployment
    - rpm - Optional[int] - requests per minute allowed for that deployment
    - tpm - Optional[int] - tokens per minute allowed for that deployment
    - default_max_parallel_requests - Optional[int] - default_max_parallel_requests allowed for any deployment

    Returns:
    - int or None (if all params are None)

    Order:
    max_parallel_requests > rpm > tpm / 6 (azure formula) > default max_parallel_requests

    Azure RPM formula:
    6 rpm per 1000 TPM
    https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits


    """
    if max_parallel_requests is not None:
        return max_parallel_requests
    elif rpm is not None:
        return rpm
    elif tpm is not None:
        calculated_rpm = int(tpm / 1000 / 6)
        if calculated_rpm == 0:
            calculated_rpm = 1
        return calculated_rpm
    elif default_max_parallel_requests is not None:
        return default_max_parallel_requests
    return None


def _get_order_filtered_deployments(healthy_deployments: List[Dict]) -> List:
    min_order = min(
        (
            deployment["litellm_params"]["order"]
            for deployment in healthy_deployments
            if "order" in deployment["litellm_params"]
        ),
        default=None,
    )

    if min_order is not None:
        filtered_deployments = [
            deployment
            for deployment in healthy_deployments
            if deployment["litellm_params"].get("order") == min_order
        ]

        return filtered_deployments
    return healthy_deployments


def _get_model_region(
    custom_llm_provider: str, litellm_params: LiteLLM_Params
) -> Optional[str]:
    """
    Return the region for a model, for a given provider
    """
    if custom_llm_provider == "vertex_ai":
        # check 'vertex_location'
        vertex_ai_location = (
            litellm_params.vertex_location
            or litellm.vertex_location
            or get_secret("VERTEXAI_LOCATION")
            or get_secret("VERTEX_LOCATION")
        )
        if vertex_ai_location is not None and isinstance(vertex_ai_location, str):
            return vertex_ai_location
    elif custom_llm_provider == "bedrock":
        aws_region_name = litellm_params.aws_region_name
        if aws_region_name is not None:
            return aws_region_name
    elif custom_llm_provider == "watsonx":
        watsonx_region_name = litellm_params.watsonx_region_name
        if watsonx_region_name is not None:
            return watsonx_region_name
    return litellm_params.region_name


def _infer_model_region(litellm_params: LiteLLM_Params) -> Optional[AllowedModelRegion]:
    """
    Infer if a model is in the EU or US region

    Returns:
    - str (region) - "eu" or "us"
    - None (if region not found)
    """
    model, custom_llm_provider, _, _ = litellm.get_llm_provider(
        model=litellm_params.model, litellm_params=litellm_params
    )

    model_region = _get_model_region(
        custom_llm_provider=custom_llm_provider, litellm_params=litellm_params
    )

    if model_region is None:
        verbose_logger.debug(
            "Cannot infer model region for model: {}".format(litellm_params.model)
        )
        return None

    if custom_llm_provider == "azure":
        eu_regions = litellm.AzureOpenAIConfig().get_eu_regions()
        us_regions = litellm.AzureOpenAIConfig().get_us_regions()
    elif custom_llm_provider == "vertex_ai":
        eu_regions = litellm.VertexAIConfig().get_eu_regions()
        us_regions = litellm.VertexAIConfig().get_us_regions()
    elif custom_llm_provider == "bedrock":
        eu_regions = litellm.AmazonBedrockGlobalConfig().get_eu_regions()
        us_regions = litellm.AmazonBedrockGlobalConfig().get_us_regions()
    elif custom_llm_provider == "watsonx":
        eu_regions = litellm.IBMWatsonXAIConfig().get_eu_regions()
        us_regions = litellm.IBMWatsonXAIConfig().get_us_regions()
    else:
        eu_regions = []
        us_regions = []

    for region in eu_regions:
        if region in model_region.lower():
            return "eu"
    for region in us_regions:
        if region in model_region.lower():
            return "us"
    return None


def _is_region_eu(litellm_params: LiteLLM_Params) -> bool:
    """
    Return true/false if a deployment is in the EU
    """
    if litellm_params.region_name == "eu":
        return True

    ## Else - try and infer from model region
    model_region = _infer_model_region(litellm_params=litellm_params)
    if model_region is not None and model_region == "eu":
        return True
    return False


def _is_region_us(litellm_params: LiteLLM_Params) -> bool:
    """
    Return true/false if a deployment is in the US
    """
    if litellm_params.region_name == "us":
        return True

    ## Else - try and infer from model region
    model_region = _infer_model_region(litellm_params=litellm_params)
    if model_region is not None and model_region == "us":
        return True
    return False


def is_region_allowed(
    litellm_params: LiteLLM_Params, allowed_model_region: str
) -> bool:
    """
    Return true/false if a deployment is in the EU
    """
    if litellm_params.region_name == allowed_model_region:
        return True
    return False


def get_model_region(
    litellm_params: LiteLLM_Params, mode: Optional[str]
) -> Optional[str]:
    """
    Pass the litellm params for an azure model, and get back the region
    """
    if (
        "azure" in litellm_params.model
        and isinstance(litellm_params.api_key, str)
        and isinstance(litellm_params.api_base, str)
    ):
        _model = litellm_params.model.replace("azure/", "")
        response: dict = litellm.AzureChatCompletion().get_headers(
            model=_model,
            api_key=litellm_params.api_key,
            api_base=litellm_params.api_base,
            api_version=litellm_params.api_version or litellm.AZURE_DEFAULT_API_VERSION,
            timeout=10,
            mode=mode or "chat",
        )

        region: Optional[str] = response.get("x-ms-region", None)
        return region
    return None


def get_first_chars_messages(kwargs: dict) -> str:
    try:
        _messages = kwargs.get("messages")
        _messages = str(_messages)[:100]
        return _messages
    except Exception:
        return ""


def _count_characters(text: str) -> int:
    # Remove white spaces and count characters
    filtered_text = "".join(char for char in text if not char.isspace())
    return len(filtered_text)


def get_response_string(response_obj: Union[ModelResponse, ModelResponseStream]) -> str:
    _choices: Union[List[Union[Choices, StreamingChoices]], List[StreamingChoices]] = (
        response_obj.choices
    )

    response_str = ""
    for choice in _choices:
        if isinstance(choice, Choices):
            if choice.message.content is not None:
                response_str += choice.message.content
        elif isinstance(choice, StreamingChoices):
            if choice.delta.content is not None:
                response_str += choice.delta.content

    return response_str


def get_api_key(llm_provider: str, dynamic_api_key: Optional[str]):
    api_key = dynamic_api_key or litellm.api_key
    # openai
    if llm_provider == "openai" or llm_provider == "text-completion-openai":
        api_key = api_key or litellm.openai_key or get_secret("OPENAI_API_KEY")
    # anthropic
    elif llm_provider == "anthropic" or llm_provider == "anthropic_text":
        api_key = api_key or litellm.anthropic_key or get_secret("ANTHROPIC_API_KEY")
    # ai21
    elif llm_provider == "ai21":
        api_key = api_key or litellm.ai21_key or get_secret("AI211_API_KEY")
    # aleph_alpha
    elif llm_provider == "aleph_alpha":
        api_key = (
            api_key or litellm.aleph_alpha_key or get_secret("ALEPH_ALPHA_API_KEY")
        )
    # baseten
    elif llm_provider == "baseten":
        api_key = api_key or litellm.baseten_key or get_secret("BASETEN_API_KEY")
    # cohere
    elif llm_provider == "cohere" or llm_provider == "cohere_chat":
        api_key = api_key or litellm.cohere_key or get_secret("COHERE_API_KEY")
    # huggingface
    elif llm_provider == "huggingface":
        api_key = (
            api_key or litellm.huggingface_key or get_secret("HUGGINGFACE_API_KEY")
        )
    # nlp_cloud
    elif llm_provider == "nlp_cloud":
        api_key = api_key or litellm.nlp_cloud_key or get_secret("NLP_CLOUD_API_KEY")
    # replicate
    elif llm_provider == "replicate":
        api_key = api_key or litellm.replicate_key or get_secret("REPLICATE_API_KEY")
    # together_ai
    elif llm_provider == "together_ai":
        api_key = (
            api_key
            or litellm.togetherai_api_key
            or get_secret("TOGETHERAI_API_KEY")
            or get_secret("TOGETHER_AI_TOKEN")
        )
    # nebius
    elif llm_provider == "nebius":
        api_key = api_key or litellm.nebius_key or get_secret("NEBIUS_API_KEY")
    return api_key


def get_utc_datetime():
    import datetime as dt
    from datetime import datetime

    if hasattr(dt, "UTC"):
        return datetime.now(dt.UTC)  # type: ignore
    else:
        return datetime.utcnow()  # type: ignore


def get_max_tokens(model: str) -> Optional[int]:
    """
    Get the maximum number of output tokens allowed for a given model.

    Parameters:
    model (str): The name of the model.

    Returns:
        int: The maximum number of tokens allowed for the given model.

    Raises:
        Exception: If the model is not mapped yet.

    Example:
        >>> get_max_tokens("gpt-4")
        8192
    """

    def _get_max_position_embeddings(model_name):
        # Construct the URL for the config.json file
        config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"
        try:
            # Make the HTTP request to get the raw JSON file
            response = litellm.module_level_client.get(config_url)
            response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)

            # Parse the JSON response
            config_json = response.json()
            # Extract and return the max_position_embeddings
            max_position_embeddings = config_json.get("max_position_embeddings")
            if max_position_embeddings is not None:
                return max_position_embeddings
            else:
                return None
        except Exception:
            return None

    try:
        if model in litellm.model_cost:
            if "max_output_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_output_tokens"]
            elif "max_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_tokens"]
        model, custom_llm_provider, _, _ = get_llm_provider(model=model)
        if custom_llm_provider == "huggingface":
            max_tokens = _get_max_position_embeddings(model_name=model)
            return max_tokens
        if model in litellm.model_cost:  # check if extracted model is in model_list
            if "max_output_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_output_tokens"]
            elif "max_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_tokens"]
        else:
            raise Exception()
        return None
    except Exception:
        raise Exception(
            f"Model {model} isn't mapped yet. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
        )


def _strip_stable_vertex_version(model_name) -> str:
    return re.sub(r"-\d+$", "", model_name)


def _get_base_bedrock_model(model_name) -> str:
    """
    Get the base model from the given model name.

    Handle model names like - "us.meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
    AND "meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
    """
    from litellm.llms.bedrock.common_utils import BedrockModelInfo

    return BedrockModelInfo.get_base_model(model_name)


def _strip_openai_finetune_model_name(model_name: str) -> str:
    """
    Strips the organization, custom suffix, and ID from an OpenAI fine-tuned model name.

    input: ft:gpt-3.5-turbo:my-org:custom_suffix:id
    output: ft:gpt-3.5-turbo

    Args:
    model_name (str): The full model name

    Returns:
    str: The stripped model name
    """
    return re.sub(r"(:[^:]+){3}$", "", model_name)


def _strip_model_name(model: str, custom_llm_provider: Optional[str]) -> str:
    if custom_llm_provider and custom_llm_provider in ["bedrock", "bedrock_converse"]:
        stripped_bedrock_model = _get_base_bedrock_model(model_name=model)
        return stripped_bedrock_model
    elif custom_llm_provider and (
        custom_llm_provider == "vertex_ai" or custom_llm_provider == "gemini"
    ):
        strip_version = _strip_stable_vertex_version(model_name=model)
        return strip_version
    elif custom_llm_provider and (custom_llm_provider == "databricks"):
        strip_version = _strip_stable_vertex_version(model_name=model)
        return strip_version
    elif "ft:" in model:
        strip_finetune = _strip_openai_finetune_model_name(model_name=model)
        return strip_finetune
    else:
        return model


def _get_model_info_from_model_cost(key: str) -> dict:
    return litellm.model_cost[key]


def _check_provider_match(model_info: dict, custom_llm_provider: Optional[str]) -> bool:
    """
    Check if the model info provider matches the custom provider.
    """
    if custom_llm_provider and (
        "litellm_provider" in model_info
        and model_info["litellm_provider"] != custom_llm_provider
    ):
        if custom_llm_provider == "vertex_ai" and model_info[
            "litellm_provider"
        ].startswith("vertex_ai"):
            return True
        elif custom_llm_provider == "fireworks_ai" and model_info[
            "litellm_provider"
        ].startswith("fireworks_ai"):
            return True
        elif custom_llm_provider.startswith("bedrock") and model_info[
            "litellm_provider"
        ].startswith("bedrock"):
            return True
        elif (
            custom_llm_provider == "litellm_proxy"
        ):  # litellm_proxy is a special case, it's not a provider, it's a proxy for the provider
            return True
        else:
            return False

    return True


from typing import TypedDict


class PotentialModelNamesAndCustomLLMProvider(TypedDict):
    split_model: str
    combined_model_name: str
    stripped_model_name: str
    combined_stripped_model_name: str
    custom_llm_provider: str


def _get_potential_model_names(
    model: str, custom_llm_provider: Optional[str]
) -> PotentialModelNamesAndCustomLLMProvider:
    if custom_llm_provider is None:
        # Get custom_llm_provider
        try:
            split_model, custom_llm_provider, _, _ = get_llm_provider(model=model)
        except Exception:
            split_model = model
        combined_model_name = model
        stripped_model_name = _strip_model_name(
            model=model, custom_llm_provider=custom_llm_provider
        )
        combined_stripped_model_name = stripped_model_name
    elif custom_llm_provider and model.startswith(
        custom_llm_provider + "/"
    ):  # handle case where custom_llm_provider is provided and model starts with custom_llm_provider
        split_model = model.split("/", 1)[1]
        combined_model_name = model
        stripped_model_name = _strip_model_name(
            model=split_model, custom_llm_provider=custom_llm_provider
        )
        combined_stripped_model_name = "{}/{}".format(
            custom_llm_provider, stripped_model_name
        )
    else:
        split_model = model
        combined_model_name = "{}/{}".format(custom_llm_provider, model)
        stripped_model_name = _strip_model_name(
            model=model, custom_llm_provider=custom_llm_provider
        )
        combined_stripped_model_name = "{}/{}".format(
            custom_llm_provider,
            stripped_model_name,
        )

    return PotentialModelNamesAndCustomLLMProvider(
        split_model=split_model,
        combined_model_name=combined_model_name,
        stripped_model_name=stripped_model_name,
        combined_stripped_model_name=combined_stripped_model_name,
        custom_llm_provider=cast(str, custom_llm_provider),
    )


def _get_max_position_embeddings(model_name: str) -> Optional[int]:
    # Construct the URL for the config.json file
    config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"

    try:
        # Make the HTTP request to get the raw JSON file
        response = litellm.module_level_client.get(config_url)
        response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)

        # Parse the JSON response
        config_json = response.json()

        # Extract and return the max_position_embeddings
        max_position_embeddings = config_json.get("max_position_embeddings")

        if max_position_embeddings is not None:
            return max_position_embeddings
        else:
            return None
    except Exception:
        return None


def _cached_get_model_info_helper(
    model: str, custom_llm_provider: Optional[str]
) -> ModelInfoBase:
    """
    _get_model_info_helper wrapped with lru_cache

    Speed Optimization to hit high RPS
    """
    return _get_model_info_helper(model=model, custom_llm_provider=custom_llm_provider)


def get_provider_info(
    model: str, custom_llm_provider: Optional[str]
) -> Optional[ProviderSpecificModelInfo]:
    ## PROVIDER-SPECIFIC INFORMATION
    # if custom_llm_provider == "predibase":
    #     _model_info["supports_response_schema"] = True
    provider_config: Optional[BaseLLMModelInfo] = None
    if custom_llm_provider and custom_llm_provider in LlmProvidersSet:
        # Check if the provider string exists in LlmProviders enum
        provider_config = ProviderConfigManager.get_provider_model_info(
            model=model, provider=LlmProviders(custom_llm_provider)
        )

    model_info: Optional[ProviderSpecificModelInfo] = None
    if provider_config:
        model_info = provider_config.get_provider_info(model=model)

    return model_info


def _is_potential_model_name_in_model_cost(
    potential_model_names: PotentialModelNamesAndCustomLLMProvider,
) -> bool:
    """
    Check if the potential model name is in the model cost.
    """
    return any(
        potential_model_name in litellm.model_cost
        for potential_model_name in potential_model_names.values()
    )


def _get_model_info_helper(  # noqa: PLR0915
    model: str, custom_llm_provider: Optional[str] = None
) -> ModelInfoBase:
    """
    Helper for 'get_model_info'. Separated out to avoid infinite loop caused by returning 'supported_openai_param's
    """
    try:
        azure_llms = {**litellm.azure_llms, **litellm.azure_embedding_models}
        if model in azure_llms:
            model = azure_llms[model]
        if custom_llm_provider is not None and custom_llm_provider == "vertex_ai_beta":
            custom_llm_provider = "vertex_ai"
        if custom_llm_provider is not None and custom_llm_provider == "vertex_ai":
            if "meta/" + model in litellm.vertex_llama3_models:
                model = "meta/" + model
            elif model + "@latest" in litellm.vertex_mistral_models:
                model = model + "@latest"
            elif model + "@latest" in litellm.vertex_ai_ai21_models:
                model = model + "@latest"
        ##########################
        potential_model_names = _get_potential_model_names(
            model=model, custom_llm_provider=custom_llm_provider
        )

        verbose_logger.debug(
            f"checking potential_model_names in litellm.model_cost: {potential_model_names}"
        )

        combined_model_name = potential_model_names["combined_model_name"]
        stripped_model_name = potential_model_names["stripped_model_name"]
        combined_stripped_model_name = potential_model_names[
            "combined_stripped_model_name"
        ]
        split_model = potential_model_names["split_model"]
        custom_llm_provider = potential_model_names["custom_llm_provider"]
        #########################
        if custom_llm_provider == "huggingface":
            max_tokens = _get_max_position_embeddings(model_name=model)
            return ModelInfoBase(
                key=model,
                max_tokens=max_tokens,  # type: ignore
                max_input_tokens=None,
                max_output_tokens=None,
                input_cost_per_token=0,
                output_cost_per_token=0,
                litellm_provider="huggingface",
                mode="chat",
                supports_system_messages=None,
                supports_response_schema=None,
                supports_function_calling=None,
                supports_tool_choice=None,
                supports_assistant_prefill=None,
                supports_prompt_caching=None,
                supports_computer_use=None,
                supports_pdf_input=None,
            )
        elif (
            custom_llm_provider == "ollama" or custom_llm_provider == "ollama_chat"
        ) and not _is_potential_model_name_in_model_cost(potential_model_names):
            return litellm.OllamaConfig().get_model_info(model)
        else:
            """
            Check if: (in order of specificity)
            1. 'custom_llm_provider/model' in litellm.model_cost. Checks "groq/llama3-8b-8192" if model="llama3-8b-8192" and custom_llm_provider="groq"
            2. 'model' in litellm.model_cost. Checks "gemini-1.5-pro-002" in  litellm.model_cost if model="gemini-1.5-pro-002" and custom_llm_provider=None
            3. 'combined_stripped_model_name' in litellm.model_cost. Checks if 'gemini/gemini-1.5-flash' in model map, if 'gemini/gemini-1.5-flash-001' given.
            4. 'stripped_model_name' in litellm.model_cost. Checks if 'ft:gpt-3.5-turbo' in model map, if 'ft:gpt-3.5-turbo:my-org:custom_suffix:id' given.
            5. 'split_model' in litellm.model_cost. Checks "llama3-8b-8192" in litellm.model_cost if model="groq/llama3-8b-8192"
            """

            _model_info: Optional[Dict[str, Any]] = None
            key: Optional[str] = None

            if combined_model_name in litellm.model_cost:
                key = combined_model_name
                _model_info = _get_model_info_from_model_cost(key=cast(str, key))
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if _model_info is None and model in litellm.model_cost:
                key = model
                _model_info = _get_model_info_from_model_cost(key=cast(str, key))
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if (
                _model_info is None
                and combined_stripped_model_name in litellm.model_cost
            ):
                key = combined_stripped_model_name
                _model_info = _get_model_info_from_model_cost(key=cast(str, key))
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if _model_info is None and stripped_model_name in litellm.model_cost:
                key = stripped_model_name
                _model_info = _get_model_info_from_model_cost(key=cast(str, key))
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if _model_info is None and split_model in litellm.model_cost:
                key = split_model
                _model_info = _get_model_info_from_model_cost(key=cast(str, key))
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None

            if _model_info is None or key is None:
                raise ValueError(
                    "This model isn't mapped yet. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                )

            _input_cost_per_token: Optional[float] = _model_info.get(
                "input_cost_per_token"
            )
            if _input_cost_per_token is None:
                # default value to 0, be noisy about this
                verbose_logger.debug(
                    "model={}, custom_llm_provider={} has no input_cost_per_token in model_cost_map. Defaulting to 0.".format(
                        model, custom_llm_provider
                    )
                )
                _input_cost_per_token = 0

            _output_cost_per_token: Optional[float] = _model_info.get(
                "output_cost_per_token"
            )
            if _output_cost_per_token is None:
                # default value to 0, be noisy about this
                verbose_logger.debug(
                    "model={}, custom_llm_provider={} has no output_cost_per_token in model_cost_map. Defaulting to 0.".format(
                        model, custom_llm_provider
                    )
                )
                _output_cost_per_token = 0

            return ModelInfoBase(
                key=key,
                max_tokens=_model_info.get("max_tokens", None),
                max_input_tokens=_model_info.get("max_input_tokens", None),
                max_output_tokens=_model_info.get("max_output_tokens", None),
                input_cost_per_token=_input_cost_per_token,
                cache_creation_input_token_cost=_model_info.get(
                    "cache_creation_input_token_cost", None
                ),
                cache_read_input_token_cost=_model_info.get(
                    "cache_read_input_token_cost", None
                ),
                input_cost_per_character=_model_info.get(
                    "input_cost_per_character", None
                ),
                input_cost_per_token_above_128k_tokens=_model_info.get(
                    "input_cost_per_token_above_128k_tokens", None
                ),
                input_cost_per_token_above_200k_tokens=_model_info.get(
                    "input_cost_per_token_above_200k_tokens", None
                ),
                input_cost_per_query=_model_info.get("input_cost_per_query", None),
                input_cost_per_second=_model_info.get("input_cost_per_second", None),
                input_cost_per_audio_token=_model_info.get(
                    "input_cost_per_audio_token", None
                ),
                input_cost_per_token_batches=_model_info.get(
                    "input_cost_per_token_batches"
                ),
                output_cost_per_token_batches=_model_info.get(
                    "output_cost_per_token_batches"
                ),
                output_cost_per_token=_output_cost_per_token,
                output_cost_per_audio_token=_model_info.get(
                    "output_cost_per_audio_token", None
                ),
                output_cost_per_character=_model_info.get(
                    "output_cost_per_character", None
                ),
                output_cost_per_reasoning_token=_model_info.get(
                    "output_cost_per_reasoning_token", None
                ),
                output_cost_per_token_above_128k_tokens=_model_info.get(
                    "output_cost_per_token_above_128k_tokens", None
                ),
                output_cost_per_character_above_128k_tokens=_model_info.get(
                    "output_cost_per_character_above_128k_tokens", None
                ),
                output_cost_per_token_above_200k_tokens=_model_info.get(
                    "output_cost_per_token_above_200k_tokens", None
                ),
                output_cost_per_second=_model_info.get("output_cost_per_second", None),
                output_cost_per_image=_model_info.get("output_cost_per_image", None),
                output_vector_size=_model_info.get("output_vector_size", None),
                citation_cost_per_token=_model_info.get(
                    "citation_cost_per_token", None
                ),
                tiered_pricing=_model_info.get("tiered_pricing", None),
                litellm_provider=_model_info.get(
                    "litellm_provider", custom_llm_provider
                ),
                mode=_model_info.get("mode"),  # type: ignore
                supports_system_messages=_model_info.get(
                    "supports_system_messages", None
                ),
                supports_response_schema=_model_info.get(
                    "supports_response_schema", None
                ),
                supports_vision=_model_info.get("supports_vision", None),
                supports_function_calling=_model_info.get(
                    "supports_function_calling", None
                ),
                supports_tool_choice=_model_info.get("supports_tool_choice", None),
                supports_assistant_prefill=_model_info.get(
                    "supports_assistant_prefill", None
                ),
                supports_prompt_caching=_model_info.get(
                    "supports_prompt_caching", None
                ),
                supports_audio_input=_model_info.get("supports_audio_input", None),
                supports_audio_output=_model_info.get("supports_audio_output", None),
                supports_pdf_input=_model_info.get("supports_pdf_input", None),
                supports_embedding_image_input=_model_info.get(
                    "supports_embedding_image_input", None
                ),
                supports_native_streaming=_model_info.get(
                    "supports_native_streaming", None
                ),
                supports_web_search=_model_info.get("supports_web_search", None),
                supports_url_context=_model_info.get("supports_url_context", None),
                supports_reasoning=_model_info.get("supports_reasoning", None),
                supports_computer_use=_model_info.get("supports_computer_use", None),
                search_context_cost_per_query=_model_info.get(
                    "search_context_cost_per_query", None
                ),
                tpm=_model_info.get("tpm", None),
                rpm=_model_info.get("rpm", None),
            )
    except Exception as e:
        verbose_logger.debug(f"Error getting model info: {e}")
        if "OllamaError" in str(e):
            raise e
        raise Exception(
            "This model isn't mapped yet. model={}, custom_llm_provider={}. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json.".format(
                model, custom_llm_provider
            )
        )


def get_model_info(model: str, custom_llm_provider: Optional[str] = None) -> ModelInfo:
    """
    Get a dict for the maximum tokens (context window), input_cost_per_token, output_cost_per_token  for a given model.

    Parameters:
    - model (str): The name of the model.
    - custom_llm_provider (str | null): the provider used for the model. If provided, used to check if the litellm model info is for that provider.

    Returns:
        dict: A dictionary containing the following information:
            key: Required[str] # the key in litellm.model_cost which is returned
            max_tokens: Required[Optional[int]]
            max_input_tokens: Required[Optional[int]]
            max_output_tokens: Required[Optional[int]]
            input_cost_per_token: Required[float]
            input_cost_per_character: Optional[float]  # only for vertex ai models
            input_cost_per_token_above_128k_tokens: Optional[float]  # only for vertex ai models
            input_cost_per_character_above_128k_tokens: Optional[
                float
            ]  # only for vertex ai models
            input_cost_per_query: Optional[float] # only for rerank models
            input_cost_per_image: Optional[float]  # only for vertex ai models
            input_cost_per_audio_token: Optional[float]
            input_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
            input_cost_per_video_per_second: Optional[float]  # only for vertex ai models
            output_cost_per_token: Required[float]
            output_cost_per_audio_token: Optional[float]
            output_cost_per_character: Optional[float]  # only for vertex ai models
            output_cost_per_token_above_128k_tokens: Optional[
                float
            ]  # only for vertex ai models
            output_cost_per_character_above_128k_tokens: Optional[
                float
            ]  # only for vertex ai models
            output_cost_per_image: Optional[float]
            output_vector_size: Optional[int]
            output_cost_per_video_per_second: Optional[float]  # only for vertex ai models
            output_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
            litellm_provider: Required[str]
            mode: Required[
                Literal[
                    "completion", "embedding", "image_generation", "chat", "audio_transcription"
                ]
            ]
            supported_openai_params: Required[Optional[List[str]]]
            supports_system_messages: Optional[bool]
            supports_response_schema: Optional[bool]
            supports_vision: Optional[bool]
            supports_function_calling: Optional[bool]
            supports_tool_choice: Optional[bool]
            supports_prompt_caching: Optional[bool]
            supports_audio_input: Optional[bool]
            supports_audio_output: Optional[bool]
            supports_pdf_input: Optional[bool]
            supports_web_search: Optional[bool]
            supports_url_context: Optional[bool]
            supports_reasoning: Optional[bool]
    Raises:
        Exception: If the model is not mapped yet.

    Example:
        >>> get_model_info("gpt-4")
        {
            "max_tokens": 8192,
            "input_cost_per_token": 0.00003,
            "output_cost_per_token": 0.00006,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_openai_params": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
        }
    """
    supported_openai_params = litellm.get_supported_openai_params(
        model=model, custom_llm_provider=custom_llm_provider
    )

    _model_info = _get_model_info_helper(
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    verbose_logger.debug(f"model_info: {_model_info}")

    returned_model_info = ModelInfo(
        **_model_info, supported_openai_params=supported_openai_params
    )

    return returned_model_info


def json_schema_type(python_type_name: str):
    """Converts standard python types to json schema types

    Parameters
    ----------
    python_type_name : str
        __name__ of type

    Returns
    -------
    str
        a standard JSON schema type, "string" if not recognized.
    """
    python_to_json_schema_types = {
        str.__name__: "string",
        int.__name__: "integer",
        float.__name__: "number",
        bool.__name__: "boolean",
        list.__name__: "array",
        dict.__name__: "object",
        "NoneType": "null",
    }

    return python_to_json_schema_types.get(python_type_name, "string")


def function_to_dict(input_function):  # noqa: C901
    """Using type hints and numpy-styled docstring,
    produce a dictionnary usable for OpenAI function calling

    Parameters
    ----------
    input_function : function
        A function with a numpy-style docstring

    Returns
    -------
    dictionnary
        A dictionnary to add to the list passed to `functions` parameter of `litellm.completion`
    """
    # Get function name and docstring
    try:
        import inspect
        from ast import literal_eval

        from numpydoc.docscrape import NumpyDocString
    except Exception as e:
        raise e

    name = input_function.__name__
    docstring = inspect.getdoc(input_function)
    numpydoc = NumpyDocString(docstring)
    description = "\n".join([s.strip() for s in numpydoc["Summary"]])

    # Get function parameters and their types from annotations and docstring
    parameters = {}
    required_params = []
    param_info = inspect.signature(input_function).parameters

    for param_name, param in param_info.items():
        if hasattr(param, "annotation"):
            param_type = json_schema_type(param.annotation.__name__)
        else:
            param_type = None
        param_description = None
        param_enum = None

        # Try to extract param description from docstring using numpydoc
        for param_data in numpydoc["Parameters"]:
            if param_data.name == param_name:
                if hasattr(param_data, "type"):
                    # replace type from docstring rather than annotation
                    param_type = param_data.type
                    if "optional" in param_type:
                        param_type = param_type.split(",")[0]
                    elif "{" in param_type:
                        # may represent a set of acceptable values
                        # translating as enum for function calling
                        try:
                            param_enum = str(list(literal_eval(param_type)))
                            param_type = "string"
                        except Exception:
                            pass
                    param_type = json_schema_type(param_type)
                param_description = "\n".join([s.strip() for s in param_data.desc])

        param_dict = {
            "type": param_type,
            "description": param_description,
            "enum": param_enum,
        }

        parameters[param_name] = dict(
            [(k, v) for k, v in param_dict.items() if isinstance(v, str)]
        )

        # Check if the parameter has no default value (i.e., it's required)
        if param.default == param.empty:
            required_params.append(param_name)

    # Create the dictionary
    result = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
        },
    }

    # Add "required" key if there are required parameters
    if required_params:
        result["parameters"]["required"] = required_params

    return result


def modify_url(original_url, new_path):
    url = httpx.URL(original_url)
    modified_url = url.copy_with(path=new_path)
    return str(modified_url)


def load_test_model(
    model: str,
    custom_llm_provider: str = "",
    api_base: str = "",
    prompt: str = "",
    num_calls: int = 0,
    force_timeout: int = 0,
):
    test_prompt = "Hey, how's it going"
    test_calls = 100
    if prompt:
        test_prompt = prompt
    if num_calls:
        test_calls = num_calls
    messages = [[{"role": "user", "content": test_prompt}] for _ in range(test_calls)]
    start_time = time.time()
    try:
        litellm.batch_completion(
            model=model,
            messages=messages,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            force_timeout=force_timeout,
        )
        end_time = time.time()
        response_time = end_time - start_time
        return {
            "total_response_time": response_time,
            "calls_made": 100,
            "status": "success",
            "exception": None,
        }
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        return {
            "total_response_time": response_time,
            "calls_made": 100,
            "status": "failed",
            "exception": e,
        }


def get_provider_fields(custom_llm_provider: str) -> List[ProviderField]:
    """Return the fields required for each provider"""

    if custom_llm_provider == "databricks":
        return litellm.DatabricksConfig().get_required_params()

    elif custom_llm_provider == "ollama":
        return litellm.OllamaConfig().get_required_params()

    elif custom_llm_provider == "azure_ai":
        return litellm.AzureAIStudioConfig().get_required_params()

    else:
        return []


def create_proxy_transport_and_mounts():
    proxies = {
        key: None if url is None else Proxy(url=url)
        for key, url in get_environment_proxies().items()
    }

    sync_proxy_mounts = {}
    async_proxy_mounts = {}

    # Retrieve NO_PROXY environment variable
    no_proxy = os.getenv("NO_PROXY", None)
    no_proxy_urls = no_proxy.split(",") if no_proxy else []

    for key, proxy in proxies.items():
        if proxy is None:
            sync_proxy_mounts[key] = httpx.HTTPTransport()
            async_proxy_mounts[key] = httpx.AsyncHTTPTransport()
        else:
            sync_proxy_mounts[key] = httpx.HTTPTransport(proxy=proxy)
            async_proxy_mounts[key] = httpx.AsyncHTTPTransport(proxy=proxy)

    for url in no_proxy_urls:
        sync_proxy_mounts[url] = httpx.HTTPTransport()
        async_proxy_mounts[url] = httpx.AsyncHTTPTransport()

    return sync_proxy_mounts, async_proxy_mounts


def validate_environment(  # noqa: PLR0915
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> dict:
    """
    Checks if the environment variables are valid for the given model.

    Args:
        model (Optional[str]): The name of the model. Defaults to None.
        api_key (Optional[str]): If the user passed in an api key, of their own.

    Returns:
        dict: A dictionary containing the following keys:
            - keys_in_environment (bool): True if all the required keys are present in the environment, False otherwise.
            - missing_keys (List[str]): A list of missing keys in the environment.
    """
    keys_in_environment = False
    missing_keys: List[str] = []

    if model is None:
        return {
            "keys_in_environment": keys_in_environment,
            "missing_keys": missing_keys,
        }
    ## EXTRACT LLM PROVIDER - if model name provided
    try:
        _, custom_llm_provider, _, _ = get_llm_provider(model=model)
    except Exception:
        custom_llm_provider = None

    if custom_llm_provider:
        if custom_llm_provider == "openai":
            if "OPENAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENAI_API_KEY")
        elif custom_llm_provider == "azure":
            if (
                "AZURE_API_BASE" in os.environ
                and "AZURE_API_VERSION" in os.environ
                and "AZURE_API_KEY" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.extend(
                    ["AZURE_API_BASE", "AZURE_API_VERSION", "AZURE_API_KEY"]
                )
        elif custom_llm_provider == "anthropic":
            if "ANTHROPIC_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ANTHROPIC_API_KEY")
        elif custom_llm_provider == "cohere":
            if "COHERE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("COHERE_API_KEY")
        elif custom_llm_provider == "replicate":
            if "REPLICATE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("REPLICATE_API_KEY")
        elif custom_llm_provider == "openrouter":
            if "OPENROUTER_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENROUTER_API_KEY")
        elif custom_llm_provider == "vercel_ai_gateway":
            if "VERCEL_AI_GATEWAY_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("VERCEL_AI_GATEWAY_API_KEY")
        elif custom_llm_provider == "datarobot":
            if "DATAROBOT_API_TOKEN" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("DATAROBOT_API_TOKEN")
        elif custom_llm_provider == "vertex_ai":
            if "VERTEXAI_PROJECT" in os.environ and "VERTEXAI_LOCATION" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.extend(["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"])
        elif custom_llm_provider == "huggingface":
            if "HUGGINGFACE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("HUGGINGFACE_API_KEY")
        elif custom_llm_provider == "ai21":
            if "AI21_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("AI21_API_KEY")
        elif custom_llm_provider == "together_ai":
            if "TOGETHERAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("TOGETHERAI_API_KEY")
        elif custom_llm_provider == "aleph_alpha":
            if "ALEPH_ALPHA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ALEPH_ALPHA_API_KEY")
        elif custom_llm_provider == "baseten":
            if "BASETEN_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("BASETEN_API_KEY")
        elif custom_llm_provider == "nlp_cloud":
            if "NLP_CLOUD_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NLP_CLOUD_API_KEY")
        elif custom_llm_provider == "bedrock" or custom_llm_provider == "sagemaker":
            if (
                "AWS_ACCESS_KEY_ID" in os.environ
                and "AWS_SECRET_ACCESS_KEY" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.append("AWS_ACCESS_KEY_ID")
                missing_keys.append("AWS_SECRET_ACCESS_KEY")
        elif custom_llm_provider in ["ollama", "ollama_chat"]:
            if "OLLAMA_API_BASE" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OLLAMA_API_BASE")
        elif custom_llm_provider == "anyscale":
            if "ANYSCALE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ANYSCALE_API_KEY")
        elif custom_llm_provider == "deepinfra":
            if "DEEPINFRA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("DEEPINFRA_API_KEY")
        elif custom_llm_provider == "featherless_ai":
            if "FEATHERLESS_AI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("FEATHERLESS_AI_API_KEY")
        elif custom_llm_provider == "gemini":
            if ("GOOGLE_API_KEY" in os.environ) or ("GEMINI_API_KEY" in os.environ):
                keys_in_environment = True
            else:
                missing_keys.append("GOOGLE_API_KEY")
                missing_keys.append("GEMINI_API_KEY")
        elif custom_llm_provider == "groq":
            if "GROQ_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("GROQ_API_KEY")
        elif custom_llm_provider == "nvidia_nim":
            if "NVIDIA_NIM_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NVIDIA_NIM_API_KEY")
        elif custom_llm_provider == "cerebras":
            if "CEREBRAS_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("CEREBRAS_API_KEY")
        elif custom_llm_provider == "baseten":
            if "BASETEN_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("BASETEN_API_KEY")
        elif custom_llm_provider == "xai":
            if "XAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("XAI_API_KEY")
        elif custom_llm_provider == "ai21_chat":
            if "AI21_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("AI21_API_KEY")
        elif custom_llm_provider == "volcengine":
            if "VOLCENGINE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("VOLCENGINE_API_KEY")
        elif (
            custom_llm_provider == "codestral"
            or custom_llm_provider == "text-completion-codestral"
        ):
            if "CODESTRAL_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("CODESTRAL_API_KEY")
        elif custom_llm_provider == "deepseek":
            if "DEEPSEEK_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("DEEPSEEK_API_KEY")
        elif custom_llm_provider == "mistral":
            if "MISTRAL_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("MISTRAL_API_KEY")
        elif custom_llm_provider == "palm":
            if "PALM_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("PALM_API_KEY")
        elif custom_llm_provider == "perplexity":
            if "PERPLEXITYAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("PERPLEXITYAI_API_KEY")
        elif custom_llm_provider == "voyage":
            if "VOYAGE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("VOYAGE_API_KEY")
        elif custom_llm_provider == "infinity":
            if "INFINITY_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("INFINITY_API_KEY")
        elif custom_llm_provider == "fireworks_ai":
            if (
                "FIREWORKS_AI_API_KEY" in os.environ
                or "FIREWORKS_API_KEY" in os.environ
                or "FIREWORKSAI_API_KEY" in os.environ
                or "FIREWORKS_AI_TOKEN" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.append("FIREWORKS_AI_API_KEY")
        elif custom_llm_provider == "cloudflare":
            if "CLOUDFLARE_API_KEY" in os.environ and (
                "CLOUDFLARE_ACCOUNT_ID" in os.environ
                or "CLOUDFLARE_API_BASE" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.append("CLOUDFLARE_API_KEY")
                missing_keys.append("CLOUDFLARE_API_BASE")
        elif custom_llm_provider == "novita":
            if "NOVITA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NOVITA_API_KEY")
        elif custom_llm_provider == "nebius":
            if "NEBIUS_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NEBIUS_API_KEY")
        elif custom_llm_provider == "dashscope":
            if "DASHSCOPE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("DASHSCOPE_API_KEY")
        elif custom_llm_provider == "moonshot":
            if "MOONSHOT_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("MOONSHOT_API_KEY")
    else:
        ## openai - chatcompletion + text completion
        if (
            model in litellm.open_ai_chat_completion_models
            or model in litellm.open_ai_text_completion_models
            or model in litellm.open_ai_embedding_models
            or model in litellm.openai_image_generation_models
        ):
            if "OPENAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENAI_API_KEY")
        ## anthropic
        elif model in litellm.anthropic_models:
            if "ANTHROPIC_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ANTHROPIC_API_KEY")
        ## cohere
        elif model in litellm.cohere_models:
            if "COHERE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("COHERE_API_KEY")
        ## replicate
        elif model in litellm.replicate_models:
            if "REPLICATE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("REPLICATE_API_KEY")
        ## openrouter
        elif model in litellm.openrouter_models:
            if "OPENROUTER_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENROUTER_API_KEY")
        ## vercel_ai_gateway
        elif model in litellm.vercel_ai_gateway_models:
            if "VERCEL_AI_GATEWAY_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("VERCEL_AI_GATEWAY_API_KEY")
        ## datarobot
        elif model in litellm.datarobot_models:
            if "DATAROBOT_API_TOKEN" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("DATAROBOT_API_TOKEN")
        ## vertex - text + chat models
        elif (
            model in litellm.vertex_chat_models
            or model in litellm.vertex_text_models
            or model in litellm.models_by_provider["vertex_ai"]
        ):
            if "VERTEXAI_PROJECT" in os.environ and "VERTEXAI_LOCATION" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.extend(["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"])
        ## huggingface
        elif model in litellm.huggingface_models:
            if "HUGGINGFACE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("HUGGINGFACE_API_KEY")
        ## ai21
        elif model in litellm.ai21_models:
            if "AI21_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("AI21_API_KEY")
        ## together_ai
        elif model in litellm.together_ai_models:
            if "TOGETHERAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("TOGETHERAI_API_KEY")
        ## aleph_alpha
        elif model in litellm.aleph_alpha_models:
            if "ALEPH_ALPHA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ALEPH_ALPHA_API_KEY")
        ## baseten
        elif model in litellm.baseten_models:
            if "BASETEN_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("BASETEN_API_KEY")
        ## nlp_cloud
        elif model in litellm.nlp_cloud_models:
            if "NLP_CLOUD_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NLP_CLOUD_API_KEY")
        elif model in litellm.novita_models:
            if "NOVITA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NOVITA_API_KEY")
        elif model in litellm.nebius_models:
            if "NEBIUS_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NEBIUS_API_KEY")

    def filter_missing_keys(keys: List[str], exclude_pattern: str) -> List[str]:
        """Filter out keys that contain the exclude_pattern (case insensitive)."""
        return [key for key in keys if exclude_pattern not in key.lower()]

    if api_key is not None:
        missing_keys = filter_missing_keys(missing_keys, "api_key")

    if api_base is not None:
        missing_keys = filter_missing_keys(missing_keys, "api_base")

    if api_version is not None:
        missing_keys = filter_missing_keys(missing_keys, "api_version")

    if len(missing_keys) == 0:  # no missing keys
        keys_in_environment = True

    return {"keys_in_environment": keys_in_environment, "missing_keys": missing_keys}


def acreate(*args, **kwargs):  ## Thin client to handle the acreate langchain call
    return litellm.acompletion(*args, **kwargs)


def prompt_token_calculator(model, messages):
    # use tiktoken or anthropic's tokenizer depending on the model
    text = " ".join(message["content"] for message in messages)
    num_tokens = 0
    if "claude" in model:
        try:
            import anthropic
        except Exception:
            Exception("Anthropic import failed please run `pip install anthropic`")
        from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic

        anthropic_obj = Anthropic()
        num_tokens = anthropic_obj.count_tokens(text)  # type: ignore
    else:
        num_tokens = len(encoding.encode(text))
    return num_tokens


def valid_model(model):
    try:
        # for a given model name, check if the user has the right permissions to access the model
        if (
            model in litellm.open_ai_chat_completion_models
            or model in litellm.open_ai_text_completion_models
        ):
            openai.models.retrieve(model)
        else:
            messages = [{"role": "user", "content": "Hello World"}]
            litellm.completion(model=model, messages=messages)
    except Exception:
        raise BadRequestError(message="", model=model, llm_provider="")


def check_valid_key(model: str, api_key: str):
    """
    Checks if a given API key is valid for a specific model by making a litellm.completion call with max_tokens=10

    Args:
        model (str): The name of the model to check the API key against.
        api_key (str): The API key to be checked.

    Returns:
        bool: True if the API key is valid for the model, False otherwise.
    """
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        litellm.completion(
            model=model, messages=messages, api_key=api_key, max_tokens=10
        )
        return True
    except AuthenticationError:
        return False
    except Exception:
        return False


def _should_retry(status_code: int):
    """
    Retries on 408, 409, 429 and 500 errors.

    Any client error in the 400-499 range that isn't explicitly handled (such as 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, etc.) would not trigger a retry.

    Reimplementation of openai's should retry logic, since that one can't be imported.
    https://github.com/openai/openai-python/blob/af67cfab4210d8e497c05390ce14f39105c77519/src/openai/_base_client.py#L639
    """
    # If the server explicitly says whether or not to retry, obey.
    # Retry on request timeouts.
    if status_code == 408:
        return True

    # Retry on lock timeouts.
    if status_code == 409:
        return True

    # Retry on rate limits.
    if status_code == 429:
        return True

    # Retry internal errors.
    if status_code >= 500:
        return True

    return False


def _get_retry_after_from_exception_header(
    response_headers: Optional[httpx.Headers] = None,
):
    """
    Reimplementation of openai's calculate retry after, since that one can't be imported.
    https://github.com/openai/openai-python/blob/af67cfab4210d8e497c05390ce14f39105c77519/src/openai/_base_client.py#L631
    """
    try:
        import email  # openai import

        # About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
        #
        # <http-date>". See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After#syntax for
        # details.
        if response_headers is not None:
            retry_header = response_headers.get("retry-after")
            try:
                retry_after = int(retry_header)
            except Exception:
                retry_date_tuple = email.utils.parsedate_tz(retry_header)  # type: ignore
                if retry_date_tuple is None:
                    retry_after = -1
                else:
                    retry_date = email.utils.mktime_tz(retry_date_tuple)  # type: ignore
                    retry_after = int(retry_date - time.time())
        else:
            retry_after = -1

        return retry_after

    except Exception:
        retry_after = -1


def _calculate_retry_after(
    remaining_retries: int,
    max_retries: int,
    response_headers: Optional[httpx.Headers] = None,
    min_timeout: int = 0,
) -> Union[float, int]:
    retry_after = _get_retry_after_from_exception_header(response_headers)

    # Add some jitter (default JITTER is 0.75 - so upto 0.75s)
    jitter = JITTER * random.random()

    # If the API asks us to wait a certain amount of time (and it's a reasonable amount), just do what it says.
    if retry_after is not None and 0 < retry_after <= 60:
        return retry_after + jitter

    # Calculate exponential backoff
    num_retries = max_retries - remaining_retries
    sleep_seconds = INITIAL_RETRY_DELAY * pow(2.0, num_retries)

    # Make sure sleep_seconds is boxed between min_timeout and MAX_RETRY_DELAY
    sleep_seconds = max(sleep_seconds, min_timeout)
    sleep_seconds = min(sleep_seconds, MAX_RETRY_DELAY)

    return sleep_seconds + jitter


# custom prompt helper function
def register_prompt_template(
    model: str,
    roles: dict = {},
    initial_prompt_value: str = "",
    final_prompt_value: str = "",
    tokenizer_config: dict = {},
):
    """
    Register a prompt template to follow your custom format for a given model

    Args:
        model (str): The name of the model.
        roles (dict): A dictionary mapping roles to their respective prompt values.
        initial_prompt_value (str, optional): The initial prompt value. Defaults to "".
        final_prompt_value (str, optional): The final prompt value. Defaults to "".

    Returns:
        dict: The updated custom prompt dictionary.
    Example usage:
    ```
    import litellm
    litellm.register_prompt_template(
            model="llama-2",
        initial_prompt_value="You are a good assistant" # [OPTIONAL]
            roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n", # [OPTIONAL]
                "post_message": "\n<</SYS>>\n [/INST]\n" # [OPTIONAL]
            },
            "user": {
                "pre_message": "[INST] ", # [OPTIONAL]
                "post_message": " [/INST]" # [OPTIONAL]
            },
            "assistant": {
                "pre_message": "\n" # [OPTIONAL]
                "post_message": "\n" # [OPTIONAL]
            }
        }
        final_prompt_value="Now answer as best you can:" # [OPTIONAL]
    )
    ```
    """
    complete_model = model
    potential_models = [complete_model]
    try:
        model = get_llm_provider(model=model)[0]
        potential_models.append(model)
    except Exception:
        pass
    if tokenizer_config:
        for m in potential_models:
            litellm.known_tokenizer_config[m] = {
                "tokenizer": tokenizer_config,
                "status": "success",
            }
    else:
        for m in potential_models:
            litellm.custom_prompt_dict[m] = {
                "roles": roles,
                "initial_prompt_value": initial_prompt_value,
                "final_prompt_value": final_prompt_value,
            }

    return litellm.custom_prompt_dict


class TextCompletionStreamWrapper:
    def __init__(
        self,
        completion_stream,
        model,
        stream_options: Optional[dict] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        self.completion_stream = completion_stream
        self.model = model
        self.stream_options = stream_options
        self.custom_llm_provider = custom_llm_provider

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def convert_to_text_completion_object(self, chunk: ModelResponse):
        try:
            response = TextCompletionResponse()
            response["id"] = chunk.get("id", None)
            response["object"] = "text_completion"
            response["created"] = chunk.get("created", None)
            response["model"] = chunk.get("model", None)
            text_choices = TextChoices()
            if isinstance(
                chunk, Choices
            ):  # chunk should always be of type StreamingChoices
                raise Exception
            delta = chunk["choices"][0]["delta"]
            text_choices["text"] = delta["content"]
            text_choices["reasoning_content"] = delta.get("reasoning_content")
            text_choices["index"] = chunk["choices"][0]["index"]
            text_choices["finish_reason"] = chunk["choices"][0]["finish_reason"]
            response["choices"] = [text_choices]

            # only pass usage when stream_options["include_usage"] is True
            if (
                self.stream_options
                and self.stream_options.get("include_usage", False) is True
            ):
                response["usage"] = chunk.get("usage", None)

            return response
        except Exception as e:
            raise Exception(
                f"Error occurred converting to text completion object - chunk: {chunk}; Error: {str(e)}"
            )

    def __next__(self):
        # model_response = ModelResponse(stream=True, model=self.model)
        TextCompletionResponse()
        try:
            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                processed_chunk = self.convert_to_text_completion_object(chunk=chunk)
                return processed_chunk
            raise StopIteration
        except StopIteration:
            raise StopIteration
        except Exception as e:
            raise exception_type(
                model=self.model,
                custom_llm_provider=self.custom_llm_provider or "",
                original_exception=e,
                completion_kwargs={},
                extra_kwargs={},
            )

    async def __anext__(self):
        try:
            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                processed_chunk = self.convert_to_text_completion_object(chunk=chunk)
                return processed_chunk
            raise StopIteration
        except StopIteration:
            raise StopAsyncIteration


def mock_completion_streaming_obj(
    model_response, mock_response, model, n: Optional[int] = None
):
    if isinstance(mock_response, litellm.MockException):
        raise mock_response
    for i in range(0, len(mock_response), 3):
        completion_obj = Delta(role="assistant", content=mock_response[i : i + 3])
        if n is None:
            model_response.choices[0].delta = completion_obj
        else:
            _all_choices = []
            for j in range(n):
                _streaming_choice = litellm.utils.StreamingChoices(
                    index=j,
                    delta=litellm.utils.Delta(
                        role="assistant", content=mock_response[i : i + 3]
                    ),
                )
                _all_choices.append(_streaming_choice)
            model_response.choices = _all_choices
        yield model_response


async def async_mock_completion_streaming_obj(
    model_response, mock_response, model, n: Optional[int] = None
):
    if isinstance(mock_response, litellm.MockException):
        raise mock_response
    for i in range(0, len(mock_response), 3):
        completion_obj = Delta(role="assistant", content=mock_response[i : i + 3])
        if n is None:
            model_response.choices[0].delta = completion_obj
        else:
            _all_choices = []
            for j in range(n):
                _streaming_choice = litellm.utils.StreamingChoices(
                    index=j,
                    delta=litellm.utils.Delta(
                        role="assistant", content=mock_response[i : i + 3]
                    ),
                )
                _all_choices.append(_streaming_choice)
            model_response.choices = _all_choices
        yield model_response


########## Reading Config File ############################
def read_config_args(config_path) -> dict:
    try:
        import os

        os.getcwd()
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        # read keys/ values from config file and return them
        return config
    except Exception as e:
        raise e


########## experimental completion variants ############################


def process_system_message(system_message, max_tokens, model):
    system_message_event = {"role": "system", "content": system_message}
    system_message_tokens = get_token_count([system_message_event], model)

    if system_message_tokens > max_tokens:
        print_verbose(
            "`tokentrimmer`: Warning, system message exceeds token limit. Trimming..."
        )
        # shorten system message to fit within max_tokens
        new_system_message = shorten_message_to_fit_limit(
            system_message_event, max_tokens, model
        )
        system_message_tokens = get_token_count([new_system_message], model)

    return system_message_event, max_tokens - system_message_tokens


def process_messages(messages, max_tokens, model):
    # Process messages from older to more recent
    messages = messages[::-1]
    final_messages = []
    verbose_logger.debug(
        f"calling process_messages with messages: {messages}, max_tokens: {max_tokens}, model: {model}"
    )
    for message in messages:
        verbose_logger.debug(f"processing final_messages: {final_messages}")
        used_tokens = get_token_count(final_messages, model)
        available_tokens = max_tokens - used_tokens
        verbose_logger.debug(
            f"used_tokens: {used_tokens}, available_tokens: {available_tokens}"
        )
        if available_tokens <= 3:
            break

        final_messages = attempt_message_addition(
            final_messages=final_messages,
            message=message,
            available_tokens=available_tokens,
            max_tokens=max_tokens,
            model=model,
        )
        verbose_logger.debug(
            f"final_messages after attempt_message_addition: {final_messages}"
        )
    verbose_logger.debug(f"Final messages: {final_messages}")
    return final_messages


def attempt_message_addition(
    final_messages, message, available_tokens, max_tokens, model
):
    temp_messages = [message] + final_messages
    temp_message_tokens = get_token_count(messages=temp_messages, model=model)
    verbose_logger.debug(
        f"temp_message_tokens: {temp_message_tokens}, max_tokens: {max_tokens}"
    )
    if temp_message_tokens <= max_tokens:
        return temp_messages

    # if temp_message_tokens > max_tokens, try shortening temp_messages
    elif "function_call" not in message:
        verbose_logger.debug("attempting to shorten message to fit limit")
        # fit updated_message to be within temp_message_tokens - max_tokens (aka the amount temp_message_tokens is greate than max_tokens)
        updated_message = shorten_message_to_fit_limit(message, available_tokens, model)
        if can_add_message(updated_message, final_messages, max_tokens, model):
            verbose_logger.debug(
                "can add message, returning [updated_message] + final_messages"
            )
            return [updated_message] + final_messages
        else:
            verbose_logger.debug("cannot add message, returning final_messages")
    return final_messages


def can_add_message(message, messages, max_tokens, model):
    if get_token_count(messages + [message], model) <= max_tokens:
        return True
    return False


def get_token_count(messages, model):
    return token_counter(model=model, messages=messages)


def shorten_message_to_fit_limit(
    message, tokens_needed, model: Optional[str], raise_error_on_max_limit: bool = False
):
    """
    Shorten a message to fit within a token limit by removing characters from the middle.

    Args:
        message: The message to shorten
        tokens_needed: The maximum number of tokens allowed
        model: The model being used (optional)
        raise_error_on_max_limit: If True, raises an error when max attempts reached. If False, returns final trimmed content.
    """

    # For OpenAI models, even blank messages cost 7 token,
    # and if the buffer is less than 3, the while loop will never end,
    # hence the value 10.
    if model is not None and "gpt" in model and tokens_needed <= 10:
        return message

    content = message["content"]
    attempts = 0

    verbose_logger.debug(f"content: {content}")

    while attempts < MAX_TOKEN_TRIMMING_ATTEMPTS:
        verbose_logger.debug(f"getting token count for message: {message}")
        total_tokens = get_token_count([message], model)
        verbose_logger.debug(
            f"total_tokens: {total_tokens}, tokens_needed: {tokens_needed}"
        )

        if total_tokens <= tokens_needed:
            break

        ratio = (tokens_needed) / total_tokens

        new_length = int(len(content) * ratio) - 1
        new_length = max(0, new_length)

        half_length = new_length // 2
        left_half = content[:half_length]
        right_half = content[-half_length:]

        trimmed_content = left_half + ".." + right_half
        message["content"] = trimmed_content
        verbose_logger.debug(f"trimmed_content: {trimmed_content}")
        content = trimmed_content
        attempts += 1

    if attempts >= MAX_TOKEN_TRIMMING_ATTEMPTS and raise_error_on_max_limit:
        raise Exception(
            f"Failed to trim message to fit within {tokens_needed} tokens after {MAX_TOKEN_TRIMMING_ATTEMPTS} attempts"
        )

    return message


# LiteLLM token trimmer
# this code is borrowed from https://github.com/KillianLucas/tokentrim/blob/main/tokentrim/tokentrim.py
# Credits for this code go to Killian Lucas
def trim_messages(
    messages,
    model: Optional[str] = None,
    trim_ratio: float = DEFAULT_TRIM_RATIO,
    return_response_tokens: bool = False,
    max_tokens=None,
):
    """
    Trim a list of messages to fit within a model's token limit.

    Args:
        messages: Input messages to be trimmed. Each message is a dictionary with 'role' and 'content'.
        model: The LiteLLM model being used (determines the token limit).
        trim_ratio: Target ratio of tokens to use after trimming. Default is 0.75, meaning it will trim messages so they use about 75% of the model's token limit.
        return_response_tokens: If True, also return the number of tokens left available for the response after trimming.
        max_tokens: Instead of specifying a model or trim_ratio, you can specify this directly.

    Returns:
        Trimmed messages and optionally the number of tokens available for response.
    """
    # Initialize max_tokens
    # if users pass in max tokens, trim to this amount
    original_messages = messages
    messages = copy.deepcopy(messages)
    try:
        if max_tokens is None:
            # Check if model is valid
            if model in litellm.model_cost:
                max_tokens_for_model = litellm.model_cost[model].get(
                    "max_input_tokens", litellm.model_cost[model]["max_tokens"]
                )
                max_tokens = int(max_tokens_for_model * trim_ratio)
            else:
                # if user did not specify max (input) tokens
                # or passed an llm litellm does not know
                # do nothing, just return messages
                return messages

        system_message = ""
        for message in messages:
            if message["role"] == "system":
                system_message += "\n" if system_message else ""
                system_message += message["content"]

        ## Handle Tool Call ## - check if last message is a tool response, return as is - https://github.com/BerriAI/litellm/issues/4931
        tool_messages = []

        for message in reversed(messages):
            if message["role"] != "tool":
                break
            tool_messages.append(message)
        tool_messages.reverse()
        # # Remove the collected tool messages from the original list
        if len(tool_messages):
            messages = messages[: -len(tool_messages)]

        current_tokens = token_counter(model=model or "", messages=messages)
        print_verbose(f"Current tokens: {current_tokens}, max tokens: {max_tokens}")

        # Do nothing if current tokens under messages
        if current_tokens < max_tokens:
            return messages + tool_messages

        #### Trimming messages if current_tokens > max_tokens
        print_verbose(
            f"Need to trim input messages: {messages}, current_tokens{current_tokens}, max_tokens: {max_tokens}"
        )
        system_message_event: Optional[dict] = None
        if system_message:
            system_message_event, max_tokens = process_system_message(
                system_message=system_message, max_tokens=max_tokens, model=model
            )

            if max_tokens == 0:  # the system messages are too long
                return [system_message_event]

            # Since all system messages are combined and trimmed to fit the max_tokens,
            # we remove all system messages from the messages list
            messages = [message for message in messages if message["role"] != "system"]

        verbose_logger.debug(f"Processed system message: {system_message_event}")
        final_messages = process_messages(
            messages=messages, max_tokens=max_tokens, model=model
        )
        verbose_logger.debug(f"Processed messages: {final_messages}")

        # Add system message to the beginning of the final messages
        if system_message_event:
            final_messages = [system_message_event] + final_messages

        if len(tool_messages) > 0:
            final_messages.extend(tool_messages)

        verbose_logger.debug(
            f"Final messages: {final_messages}, return_response_tokens: {return_response_tokens}"
        )
        if (
            return_response_tokens
        ):  # if user wants token count with new trimmed messages
            response_tokens = max_tokens - get_token_count(final_messages, model)
            return final_messages, response_tokens
        return final_messages
    except Exception as e:  # [NON-Blocking, if error occurs just return final_messages
        verbose_logger.exception(
            "Got exception while token trimming - {}".format(str(e))
        )
        return original_messages


from litellm.caching.in_memory_cache import InMemoryCache


class AvailableModelsCache(InMemoryCache):
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        super().__init__(ttl_seconds, max_size)
        self._env_hash: Optional[str] = None

    def _get_env_hash(self) -> str:
        """Create a hash of relevant environment variables"""
        env_vars = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("OPENAI", "ANTHROPIC", "AZURE", "AWS"))
        }
        return str(hash(frozenset(env_vars.items())))

    def _check_env_changed(self) -> bool:
        """Check if environment variables have changed"""
        current_hash = self._get_env_hash()
        if self._env_hash is None:
            self._env_hash = current_hash
            return True
        return current_hash != self._env_hash

    def _get_cache_key(
        self,
        custom_llm_provider: Optional[str],
        litellm_params: Optional[LiteLLM_Params],
    ) -> str:
        valid_str = ""

        if litellm_params is not None:
            valid_str = litellm_params.model_dump_json()
        if custom_llm_provider is not None:
            valid_str = f"{custom_llm_provider}:{valid_str}"
        return hashlib.sha256(valid_str.encode()).hexdigest()

    def get_cached_model_info(
        self,
        custom_llm_provider: Optional[str] = None,
        litellm_params: Optional[LiteLLM_Params] = None,
    ) -> Optional[List[str]]:
        """Get cached model info"""
        # Check if environment has changed
        if litellm_params is None and self._check_env_changed():
            self.cache_dict.clear()
            return None

        cache_key = self._get_cache_key(custom_llm_provider, litellm_params)

        result = cast(Optional[List[str]], self.get_cache(cache_key))

        if result is not None:
            return copy.deepcopy(result)
        return result

    def set_cached_model_info(
        self,
        custom_llm_provider: str,
        litellm_params: Optional[LiteLLM_Params],
        available_models: List[str],
    ):
        """Set cached model info"""
        cache_key = self._get_cache_key(custom_llm_provider, litellm_params)
        self.set_cache(cache_key, copy.deepcopy(available_models))


# Global cache instance
_model_cache = AvailableModelsCache()


def _infer_valid_provider_from_env_vars(
    custom_llm_provider: Optional[str] = None,
) -> List[str]:
    valid_providers: List[str] = []
    environ_keys = os.environ.keys()
    for provider in litellm.provider_list:
        if custom_llm_provider and provider != custom_llm_provider:
            continue

        # edge case litellm has together_ai as a provider, it should be togetherai
        env_provider_1 = provider.replace("_", "")
        env_provider_2 = provider

        # litellm standardizes expected provider keys to
        # PROVIDER_API_KEY. Example: OPENAI_API_KEY, COHERE_API_KEY
        expected_provider_key_1 = f"{env_provider_1.upper()}_API_KEY"
        expected_provider_key_2 = f"{env_provider_2.upper()}_API_KEY"
        if (
            expected_provider_key_1 in environ_keys
            or expected_provider_key_2 in environ_keys
        ):
            # key is set
            valid_providers.append(provider)

    return valid_providers


def _get_valid_models_from_provider_api(
    provider_config: BaseLLMModelInfo,
    custom_llm_provider: str,
    litellm_params: Optional[LiteLLM_Params] = None,
) -> List[str]:
    try:
        cached_result = _model_cache.get_cached_model_info(
            custom_llm_provider, litellm_params
        )

        if cached_result is not None:
            return cached_result
        models = provider_config.get_models(
            api_key=litellm_params.api_key if litellm_params is not None else None,
            api_base=litellm_params.api_base if litellm_params is not None else None,
        )

        _model_cache.set_cached_model_info(custom_llm_provider, litellm_params, models)
        return models
    except Exception as e:
        verbose_logger.warning(f"Error getting valid models: {e}")
        return []


def get_valid_models(
    check_provider_endpoint: Optional[bool] = None,
    custom_llm_provider: Optional[str] = None,
    litellm_params: Optional[LiteLLM_Params] = None,
) -> List[str]:
    """
    Returns a list of valid LLMs based on the set environment variables

    Args:
        check_provider_endpoint: If True, will check the provider's endpoint for valid models.
        custom_llm_provider: If provided, will only check the provider's endpoint for valid models.
    Returns:
        A list of valid LLMs
    """

    try:
        check_provider_endpoint = (
            check_provider_endpoint or litellm.check_provider_endpoint
        )
        # get keys set in .env

        valid_providers: List[str] = []
        valid_models: List[str] = []
        # for all valid providers, make a list of supported llms

        if custom_llm_provider:
            valid_providers = [custom_llm_provider]
        else:
            valid_providers = _infer_valid_provider_from_env_vars(custom_llm_provider)

        for provider in valid_providers:
            provider_config = ProviderConfigManager.get_provider_model_info(
                model=None,
                provider=LlmProviders(provider),
            )

            if custom_llm_provider and provider != custom_llm_provider:
                continue

            if provider == "azure":
                valid_models.append("Azure-LLM")
            elif (
                provider_config is not None
                and check_provider_endpoint
                and provider is not None
            ):
                valid_models.extend(
                    _get_valid_models_from_provider_api(
                        provider_config,
                        provider,
                        litellm_params,
                    )
                )
            else:
                models_for_provider = copy.deepcopy(
                    litellm.models_by_provider.get(provider, [])
                )
                valid_models.extend(models_for_provider)

        return valid_models
    except Exception as e:
        verbose_logger.warning(f"Error getting valid models: {e}")
        return []  # NON-Blocking


def print_args_passed_to_litellm(original_function, args, kwargs):
    if not _is_debugging_on():
        return
    try:
        # we've already printed this for acompletion, don't print for completion
        if (
            "acompletion" in kwargs
            and kwargs["acompletion"] is True
            and original_function.__name__ == "completion"
        ):
            return
        elif (
            "aembedding" in kwargs
            and kwargs["aembedding"] is True
            and original_function.__name__ == "embedding"
        ):
            return
        elif (
            "aimg_generation" in kwargs
            and kwargs["aimg_generation"] is True
            and original_function.__name__ == "img_generation"
        ):
            return

        args_str = ", ".join(map(repr, args))
        kwargs_str = ", ".join(f"{key}={repr(value)}" for key, value in kwargs.items())
        print_verbose(
            "\n",
        )  # new line before
        print_verbose(
            "\033[92mRequest to litellm:\033[0m",
        )
        if args and kwargs:
            print_verbose(
                f"\033[92mlitellm.{original_function.__name__}({args_str}, {kwargs_str})\033[0m"
            )
        elif args:
            print_verbose(
                f"\033[92mlitellm.{original_function.__name__}({args_str})\033[0m"
            )
        elif kwargs:
            print_verbose(
                f"\033[92mlitellm.{original_function.__name__}({kwargs_str})\033[0m"
            )
        else:
            print_verbose(f"\033[92mlitellm.{original_function.__name__}()\033[0m")
        print_verbose("\n")  # new line after
    except Exception:
        # This should always be non blocking
        pass


def get_logging_id(start_time, response_obj):
    try:
        response_id = (
            "time-" + start_time.strftime("%H-%M-%S-%f") + "_" + response_obj.get("id")
        )
        return response_id
    except Exception:
        return None


def _get_base_model_from_metadata(model_call_details=None):
    if model_call_details is None:
        return None
    litellm_params = model_call_details.get("litellm_params", {})
    if litellm_params is not None:
        _base_model = litellm_params.get("base_model", None)
        if _base_model is not None:
            return _base_model
        metadata = litellm_params.get("metadata", {})

        return _get_base_model_from_litellm_call_metadata(metadata=metadata)
    return None


class ModelResponseIterator:
    def __init__(self, model_response: ModelResponse, convert_to_delta: bool = False):
        if convert_to_delta is True:
            self.model_response = ModelResponse(stream=True)
            _delta = self.model_response.choices[0].delta  # type: ignore
            _delta.content = model_response.choices[0].message.content  # type: ignore
        else:
            self.model_response = model_response
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.model_response


class ModelResponseListIterator:
    def __init__(self, model_responses, delay: Optional[float] = None):
        self.model_responses = model_responses
        self.index = 0
        self.delay = delay

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.model_responses):
            raise StopIteration
        model_response = self.model_responses[self.index]
        self.index += 1
        if self.delay:
            time.sleep(self.delay)
        return model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.model_responses):
            raise StopAsyncIteration
        model_response = self.model_responses[self.index]
        self.index += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return model_response


class CustomModelResponseIterator(Iterable):
    def __init__(self) -> None:
        super().__init__()


def is_cached_message(message: AllMessageValues) -> bool:
    """
    Returns true, if message is marked as needing to be cached.

    Used for anthropic/gemini context caching.

    Follows the anthropic format {"cache_control": {"type": "ephemeral"}}
    """
    if "content" not in message:
        return False
    if message["content"] is None or isinstance(message["content"], str):
        return False

    for content in message["content"]:
        if (
            content["type"] == "text"
            and content.get("cache_control") is not None
            and content["cache_control"]["type"] == "ephemeral"  # type: ignore
        ):
            return True

    return False


def is_base64_encoded(s: str) -> bool:
    try:
        # Strip out the prefix if it exists
        if not s.startswith(
            "data:"
        ):  # require `data:` for base64 str, like openai. Prevents false positives like s='Dog'
            return False

        s = s.split(",")[1]

        # Try to decode the string
        decoded_bytes = base64.b64decode(s, validate=True)

        # Check if the original string can be re-encoded to the same string
        return base64.b64encode(decoded_bytes).decode("utf-8") == s
    except Exception:
        return False


def get_base64_str(s: str) -> str:
    """
    s: b64str OR data:image/png;base64,b64str
    """
    if "," in s:
        return s.split(",")[1]
    return s


def has_tool_call_blocks(messages: List[AllMessageValues]) -> bool:
    """
    Returns true, if messages has tool call blocks.

    Used for anthropic/bedrock message validation.
    """
    for message in messages:
        if message.get("tool_calls") is not None:
            return True
    return False


def add_dummy_tool(custom_llm_provider: str) -> List[ChatCompletionToolParam]:
    """
    Prevent Anthropic from raising error when tool_use block exists but no tools are provided.

    Relevent Issues: https://github.com/BerriAI/litellm/issues/5388, https://github.com/BerriAI/litellm/issues/5747
    """
    return [
        ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name="dummy_tool",
                description="This is a dummy tool call",  # provided to satisfy bedrock constraint.
                parameters={
                    "type": "object",
                    "properties": {},
                },
            ),
        )
    ]


from litellm.types.llms.openai import (
    ChatCompletionAudioObject,
    ChatCompletionImageObject,
    ChatCompletionTextObject,
    ChatCompletionUserMessage,
    OpenAIMessageContent,
    ValidUserMessageContentTypes,
)


def convert_to_dict(message: Union[BaseModel, dict]) -> dict:
    """
    Converts a message to a dictionary if it's a Pydantic model.

    Args:
        message: The message, which may be a Pydantic model or a dictionary.

    Returns:
        dict: The converted message.
    """
    if isinstance(message, BaseModel):
        return message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        return message
    else:
        raise TypeError(
            f"Invalid message type: {type(message)}. Expected dict or Pydantic model."
        )


def convert_list_message_to_dict(messages: List):
    new_messages = []
    for message in messages:
        convert_msg_to_dict = cast(AllMessageValues, convert_to_dict(message))
        cleaned_message = cleanup_none_field_in_message(message=convert_msg_to_dict)
        new_messages.append(cleaned_message)
    return new_messages


def validate_and_fix_openai_messages(messages: List):
    """
    Ensures all messages are valid OpenAI chat completion messages.

    Handles missing role for assistant messages.
    """
    new_messages = []
    for message in messages:
        if not message.get("role"):
            message["role"] = "assistant"
        if message.get("tool_calls"):
            message["tool_calls"] = jsonify_tools(tools=message["tool_calls"])

        convert_msg_to_dict = cast(AllMessageValues, convert_to_dict(message))
        cleaned_message = cleanup_none_field_in_message(message=convert_msg_to_dict)
        new_messages.append(cleaned_message)
    return validate_chat_completion_user_messages(messages=new_messages)


def validate_and_fix_openai_tools(tools: Optional[List]) -> Optional[List[dict]]:
    """
    Ensure tools is List[dict] and not List[BaseModel]
    """
    new_tools = []
    if tools is None:
        return tools
    for tool in tools:
        if isinstance(tool, BaseModel):
            new_tools.append(tool.model_dump())
        elif isinstance(tool, dict):
            new_tools.append(tool)
    return new_tools


def cleanup_none_field_in_message(message: AllMessageValues):
    """
    Cleans up the message by removing the none field.

    remove None fields in the message - e.g. {"function": None} - some providers raise validation errors
    """
    new_message = message.copy()
    return {k: v for k, v in new_message.items() if v is not None}


def validate_chat_completion_user_messages(messages: List[AllMessageValues]):
    """
    Ensures all user messages are valid OpenAI chat completion messages.

    Args:
        messages: List of message dictionaries
        message_content_type: Type to validate content against

    Returns:
        List[dict]: The validated messages

    Raises:
        ValueError: If any message is invalid
    """
    for idx, m in enumerate(messages):
        try:
            if m["role"] == "user":
                user_content = m.get("content")
                if user_content is not None:
                    if isinstance(user_content, str):
                        continue
                    elif isinstance(user_content, list):
                        for item in user_content:
                            if isinstance(item, dict):
                                if item.get("type") not in ValidUserMessageContentTypes:
                                    raise Exception("invalid content type")
        except Exception as e:
            if isinstance(e, KeyError):
                raise Exception(
                    f"Invalid message={m} at index {idx}. Please ensure all messages are valid OpenAI chat completion messages."
                )
            if "invalid content type" in str(e):
                raise Exception(
                    f"Invalid user message={m} at index {idx}. Please ensure all user messages are valid OpenAI chat completion messages."
                )
            else:
                raise e

    return messages


def validate_chat_completion_tool_choice(
    tool_choice: Optional[Union[dict, str]],
) -> Optional[Union[dict, str]]:
    """
    Confirm the tool choice is passed in the OpenAI format.

    Prevents user errors like: https://github.com/BerriAI/litellm/issues/7483
    """
    from litellm.types.llms.openai import (
        ChatCompletionToolChoiceObjectParam,
        ChatCompletionToolChoiceStringValues,
    )

    if tool_choice is None:
        return tool_choice
    elif isinstance(tool_choice, str):
        return tool_choice
    elif isinstance(tool_choice, dict):
        # Handle Cursor IDE format: {"type": "auto"} -> return as-is
        if (
            tool_choice.get("type") in ["auto", "none", "required"]
            and "function" not in tool_choice
        ):
            return tool_choice

        # Standard OpenAI format: {"type": "function", "function": {...}}
        if tool_choice.get("type") is None or tool_choice.get("function") is None:
            raise Exception(
                f"Invalid tool choice, tool_choice={tool_choice}. Please ensure tool_choice follows the OpenAI spec"
            )
        return tool_choice
    raise Exception(
        f"Invalid tool choice, tool_choice={tool_choice}. Got={type(tool_choice)}. Expecting str, or dict. Please ensure tool_choice follows the OpenAI tool_choice spec"
    )


class ProviderConfigManager:
    @staticmethod
    def get_provider_chat_config(  # noqa: PLR0915
        model: str, provider: LlmProviders
    ) -> Optional[BaseConfig]:
        """
        Returns the provider config for a given provider.
        """

        if (
            provider == LlmProviders.OPENAI
            and litellm.openaiOSeriesConfig.is_model_o_series_model(model=model)
        ):
            return litellm.openaiOSeriesConfig
        elif (
            provider == LlmProviders.OPENAI
            and litellm.OpenAIGPT5Config.is_model_gpt_5_model(model=model)
        ):
            return litellm.OpenAIGPT5Config()
        elif litellm.LlmProviders.DEEPSEEK == provider:
            return litellm.DeepSeekChatConfig()
        elif litellm.LlmProviders.GROQ == provider:
            return litellm.GroqChatConfig()
        elif litellm.LlmProviders.BYTEZ == provider:
            return litellm.BytezChatConfig()
        elif litellm.LlmProviders.DATABRICKS == provider:
            return litellm.DatabricksConfig()
        elif litellm.LlmProviders.XAI == provider:
            return litellm.XAIChatConfig()
        elif litellm.LlmProviders.LAMBDA_AI == provider:
            return litellm.LambdaAIChatConfig()
        elif litellm.LlmProviders.LLAMA == provider:
            return litellm.LlamaAPIConfig()
        elif litellm.LlmProviders.TEXT_COMPLETION_OPENAI == provider:
            return litellm.OpenAITextCompletionConfig()
        elif litellm.LlmProviders.COHERE_CHAT == provider:
            return litellm.CohereChatConfig()
        elif litellm.LlmProviders.COHERE == provider:
            return litellm.CohereConfig()
        elif litellm.LlmProviders.SNOWFLAKE == provider:
            return litellm.SnowflakeConfig()
        elif litellm.LlmProviders.CLARIFAI == provider:
            return litellm.ClarifaiConfig()
        elif litellm.LlmProviders.ANTHROPIC == provider:
            return litellm.AnthropicConfig()
        elif litellm.LlmProviders.ANTHROPIC_TEXT == provider:
            return litellm.AnthropicTextConfig()
        elif litellm.LlmProviders.VERTEX_AI_BETA == provider:
            return litellm.VertexGeminiConfig()
        elif litellm.LlmProviders.VERTEX_AI == provider:
            if "gemini" in model:
                return litellm.VertexGeminiConfig()
            elif "claude" in model:
                return litellm.VertexAIAnthropicConfig()
            elif "gpt-oss" in model:
                from litellm.llms.vertex_ai.vertex_ai_partner_models.gpt_oss.transformation import (
                    VertexAIGPTOSSTransformation,
                )

                return VertexAIGPTOSSTransformation()
            elif model in litellm.vertex_mistral_models:
                if "codestral" in model:
                    return litellm.CodestralTextCompletionConfig()
                else:
                    return litellm.MistralConfig()
            elif model in litellm.vertex_ai_ai21_models:
                return litellm.VertexAIAi21Config()
            else:  # use generic openai-like param mapping
                return litellm.VertexAILlama3Config()
        elif litellm.LlmProviders.CLOUDFLARE == provider:
            return litellm.CloudflareChatConfig()
        elif litellm.LlmProviders.SAGEMAKER_CHAT == provider:
            return litellm.SagemakerChatConfig()
        elif litellm.LlmProviders.SAGEMAKER == provider:
            return litellm.SagemakerConfig()
        elif litellm.LlmProviders.FIREWORKS_AI == provider:
            return litellm.FireworksAIConfig()
        elif litellm.LlmProviders.FRIENDLIAI == provider:
            return litellm.FriendliaiChatConfig()
        elif litellm.LlmProviders.WATSONX == provider:
            return litellm.IBMWatsonXChatConfig()
        elif litellm.LlmProviders.WATSONX_TEXT == provider:
            return litellm.IBMWatsonXAIConfig()
        elif litellm.LlmProviders.EMPOWER == provider:
            return litellm.EmpowerChatConfig()
        elif litellm.LlmProviders.GITHUB == provider:
            return litellm.GithubChatConfig()
        elif litellm.LlmProviders.GITHUB_COPILOT == provider:
            return litellm.GithubCopilotConfig()
        elif (
            litellm.LlmProviders.CUSTOM == provider
            or litellm.LlmProviders.CUSTOM_OPENAI == provider
            or litellm.LlmProviders.OPENAI_LIKE == provider
        ):
            return litellm.OpenAILikeChatConfig()
        elif litellm.LlmProviders.AIOHTTP_OPENAI == provider:
            return litellm.AiohttpOpenAIChatConfig()
        elif litellm.LlmProviders.HOSTED_VLLM == provider:
            return litellm.HostedVLLMChatConfig()
        elif litellm.LlmProviders.LLAMAFILE == provider:
            return litellm.LlamafileChatConfig()
        elif litellm.LlmProviders.LM_STUDIO == provider:
            return litellm.LMStudioChatConfig()
        elif litellm.LlmProviders.GALADRIEL == provider:
            return litellm.GaladrielChatConfig()
        elif litellm.LlmProviders.REPLICATE == provider:
            return litellm.ReplicateConfig()
        elif litellm.LlmProviders.HUGGINGFACE == provider:
            return litellm.HuggingFaceChatConfig()
        elif litellm.LlmProviders.TOGETHER_AI == provider:
            return litellm.TogetherAIConfig()
        elif litellm.LlmProviders.OPENROUTER == provider:
            return litellm.OpenrouterConfig()
        elif litellm.LlmProviders.VERCEL_AI_GATEWAY == provider:
            return litellm.VercelAIGatewayConfig()
        elif litellm.LlmProviders.COMETAPI == provider:
            return litellm.CometAPIConfig()
        elif litellm.LlmProviders.DATAROBOT == provider:
            return litellm.DataRobotConfig()
        elif litellm.LlmProviders.GEMINI == provider:
            return litellm.GoogleAIStudioGeminiConfig()
        elif (
            litellm.LlmProviders.AI21 == provider
            or litellm.LlmProviders.AI21_CHAT == provider
        ):
            return litellm.AI21ChatConfig()
        elif litellm.LlmProviders.AZURE == provider:
            if litellm.AzureOpenAIO1Config().is_o_series_model(model=model):
                return litellm.AzureOpenAIO1Config()
            if litellm.AzureOpenAIGPT5Config.is_model_gpt_5_model(model=model):
                return litellm.AzureOpenAIGPT5Config()
            return litellm.AzureOpenAIConfig()
        elif litellm.LlmProviders.AZURE_AI == provider:
            return litellm.AzureAIStudioConfig()
        elif litellm.LlmProviders.AZURE_TEXT == provider:
            return litellm.AzureOpenAITextConfig()
        elif litellm.LlmProviders.HOSTED_VLLM == provider:
            return litellm.HostedVLLMChatConfig()
        elif litellm.LlmProviders.NLP_CLOUD == provider:
            return litellm.NLPCloudConfig()
        elif litellm.LlmProviders.OOBABOOGA == provider:
            return litellm.OobaboogaConfig()
        elif litellm.LlmProviders.OLLAMA_CHAT == provider:
            return litellm.OllamaChatConfig()
        elif litellm.LlmProviders.DEEPINFRA == provider:
            return litellm.DeepInfraConfig()
        elif litellm.LlmProviders.PERPLEXITY == provider:
            return litellm.PerplexityChatConfig()
        elif (
            litellm.LlmProviders.MISTRAL == provider
            or litellm.LlmProviders.CODESTRAL == provider
        ):
            return litellm.MistralConfig()
        elif litellm.LlmProviders.NVIDIA_NIM == provider:
            return litellm.NvidiaNimConfig()
        elif litellm.LlmProviders.CEREBRAS == provider:
            return litellm.CerebrasConfig()
        elif litellm.LlmProviders.BASETEN == provider:
            return litellm.BasetenConfig()
        elif litellm.LlmProviders.VOLCENGINE == provider:
            return litellm.VolcEngineConfig()
        elif litellm.LlmProviders.TEXT_COMPLETION_CODESTRAL == provider:
            return litellm.CodestralTextCompletionConfig()
        elif litellm.LlmProviders.SAMBANOVA == provider:
            return litellm.SambanovaConfig()
        elif litellm.LlmProviders.MARITALK == provider:
            return litellm.MaritalkConfig()
        elif litellm.LlmProviders.CLOUDFLARE == provider:
            return litellm.CloudflareChatConfig()
        elif litellm.LlmProviders.ANTHROPIC_TEXT == provider:
            return litellm.AnthropicTextConfig()
        elif litellm.LlmProviders.VLLM == provider:
            return litellm.VLLMConfig()
        elif litellm.LlmProviders.OLLAMA == provider:
            return litellm.OllamaConfig()
        elif litellm.LlmProviders.PREDIBASE == provider:
            return litellm.PredibaseConfig()
        elif litellm.LlmProviders.TRITON == provider:
            return litellm.TritonConfig()
        elif litellm.LlmProviders.PETALS == provider:
            return litellm.PetalsConfig()
        elif litellm.LlmProviders.FEATHERLESS_AI == provider:
            return litellm.FeatherlessAIConfig()
        elif litellm.LlmProviders.NOVITA == provider:
            return litellm.NovitaConfig()
        elif litellm.LlmProviders.NEBIUS == provider:
            return litellm.NebiusConfig()
        elif litellm.LlmProviders.DASHSCOPE == provider:
            return litellm.DashScopeChatConfig()
        elif litellm.LlmProviders.MOONSHOT == provider:
            return litellm.MoonshotChatConfig()
        elif litellm.LlmProviders.V0 == provider:
            return litellm.V0ChatConfig()
        elif litellm.LlmProviders.MORPH == provider:
            return litellm.MorphChatConfig()
        elif litellm.LlmProviders.BEDROCK == provider:
            bedrock_route = BedrockModelInfo.get_bedrock_route(model)
            bedrock_invoke_provider = litellm.BedrockLLM.get_bedrock_invoke_provider(
                model=model
            )

            base_model = BedrockModelInfo.get_base_model(model)

            if bedrock_route == "converse" or bedrock_route == "converse_like":
                return litellm.AmazonConverseConfig()
            elif bedrock_route == "agent":
                from litellm.llms.bedrock.chat.invoke_agent.transformation import (
                    AmazonInvokeAgentConfig,
                )

                return AmazonInvokeAgentConfig()
            elif bedrock_invoke_provider == "amazon":  # amazon titan llms
                return litellm.AmazonTitanConfig()
            elif bedrock_invoke_provider == "anthropic":
                if (
                    base_model
                    in litellm.AmazonAnthropicConfig.get_legacy_anthropic_model_names()
                ):
                    return litellm.AmazonAnthropicConfig()
                else:
                    return litellm.AmazonAnthropicClaudeConfig()
            elif (
                bedrock_invoke_provider == "meta" or bedrock_invoke_provider == "llama"
            ):  # amazon / meta llms
                return litellm.AmazonLlamaConfig()
            elif bedrock_invoke_provider == "ai21":  # ai21 llms
                return litellm.AmazonAI21Config()
            elif bedrock_invoke_provider == "cohere":  # cohere models on bedrock
                return litellm.AmazonCohereConfig()
            elif bedrock_invoke_provider == "mistral":  # mistral models on bedrock
                return litellm.AmazonMistralConfig()
            elif bedrock_invoke_provider == "deepseek_r1":  # deepseek models on bedrock
                return litellm.AmazonDeepSeekR1Config()
            elif bedrock_invoke_provider == "nova":
                return litellm.AmazonInvokeNovaConfig()
            else:
                return litellm.AmazonInvokeConfig()
        elif litellm.LlmProviders.LITELLM_PROXY == provider:
            return litellm.LiteLLMProxyChatConfig()
        elif litellm.LlmProviders.OPENAI == provider:
            return litellm.OpenAIGPTConfig()
        elif litellm.LlmProviders.GRADIENT_AI == provider:
            return litellm.GradientAIConfig()
        elif litellm.LlmProviders.NSCALE == provider:
            return litellm.NscaleConfig()
        elif litellm.LlmProviders.HEROKU == provider:
            return litellm.HerokuChatConfig()
        elif litellm.LlmProviders.OCI == provider:
            return litellm.OCIChatConfig()
        elif litellm.LlmProviders.HYPERBOLIC == provider:
            return litellm.HyperbolicChatConfig()
        return None

    @staticmethod
    def get_provider_embedding_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseEmbeddingConfig]:
        if (
            litellm.LlmProviders.VOYAGE == provider
            and litellm.VoyageContextualEmbeddingConfig.is_contextualized_embeddings(
                model
            )
        ):
            return litellm.VoyageContextualEmbeddingConfig()
        elif litellm.LlmProviders.VOYAGE == provider:
            return litellm.VoyageEmbeddingConfig()
        elif litellm.LlmProviders.TRITON == provider:
            return litellm.TritonEmbeddingConfig()
        elif litellm.LlmProviders.WATSONX == provider:
            return litellm.IBMWatsonXEmbeddingConfig()
        elif litellm.LlmProviders.INFINITY == provider:
            return litellm.InfinityEmbeddingConfig()
        elif litellm.LlmProviders.SAMBANOVA == provider:
            return litellm.SambaNovaEmbeddingConfig()
        elif (
            litellm.LlmProviders.COHERE == provider
            or litellm.LlmProviders.COHERE_CHAT == provider
        ):
            from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig

            return CohereEmbeddingConfig()
        elif litellm.LlmProviders.JINA_AI == provider:
            from litellm.llms.jina_ai.embedding.transformation import (
                JinaAIEmbeddingConfig,
            )

            return JinaAIEmbeddingConfig()
        elif litellm.LlmProviders.VOLCENGINE == provider:
            from litellm.llms.volcengine.embedding.transformation import (
                VolcEngineEmbeddingConfig,
            )

            return VolcEngineEmbeddingConfig()
        return None

    @staticmethod
    def get_provider_rerank_config(
        model: str,
        provider: LlmProviders,
        api_base: Optional[str],
        present_version_params: List[str],
    ) -> BaseRerankConfig:
        if (
            litellm.LlmProviders.COHERE == provider
            or litellm.LlmProviders.COHERE_CHAT == provider
        ):
            if should_use_cohere_v1_client(api_base, present_version_params):
                return litellm.CohereRerankConfig()
            else:
                return litellm.CohereRerankV2Config()
        elif litellm.LlmProviders.AZURE_AI == provider:
            return litellm.AzureAIRerankConfig()
        elif litellm.LlmProviders.INFINITY == provider:
            return litellm.InfinityRerankConfig()
        elif litellm.LlmProviders.JINA_AI == provider:
            return litellm.JinaAIRerankConfig()
        elif litellm.LlmProviders.HUGGINGFACE == provider:
            return litellm.HuggingFaceRerankConfig()
        elif litellm.LlmProviders.DEEPINFRA == provider:
            return litellm.DeepinfraRerankConfig()
        return litellm.CohereRerankConfig()

    @staticmethod
    def get_provider_anthropic_messages_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseAnthropicMessagesConfig]:
        if litellm.LlmProviders.ANTHROPIC == provider:
            return litellm.AnthropicMessagesConfig()
        # The 'BEDROCK' provider corresponds to Amazon's implementation of Anthropic Claude v3.
        # This mapping ensures that the correct configuration is returned for BEDROCK.
        elif litellm.LlmProviders.BEDROCK == provider:
            from litellm.llms.bedrock.common_utils import BedrockModelInfo

            return BedrockModelInfo.get_bedrock_provider_config_for_messages_api(model)
        elif litellm.LlmProviders.VERTEX_AI == provider:
            if "claude" in model:
                from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
                    VertexAIPartnerModelsAnthropicMessagesConfig,
                )

                return VertexAIPartnerModelsAnthropicMessagesConfig()
        return None

    @staticmethod
    def get_provider_audio_transcription_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseAudioTranscriptionConfig]:
        if litellm.LlmProviders.FIREWORKS_AI == provider:
            return litellm.FireworksAIAudioTranscriptionConfig()
        elif litellm.LlmProviders.DEEPGRAM == provider:
            return litellm.DeepgramAudioTranscriptionConfig()
        elif litellm.LlmProviders.ELEVENLABS == provider:
            from litellm.llms.elevenlabs.audio_transcription.transformation import (
                ElevenLabsAudioTranscriptionConfig,
            )

            return ElevenLabsAudioTranscriptionConfig()
        elif litellm.LlmProviders.OPENAI == provider:
            if "gpt-4o" in model:
                return litellm.OpenAIGPTAudioTranscriptionConfig()
            else:
                return litellm.OpenAIWhisperAudioTranscriptionConfig()
        return None

    @staticmethod
    def get_provider_responses_api_config(
        provider: LlmProviders,
        model: Optional[str] = None,
    ) -> Optional[BaseResponsesAPIConfig]:
        if litellm.LlmProviders.OPENAI == provider:
            return litellm.OpenAIResponsesAPIConfig()
        elif litellm.LlmProviders.AZURE == provider:
            # Check if it's an O-series model
            if model and ("o_series" in model.lower() or supports_reasoning(model)):
                return litellm.AzureOpenAIOSeriesResponsesAPIConfig()
            else:
                return litellm.AzureOpenAIResponsesAPIConfig()
        return None

    @staticmethod
    def get_provider_text_completion_config(
        model: str,
        provider: LlmProviders,
    ) -> BaseTextCompletionConfig:
        if LlmProviders.FIREWORKS_AI == provider:
            return litellm.FireworksAITextCompletionConfig()
        elif LlmProviders.TOGETHER_AI == provider:
            return litellm.TogetherAITextCompletionConfig()
        return litellm.OpenAITextCompletionConfig()

    @staticmethod
    def get_provider_model_info(
        model: Optional[str],
        provider: LlmProviders,
    ) -> Optional[BaseLLMModelInfo]:
        if LlmProviders.FIREWORKS_AI == provider:
            return litellm.FireworksAIConfig()
        elif LlmProviders.OPENAI == provider:
            return litellm.OpenAIGPTConfig()
        elif LlmProviders.GEMINI == provider:
            return litellm.GeminiModelInfo()
        elif LlmProviders.VERTEX_AI == provider:
            from litellm.llms.vertex_ai.common_utils import VertexAIModelInfo

            return VertexAIModelInfo()
        elif LlmProviders.LITELLM_PROXY == provider:
            return litellm.LiteLLMProxyChatConfig()
        elif LlmProviders.TOPAZ == provider:
            return litellm.TopazModelInfo()
        elif LlmProviders.ANTHROPIC == provider:
            return litellm.AnthropicModelInfo()
        elif LlmProviders.XAI == provider:
            return litellm.XAIModelInfo()
        elif LlmProviders.OLLAMA == provider or LlmProviders.OLLAMA_CHAT == provider:
            # Dynamic model listing for Ollama server
            from litellm.llms.ollama.common_utils import OllamaModelInfo

            return OllamaModelInfo()
        elif LlmProviders.VLLM == provider or LlmProviders.HOSTED_VLLM == provider:
            from litellm.llms.vllm.common_utils import (
                VLLMModelInfo,  # experimental approach, to reduce bloat on __init__.py
            )

            return VLLMModelInfo()
        return None

    @staticmethod
    def get_provider_passthrough_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BasePassthroughConfig]:
        if LlmProviders.BEDROCK == provider:
            from litellm.llms.bedrock.passthrough.transformation import (
                BedrockPassthroughConfig,
            )

            return BedrockPassthroughConfig()
        elif LlmProviders.VLLM == provider:
            from litellm.llms.vllm.passthrough.transformation import (
                VLLMPassthroughConfig,
            )

            return VLLMPassthroughConfig()

        return None

    @staticmethod
    def get_provider_image_variation_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseImageVariationConfig]:
        if LlmProviders.OPENAI == provider:
            return litellm.OpenAIImageVariationConfig()
        elif LlmProviders.TOPAZ == provider:
            return litellm.TopazImageVariationConfig()
        return None

    @staticmethod
    def get_provider_files_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseFilesConfig]:
        if LlmProviders.GEMINI == provider:
            from litellm.llms.gemini.files.transformation import (
                GoogleAIStudioFilesHandler,  # experimental approach, to reduce bloat on __init__.py
            )

            return GoogleAIStudioFilesHandler()
        elif LlmProviders.VERTEX_AI == provider:
            from litellm.llms.vertex_ai.files.transformation import VertexAIFilesConfig

            return VertexAIFilesConfig()
        elif LlmProviders.BEDROCK == provider:
            from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

            return BedrockFilesConfig()
        return None

    @staticmethod
    def get_provider_batches_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseBatchesConfig]:
        if LlmProviders.BEDROCK == provider:
            from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig

            return BedrockBatchesConfig()
        return None

    @staticmethod
    def get_provider_vector_store_config(
        provider: LlmProviders,
    ) -> Optional[CustomLogger]:
        from litellm.integrations.vector_store_integrations.bedrock_vector_store import (
            BedrockVectorStore,
        )

        if LlmProviders.BEDROCK == provider:
            return BedrockVectorStore.get_initialized_custom_logger()
        return None

    @staticmethod
    def get_provider_vector_stores_config(
        provider: LlmProviders,
    ) -> Optional[BaseVectorStoreConfig]:
        """
        v2 vector store config, use this for new vector store integrations
        """
        if litellm.LlmProviders.OPENAI == provider:
            from litellm.llms.openai.vector_stores.transformation import (
                OpenAIVectorStoreConfig,
            )

            return OpenAIVectorStoreConfig()
        elif litellm.LlmProviders.AZURE == provider:
            from litellm.llms.azure.vector_stores.transformation import (
                AzureOpenAIVectorStoreConfig,
            )

            return AzureOpenAIVectorStoreConfig()
        elif litellm.LlmProviders.VERTEX_AI == provider:
            from litellm.llms.vertex_ai.vector_stores.transformation import (
                VertexVectorStoreConfig,
            )

            return VertexVectorStoreConfig()
        elif litellm.LlmProviders.BEDROCK == provider:
            from litellm.llms.bedrock.vector_stores.transformation import (
                BedrockVectorStoreConfig,
            )

            return BedrockVectorStoreConfig()
        elif litellm.LlmProviders.PG_VECTOR == provider:
            from litellm.llms.pg_vector.vector_stores.transformation import (
                PGVectorStoreConfig,
            )

            return PGVectorStoreConfig()
        return None

    @staticmethod
    def get_provider_image_generation_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseImageGenerationConfig]:
        if LlmProviders.OPENAI == provider:
            from litellm.llms.openai.image_generation import (
                get_openai_image_generation_config,
            )

            return get_openai_image_generation_config(model)
        elif LlmProviders.AZURE == provider:
            from litellm.llms.azure.image_generation import (
                get_azure_image_generation_config,
            )

            return get_azure_image_generation_config(model)
        elif LlmProviders.AZURE_AI == provider:
            from litellm.llms.azure_ai.image_generation import (
                get_azure_ai_image_generation_config,
            )

            return get_azure_ai_image_generation_config(model)
        elif LlmProviders.XINFERENCE == provider:
            from litellm.llms.xinference.image_generation import (
                get_xinference_image_generation_config,
            )

            return get_xinference_image_generation_config(model)
        elif LlmProviders.RECRAFT == provider:
            from litellm.llms.recraft.image_generation import (
                get_recraft_image_generation_config,
            )

            return get_recraft_image_generation_config(model)
        elif LlmProviders.AIML == provider:
            from litellm.llms.aiml.image_generation import (
                get_aiml_image_generation_config,
            )

            return get_aiml_image_generation_config(model)
        elif LlmProviders.GEMINI == provider:
            from litellm.llms.gemini.image_generation import (
                get_gemini_image_generation_config,
            )

            return get_gemini_image_generation_config(model)
        elif LlmProviders.LITELLM_PROXY == provider:
            from litellm.llms.litellm_proxy.image_generation.transformation import (
                LiteLLMProxyImageGenerationConfig,
            )

            return LiteLLMProxyImageGenerationConfig()
        return None

    @staticmethod
    def get_provider_realtime_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseRealtimeConfig]:
        if LlmProviders.GEMINI == provider:
            from litellm.llms.gemini.realtime.transformation import GeminiRealtimeConfig

            return GeminiRealtimeConfig()
        return None

    @staticmethod
    def get_provider_image_edit_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseImageEditConfig]:
        if LlmProviders.OPENAI == provider:
            from litellm.llms.openai.image_edit.transformation import (
                OpenAIImageEditConfig,
            )

            return OpenAIImageEditConfig()
        elif LlmProviders.AZURE == provider:
            from litellm.llms.azure.image_edit.transformation import (
                AzureImageEditConfig,
            )

            return AzureImageEditConfig()
        elif LlmProviders.RECRAFT == provider:
            from litellm.llms.recraft.image_edit.transformation import (
                RecraftImageEditConfig,
            )

            return RecraftImageEditConfig()
        elif LlmProviders.LITELLM_PROXY == provider:
            from litellm.llms.litellm_proxy.image_edit.transformation import (
                LiteLLMProxyImageEditConfig,
            )

            return LiteLLMProxyImageEditConfig()
        return None

    @staticmethod
    def get_provider_google_genai_generate_content_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseGoogleGenAIGenerateContentConfig]:
        if litellm.LlmProviders.GEMINI == provider:
            from litellm.llms.gemini.google_genai.transformation import (
                GoogleGenAIConfig,
            )

            return GoogleGenAIConfig()
        elif litellm.LlmProviders.VERTEX_AI == provider:
            from litellm.llms.vertex_ai.google_genai.transformation import (
                VertexAIGoogleGenAIConfig,
            )
            from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
                VertexAIPartnerModels,
            )

            #########################################################
            # If Vertex Partner models like Anthropic, Mistral, etc. are used,
            # return None as we want this to go through the litellm.completion() adapter
            # and not the Google Gen AI adapter
            #########################################################
            if VertexAIPartnerModels.is_vertex_partner_model(model):
                return None

            #########################################################
            # If the model is not a Vertex Partner model, return the Vertex AI Google Gen AI Config
            # This is for Vertex `gemini` models
            #########################################################
            return VertexAIGoogleGenAIConfig()
        return None


def get_end_user_id_for_cost_tracking(
    litellm_params: dict,
    service_type: Literal["litellm_logging", "prometheus"] = "litellm_logging",
) -> Optional[str]:
    """
    Used for enforcing `disable_end_user_cost_tracking` param.

    service_type: "litellm_logging" or "prometheus" - used to allow prometheus only disable cost tracking.
    """
    _metadata = cast(dict, litellm_params.get("metadata", {}) or {})

    end_user_id = cast(
        Optional[str],
        litellm_params.get("user_api_key_end_user_id")
        or _metadata.get("user_api_key_end_user_id"),
    )
    if litellm.disable_end_user_cost_tracking:
        return None

    #######################################
    # By default we don't track end_user on prometheus since we don't want to increase cardinality
    # by default litellm.enable_end_user_cost_tracking_prometheus_only is None, so we don't track end_user on prometheus
    #######################################
    if service_type == "prometheus":
        if litellm.enable_end_user_cost_tracking_prometheus_only is not True:
            return None
    return end_user_id


def should_use_cohere_v1_client(
    api_base: Optional[str], present_version_params: List[str]
):
    if not api_base:
        return False
    uses_v1_params = ("max_chunks_per_doc" in present_version_params) and (
        "max_tokens_per_doc" not in present_version_params
    )
    return api_base.endswith("/v1/rerank") or (
        uses_v1_params and not api_base.endswith("/v2/rerank")
    )


def is_prompt_caching_valid_prompt(
    model: str,
    messages: Optional[List[AllMessageValues]],
    tools: Optional[List[ChatCompletionToolParam]] = None,
    custom_llm_provider: Optional[str] = None,
) -> bool:
    """
    Returns true if the prompt is valid for prompt caching.

    OpenAI + Anthropic providers have a minimum token count of 1024 for prompt caching.
    """
    try:
        if messages is None and tools is None:
            return False
        if custom_llm_provider is not None and not model.startswith(
            custom_llm_provider
        ):
            model = custom_llm_provider + "/" + model
        token_count = token_counter(
            messages=messages,
            tools=tools,
            model=model,
            use_default_image_token_count=True,
        )
        return token_count >= MINIMUM_PROMPT_CACHE_TOKEN_COUNT
    except Exception as e:
        verbose_logger.error(f"Error in is_prompt_caching_valid_prompt: {e}")
        return False


def extract_duration_from_srt_or_vtt(srt_or_vtt_content: str) -> Optional[float]:
    """
    Extracts the total duration (in seconds) from SRT or VTT content.

    Args:
        srt_or_vtt_content (str): The content of an SRT or VTT file as a string.

    Returns:
        Optional[float]: The total duration in seconds, or None if no timestamps are found.
    """
    # Regular expression to match timestamps in the format "hh:mm:ss,ms" or "hh:mm:ss.ms"
    timestamp_pattern = r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"

    timestamps = re.findall(timestamp_pattern, srt_or_vtt_content)

    if not timestamps:
        return None

    # Convert timestamps to seconds and find the max (end time)
    durations = []
    for match in timestamps:
        hours, minutes, seconds, milliseconds = map(int, match)
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        durations.append(total_seconds)

    return max(durations) if durations else None


import httpx


def _add_path_to_api_base(api_base: str, ending_path: str) -> str:
    """
    Adds an ending path to an API base URL while preventing duplicate path segments.

    Args:
        api_base: Base URL string
        ending_path: Path to append to the base URL

    Returns:
        Modified URL string with proper path handling
    """
    original_url = httpx.URL(api_base)
    base_url = original_url.copy_with(params={})  # Removes query params
    base_path = original_url.path.rstrip("/")
    end_path = ending_path.lstrip("/")

    # Split paths into segments
    base_segments = [s for s in base_path.split("/") if s]
    end_segments = [s for s in end_path.split("/") if s]

    # Find overlapping segments from the end of base_path and start of ending_path
    final_segments = []
    for i in range(len(base_segments)):
        if base_segments[i:] == end_segments[: len(base_segments) - i]:
            final_segments = base_segments[:i] + end_segments
            break
    else:
        # No overlap found, just combine all segments
        final_segments = base_segments + end_segments

    # Construct the new path
    modified_path = "/" + "/".join(final_segments)
    modified_url = base_url.copy_with(path=modified_path)

    # Re-add the original query parameters
    return str(modified_url.copy_with(params=original_url.params))


def get_standard_openai_params(params: dict) -> dict:
    return {
        k: v
        for k, v in params.items()
        if k in litellm.OPENAI_CHAT_COMPLETION_PARAMS and v is not None
    }


def get_non_default_completion_params(kwargs: dict) -> dict:
    openai_params = litellm.OPENAI_CHAT_COMPLETION_PARAMS
    default_params = openai_params + all_litellm_params
    non_default_params = {
        k: v for k, v in kwargs.items() if k not in default_params
    }  # model-specific params - pass them straight to the model/provider

    return non_default_params


def get_non_default_transcription_params(kwargs: dict) -> dict:
    from litellm.constants import OPENAI_TRANSCRIPTION_PARAMS

    default_params = OPENAI_TRANSCRIPTION_PARAMS + all_litellm_params
    non_default_params = {k: v for k, v in kwargs.items() if k not in default_params}
    return non_default_params


def add_openai_metadata(metadata: dict) -> dict:
    """
    Add metadata to openai optional parameters, excluding hidden params.

    OpenAI 'metadata' only supports string values.

    Args:
        params (dict): Dictionary of API parameters
        metadata (dict, optional): Metadata to include in the request

    Returns:
        dict: Updated parameters dictionary with visible metadata only
    """
    if metadata is None:
        return None
    # Only include non-hidden parameters
    visible_metadata = {
        k: v
        for k, v in metadata.items()
        if k != "hidden_params" and isinstance(v, (str))
    }

    # max 16 keys allowed by openai - trim down to 16
    if len(visible_metadata) > 16:
        filtered_metadata = {}
        idx = 0
        for k, v in visible_metadata.items():
            if idx < 16:
                filtered_metadata[k] = v
            idx += 1
        visible_metadata = filtered_metadata

    return visible_metadata.copy()


def return_raw_request(endpoint: CallTypes, kwargs: dict) -> RawRequestTypedDict:
    """
    Return the json str of the request

    This is currently in BETA, and tested for `/chat/completions` -> `litellm.completion` calls.
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
        log_raw_request_response=True,
    )

    llm_api_endpoint = getattr(litellm, endpoint.value)

    received_exception = ""

    try:
        llm_api_endpoint(
            **kwargs,
            litellm_logging_obj=litellm_logging_obj,
            api_key="my-fake-api-key",  # 👈 ensure the request fails
        )
    except Exception as e:
        received_exception = str(e)

    raw_request_typed_dict = litellm_logging_obj.model_call_details.get(
        "raw_request_typed_dict"
    )
    if raw_request_typed_dict:
        return cast(RawRequestTypedDict, raw_request_typed_dict)
    else:
        return RawRequestTypedDict(
            error=received_exception,
        )


def jsonify_tools(tools: List[Any]) -> List[Dict]:
    """
    Fixes https://github.com/BerriAI/litellm/issues/9321

    Where user passes in a pydantic base model
    """
    new_tools: List[Dict] = []
    for tool in tools:
        if isinstance(tool, BaseModel):
            tool = tool.model_dump(exclude_none=True)
        elif isinstance(tool, dict):
            tool = tool.copy()
        if isinstance(tool, dict):
            new_tools.append(tool)
    return new_tools


def get_empty_usage() -> Usage:
    return Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )


def should_run_mock_completion(
    mock_response: Optional[Any],
    mock_tool_calls: Optional[Any],
    mock_timeout: Optional[Any],
) -> bool:
    if mock_response or mock_tool_calls or mock_timeout:
        return True
    return False
