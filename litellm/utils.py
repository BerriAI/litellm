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
import uuid
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from importlib import resources
from inspect import iscoroutine
from os.path import abspath, dirname, join

import aiohttp
import dotenv
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
from litellm.caching._internal_lru_cache import lru_cache_wrapper
from litellm.caching.caching import DualCache
from litellm.caching.caching_handler import CachingHandlerResponse, LLMCachingHandler
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    map_finish_reason,
    process_response_headers,
)
from litellm.litellm_core_utils.default_encoding import encoding
from litellm.litellm_core_utils.exception_mapping_utils import (
    _get_response_headers,
    exception_type,
    get_error_message,
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
from litellm.litellm_core_utils.redact_messages import (
    LiteLLMLoggingObject,
    redact_message_input_output_from_logging,
)
from litellm.litellm_core_utils.rules import Rules
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.litellm_core_utils.token_counter import (
    calculate_img_tokens,
    get_modified_max_tokens,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.router_utils.get_retry_from_policy import (
    get_num_retries_from_retry_policy,
    reset_retry_policy,
)
from litellm.secret_managers.main import get_secret
from litellm.types.llms.openai import (
    AllMessageValues,
    AllPromptValues,
    ChatCompletionAssistantToolCall,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    OpenAITextCompletionUserMessage,
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
    CustomHuggingfaceTokenizer,
    Delta,
    Embedding,
    EmbeddingResponse,
    Function,
    ImageResponse,
    LlmProviders,
    LlmProvidersSet,
    Message,
    ModelInfo,
    ModelInfoBase,
    ModelResponse,
    ModelResponseStream,
    ProviderField,
    StreamingChoices,
    TextChoices,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
    all_litellm_params,
)

with resources.open_text(
    "litellm.litellm_core_utils.tokenizers", "anthropic_tokenizer.json"
) as f:
    json_data = json.load(f)
# Convert to str (if necessary)
claude_json_str = json.dumps(json_data)
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
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

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.base_llm.completion.transformation import BaseTextCompletionConfig
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.image_variations.transformation import (
    BaseImageVariationConfig,
)
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig

from ._logging import _is_debugging_on, verbose_logger
from .caching.caching import (
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
MAX_THREADS = 100

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)
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
genericAPILogger = None
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

        ## LOGGING SETUP
        function_id: Optional[str] = kwargs["id"] if "id" in kwargs else None

        if len(litellm.callbacks) > 0:
            for callback in litellm.callbacks:
                # check if callback is a string - e.g. "lago", "openmeter"
                if isinstance(callback, str):
                    callback = litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class(  # type: ignore
                        callback, internal_usage_cache=None, llm_router=None
                    )
                    if callback is None or any(
                        isinstance(cb, type(callback))
                        for cb in litellm._async_success_callback
                    ):  # don't double add a callback
                        continue
                if callback not in litellm.input_callback:
                    litellm.input_callback.append(callback)  # type: ignore
                if callback not in litellm.success_callback:
                    litellm.success_callback.append(callback)  # type: ignore
                if callback not in litellm.failure_callback:
                    litellm.failure_callback.append(callback)  # type: ignore
                if callback not in litellm._async_success_callback:
                    litellm._async_success_callback.append(callback)  # type: ignore
                if callback not in litellm._async_failure_callback:
                    litellm._async_failure_callback.append(callback)  # type: ignore
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
                    litellm._async_success_callback.append(callback)
                    removed_async_items.append(index)
                elif callback == "dynamodb" or callback == "openmeter":
                    # dynamo is an async callback, it's used for the proxy and needs to be async
                    # we only support async dynamo db logging for acompletion/aembedding since that's used on proxy
                    litellm._async_success_callback.append(callback)
                    removed_async_items.append(index)
                elif callback in litellm._known_custom_logger_compatible_callbacks:
                    from litellm.litellm_core_utils.litellm_logging import (
                        _init_custom_logger_compatible_class,
                    )

                    callback_class = _init_custom_logger_compatible_class(
                        callback,  # type: ignore
                        internal_usage_cache=None,
                        llm_router=None,  # type: ignore
                    )

                    # don't double add a callback
                    if callback_class is not None and not any(
                        isinstance(cb, type(callback_class)) for cb in litellm.callbacks
                    ):
                        litellm.callbacks.append(callback_class)  # type: ignore
                        litellm.input_callback.append(callback_class)  # type: ignore
                        litellm.success_callback.append(callback_class)  # type: ignore
                        litellm.failure_callback.append(callback_class)  # type: ignore
                        litellm._async_success_callback.append(callback_class)  # type: ignore
                        litellm._async_failure_callback.append(callback_class)  # type: ignore

            # Pop the async items from success_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                litellm.success_callback.pop(index)

        if len(litellm.failure_callback) > 0:
            removed_async_items = []
            for index, callback in enumerate(litellm.failure_callback):  # type: ignore
                if inspect.iscoroutinefunction(callback):
                    litellm._async_failure_callback.append(callback)
                    removed_async_items.append(index)

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
                details_to_log = copy.deepcopy(kwargs)
            except Exception:
                details_to_log = kwargs

            if litellm.turn_off_message_logging:
                # make a copy of the _model_Call_details and log it
                details_to_log.pop("messages", None)
                details_to_log.pop("input", None)
                details_to_log.pop("prompt", None)
            add_breadcrumb(
                category="litellm.llm_call",
                message=f"Positional Args: {args}, Keyword Args: {details_to_log}",
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
        else:
            messages = "default-message-value"
        stream = True if "stream" in kwargs and kwargs["stream"] is True else False
        logging_obj = LiteLLMLogging(
            model=model,
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
        verbose_logger.error(
            f"litellm.utils.py::function_setup() - [Non-Blocking] {traceback.format_exc()}; args - {args}; kwargs - {kwargs}"
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
        # check if user does not want this to be logged
        asyncio.create_task(
            logging_obj.async_success_handler(result, start_time, end_time)
        )
        logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=result,
            start_time=start_time,
            end_time=end_time,
        )


def client(original_function):  # noqa: PLR0915
    rules_obj = Rules()

    def check_coroutine(value) -> bool:
        if inspect.iscoroutine(value):
            return True
        elif inspect.iscoroutinefunction(value):
            return True
        else:
            return False

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

    def _get_num_retries(
        kwargs: Dict[str, Any], exception: Exception
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        num_retries = kwargs.get("num_retries", None) or litellm.num_retries or None
        if kwargs.get("retry_policy", None):
            num_retries = get_num_retries_from_retry_policy(
                exception=exception,
                retry_policy=kwargs.get("retry_policy"),
            )
            kwargs["retry_policy"] = reset_retry_policy()

        return num_retries, kwargs

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
            if "stream" in kwargs and kwargs["stream"] is True:
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
            if "stream" in kwargs and kwargs["stream"] is True:
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
            if hasattr(result, "_hidden_params"):
                result._hidden_params["model_id"] = kwargs.get("model_info", {}).get(
                    "id", None
                )
                result._hidden_params["api_base"] = get_api_base(
                    model=model or "",
                    optional_params=getattr(logging_obj, "optional_params", {}),
                )
                result._hidden_params["response_cost"] = (
                    logging_obj._response_cost_calculator(result=result)
                )

                result._hidden_params["additional_headers"] = process_response_headers(
                    result._hidden_params.get("additional_headers") or {}
                )  # GUARANTEE OPENAI HEADERS IN RESPONSE
            if result is not None:
                result._response_ms = (
                    end_time - start_time
                ).total_seconds() * 1000  # return response latency in ms like openai
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
            kwargs["litellm_logging_obj"] = logging_obj
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

            # MODEL CALL
            result = await original_function(*args, **kwargs)
            end_time = datetime.datetime.now()
            if "stream" in kwargs and kwargs["stream"] is True:
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
            elif call_type == CallTypes.arealtime.value:
                return result

            # ADD HIDDEN PARAMS - additional call metadata
            if hasattr(result, "_hidden_params"):
                result._hidden_params["litellm_call_id"] = getattr(
                    logging_obj, "litellm_call_id", None
                )
                result._hidden_params["model_id"] = kwargs.get("model_info", {}).get(
                    "id", None
                )
                result._hidden_params["api_base"] = get_api_base(
                    model=model or "",
                    optional_params=kwargs,
                )
                result._hidden_params["response_cost"] = (
                    logging_obj._response_cost_calculator(result=result)
                )
                result._hidden_params["additional_headers"] = process_response_headers(
                    result._hidden_params.get("additional_headers") or {}
                )  # GUARANTEE OPENAI HEADERS IN RESPONSE
            if (
                isinstance(result, ModelResponse)
                or isinstance(result, EmbeddingResponse)
                or isinstance(result, TranscriptionResponse)
            ):
                setattr(
                    result,
                    "_response_ms",
                    (end_time - start_time).total_seconds() * 1000,
                )  # return response latency in ms like openai

            ### POST-CALL RULES ###
            post_call_processing(
                original_response=result, model=model, optional_params=kwargs
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
            num_retries, kwargs = _get_num_retries(kwargs=kwargs, exception=e)
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


@lru_cache(maxsize=128)
def _select_tokenizer_helper(model: str):
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
    # default - tiktoken
    else:
        return {
            "type": "openai_tokenizer",
            "tokenizer": encoding,
        }  # default to openai tokenizer


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


def openai_token_counter(  # noqa: PLR0915
    messages: Optional[list] = None,
    model="gpt-3.5-turbo-0613",
    text: Optional[str] = None,
    is_tool_call: Optional[bool] = False,
    tools: Optional[List[ChatCompletionToolParam]] = None,
    tool_choice: Optional[ChatCompletionNamedToolChoiceParam] = None,
    count_response_tokens: Optional[
        bool
    ] = False,  # Flag passed from litellm.stream_chunk_builder, to indicate counting tokens for LLM Response. We need this because for LLM input we add +3 tokens per message - based on OpenAI's token counter
    use_default_image_token_count: Optional[bool] = False,
):
    """
    Return the number of tokens used by a list of messages.

    Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.
    """
    print_verbose(f"LiteLLM: Utils - Counting tokens for OpenAI model={model}")
    try:
        if "gpt-4o" in model:
            encoding = tiktoken.get_encoding("o200k_base")
        else:
            encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print_verbose("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model in litellm.open_ai_chat_completion_models:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model in litellm.azure_llms:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    includes_system_message = False

    if is_tool_call and text is not None:
        # if it's a tool call we assembled 'text' in token_counter()
        num_tokens = len(encoding.encode(text, disallowed_special=()))
    elif messages is not None:
        for message in messages:
            num_tokens += tokens_per_message
            if message.get("role", None) == "system":
                includes_system_message = True
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value, disallowed_special=()))
                    if key == "name":
                        num_tokens += tokens_per_name
                elif isinstance(value, List):
                    for c in value:
                        if c["type"] == "text":
                            text += c["text"]
                            num_tokens += len(
                                encoding.encode(c["text"], disallowed_special=())
                            )
                        elif c["type"] == "image_url":
                            if isinstance(c["image_url"], dict):
                                image_url_dict = c["image_url"]
                                detail = image_url_dict.get("detail", "auto")
                                url = image_url_dict.get("url")
                                num_tokens += calculate_img_tokens(
                                    data=url,
                                    mode=detail,
                                    use_default_image_token_count=use_default_image_token_count
                                    or False,
                                )
                            elif isinstance(c["image_url"], str):
                                image_url_str = c["image_url"]
                                num_tokens += calculate_img_tokens(
                                    data=image_url_str,
                                    mode="auto",
                                    use_default_image_token_count=use_default_image_token_count
                                    or False,
                                )
    elif text is not None and count_response_tokens is True:
        # This is the case where we need to count tokens for a streamed response. We should NOT add +3 tokens per message in this branch
        num_tokens = len(encoding.encode(text, disallowed_special=()))
        return num_tokens
    elif text is not None:
        num_tokens = len(encoding.encode(text, disallowed_special=()))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    if tools:
        num_tokens += len(encoding.encode(_format_function_definitions(tools)))
        num_tokens += 9  # Additional tokens for function definition of tools
    # If there's a system message and tools are present, subtract four tokens
    if tools and includes_system_message:
        num_tokens -= 4
    # If tool_choice is 'none', add one token.
    # If it's an object, add 4 + the number of tokens in the function name.
    # If it's undefined or 'auto', don't add anything.
    if tool_choice == "none":
        num_tokens += 1
    elif isinstance(tool_choice, dict):
        num_tokens += 7
        num_tokens += len(encoding.encode(tool_choice["function"]["name"]))

    return num_tokens


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
            identifier, revision=revision, auth_token=auth_token
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


def _format_function_definitions(tools):
    """Formats tool definitions in the format that OpenAI appears to use.
    Based on https://github.com/forestwanglin/openai-java/blob/main/jtokkit/src/main/java/xyz/felh/openai/jtokkit/utils/TikTokenUtils.java
    """
    lines = []
    lines.append("namespace functions {")
    lines.append("")
    for tool in tools:
        function = tool.get("function")
        if function_description := function.get("description"):
            lines.append(f"// {function_description}")
        function_name = function.get("name")
        parameters = function.get("parameters", {})
        properties = parameters.get("properties")
        if properties and properties.keys():
            lines.append(f"type {function_name} = (_: {{")
            lines.append(_format_object_parameters(parameters, 0))
            lines.append("}) => any;")
        else:
            lines.append(f"type {function_name} = () => any;")
        lines.append("")
    lines.append("} // namespace functions")
    return "\n".join(lines)


def _format_object_parameters(parameters, indent):
    properties = parameters.get("properties")
    if not properties:
        return ""
    required_params = parameters.get("required", [])
    lines = []
    for key, props in properties.items():
        description = props.get("description")
        if description:
            lines.append(f"// {description}")
        question = "?"
        if required_params and key in required_params:
            question = ""
        lines.append(f"{key}{question}: {_format_type(props, indent)},")
    return "\n".join([" " * max(0, indent) + line for line in lines])


def _format_type(props, indent):
    type = props.get("type")
    if type == "string":
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "string"
    elif type == "array":
        # items is required, OpenAI throws an error if it's missing
        return f"{_format_type(props['items'], indent)}[]"
    elif type == "object":
        return f"{{\n{_format_object_parameters(props, indent + 2)}\n}}"
    elif type in ["integer", "number"]:
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "number"
    elif type == "boolean":
        return "boolean"
    elif type == "null":
        return "null"
    else:
        # This is a guess, as an empty string doesn't yield the expected token count
        return "any"


def token_counter(
    model="",
    custom_tokenizer: Optional[dict] = None,
    text: Optional[Union[str, List[str]]] = None,
    messages: Optional[List] = None,
    count_response_tokens: Optional[bool] = False,
    tools: Optional[List[ChatCompletionToolParam]] = None,
    tool_choice: Optional[ChatCompletionNamedToolChoiceParam] = None,
    use_default_image_token_count: Optional[bool] = False,
) -> int:
    """
    Count the number of tokens in a given text using a specified model.

    Args:
    model (str): The name of the model to use for tokenization. Default is an empty string.
    custom_tokenizer (Optional[dict]): A custom tokenizer created with the `create_pretrained_tokenizer` or `create_tokenizer` method. Must be a dictionary with a string value for `type` and Tokenizer for `tokenizer`. Default is None.
    text (str): The raw text string to be passed to the model. Default is None.
    messages (Optional[List[Dict[str, str]]]): Alternative to passing in text. A list of dictionaries representing messages with "role" and "content" keys. Default is None.

    Returns:
    int: The number of tokens in the text.
    """
    # use tiktoken, anthropic, cohere, llama2, or llama3's tokenizer depending on the model
    is_tool_call = False
    num_tokens = 0
    if text is None:
        if messages is not None:
            print_verbose(f"token_counter messages received: {messages}")
            text = ""
            for message in messages:
                if message.get("content", None) is not None:
                    content = message.get("content")
                    if isinstance(content, str):
                        text += message["content"]
                    elif isinstance(content, List):
                        for c in content:
                            if c["type"] == "text":
                                text += c["text"]
                            elif c["type"] == "image_url":
                                if isinstance(c["image_url"], dict):
                                    image_url_dict = c["image_url"]
                                    detail = image_url_dict.get("detail", "auto")
                                    url = image_url_dict.get("url")
                                    num_tokens += calculate_img_tokens(
                                        data=url,
                                        mode=detail,
                                        use_default_image_token_count=use_default_image_token_count
                                        or False,
                                    )
                                elif isinstance(c["image_url"], str):
                                    image_url_str = c["image_url"]
                                    num_tokens += calculate_img_tokens(
                                        data=image_url_str,
                                        mode="auto",
                                        use_default_image_token_count=use_default_image_token_count
                                        or False,
                                    )
                if message.get("tool_calls"):
                    is_tool_call = True
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            function_arguments = tool_call["function"]["arguments"]
                            text += function_arguments
        else:
            raise ValueError("text and messages cannot both be None")
    elif isinstance(text, List):
        text = "".join(t for t in text if isinstance(t, str))
    elif isinstance(text, str):
        count_response_tokens = True  # user just trying to count tokens for a text. don't add the chat_ml +3 tokens to this

    if model is not None or custom_tokenizer is not None:
        tokenizer_json = custom_tokenizer or _select_tokenizer(model=model)
        if tokenizer_json["type"] == "huggingface_tokenizer":
            enc = tokenizer_json["tokenizer"].encode(text)
            num_tokens = len(enc.ids)
        elif tokenizer_json["type"] == "openai_tokenizer":
            if (
                model in litellm.open_ai_chat_completion_models
                or model in litellm.azure_llms
            ):
                if model in litellm.azure_llms:
                    # azure llms use gpt-35-turbo instead of gpt-3.5-turbo 🙃
                    model = model.replace("-35", "-3.5")

                print_verbose(
                    f"Token Counter - using OpenAI token counter, for model={model}"
                )
                num_tokens = openai_token_counter(
                    text=text,  # type: ignore
                    model=model,
                    messages=messages,
                    is_tool_call=is_tool_call,
                    count_response_tokens=count_response_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    use_default_image_token_count=use_default_image_token_count
                    or False,
                )
            else:
                print_verbose(
                    f"Token Counter - using generic token counter, for model={model}"
                )
                num_tokens = openai_token_counter(
                    text=text,  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=messages,
                    is_tool_call=is_tool_call,
                    count_response_tokens=count_response_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    use_default_image_token_count=use_default_image_token_count
                    or False,
                )
    else:
        num_tokens = len(encoding.encode(text, disallowed_special=()))  # type: ignore
    return num_tokens


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
    ]

    if custom_llm_provider in PROVIDERS_GLOBALLY_SUPPORT_RESPONSE_SCHEMA:
        return True
    return _supports_factory(
        model=model,
        custom_llm_provider=custom_llm_provider,
        key="supports_response_schema",
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
        return False
    except Exception as e:
        verbose_logger.debug(
            f"Model not found or error in checking {key} support. You passed model={model}, custom_llm_provider={custom_llm_provider}. Error: {str(e)}"
        )
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


def supports_vision(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """
    Check if the given model supports vision and return a boolean value.

    Parameters:
    model (str): The model name to be checked.
    custom_llm_provider (Optional[str]): The provider to be checked.

    Returns:
    bool: True if the model supports vision, False otherwise.
    """
    try:
        model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider
        )

        model_info = litellm.get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )

        if model_info.get("supports_vision", False) is True:
            return True
        return False
    except Exception as e:
        verbose_logger.error(
            f"Model not found or error in checking vision support. You passed model={model}, custom_llm_provider={custom_llm_provider}. Error: {str(e)}"
        )
        return False


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


def supports_parallel_function_calling(model: str):
    """
    Check if the given model supports parallel function calling and return True if it does, False otherwise.

    Parameters:
        model (str): The model to check for support of parallel function calling.

    Returns:
        bool: True if the model supports parallel function calling, False otherwise.

    Raises:
        Exception: If the model is not found in the model_cost dictionary.
    """
    if model in litellm.model_cost:
        model_info = litellm.model_cost[model]
        if model_info.get("supports_parallel_function_calling", False) is True:
            return True
        return False
    else:
        raise Exception(
            f"Model not supports parallel function calling. You passed model={model}."
        )


####### HELPER FUNCTIONS ################
def _update_dictionary(existing_dict: Dict, new_dict: dict) -> dict:
    for k, v in new_dict.items():
        existing_dict[k] = v

    return existing_dict


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
        # add new model names to provider lists
        if value.get("litellm_provider") == "openai":
            if key not in litellm.open_ai_chat_completion_models:
                litellm.open_ai_chat_completion_models.append(key)
        elif value.get("litellm_provider") == "text-completion-openai":
            if key not in litellm.open_ai_text_completion_models:
                litellm.open_ai_text_completion_models.append(key)
        elif value.get("litellm_provider") == "cohere":
            if key not in litellm.cohere_models:
                litellm.cohere_models.append(key)
        elif value.get("litellm_provider") == "anthropic":
            if key not in litellm.anthropic_models:
                litellm.anthropic_models.append(key)
        elif value.get("litellm_provider") == "openrouter":
            split_string = key.split("/", 1)
            if key not in litellm.openrouter_models:
                litellm.openrouter_models.append(split_string[1])
        elif value.get("litellm_provider") == "vertex_ai-text-models":
            if key not in litellm.vertex_text_models:
                litellm.vertex_text_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-code-text-models":
            if key not in litellm.vertex_code_text_models:
                litellm.vertex_code_text_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-chat-models":
            if key not in litellm.vertex_chat_models:
                litellm.vertex_chat_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-code-chat-models":
            if key not in litellm.vertex_code_chat_models:
                litellm.vertex_code_chat_models.append(key)
        elif value.get("litellm_provider") == "ai21":
            if key not in litellm.ai21_models:
                litellm.ai21_models.append(key)
        elif value.get("litellm_provider") == "nlp_cloud":
            if key not in litellm.nlp_cloud_models:
                litellm.nlp_cloud_models.append(key)
        elif value.get("litellm_provider") == "aleph_alpha":
            if key not in litellm.aleph_alpha_models:
                litellm.aleph_alpha_models.append(key)
        elif value.get("litellm_provider") == "bedrock":
            if key not in litellm.bedrock_models:
                litellm.bedrock_models.append(key)
    return model_cost


def get_litellm_params(
    api_key: Optional[str] = None,
    force_timeout=600,
    azure=False,
    logger_fn=None,
    verbose=False,
    hugging_face=False,
    replicate=False,
    together_ai=False,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    litellm_call_id=None,
    model_alias_map=None,
    completion_call_id=None,
    metadata: Optional[dict] = None,
    model_info=None,
    proxy_server_request=None,
    acompletion=None,
    preset_cache_key=None,
    no_log=None,
    input_cost_per_second=None,
    input_cost_per_token=None,
    output_cost_per_token=None,
    output_cost_per_second=None,
    cooldown_time=None,
    text_completion=None,
    azure_ad_token_provider=None,
    user_continue_message=None,
    base_model: Optional[str] = None,
    litellm_trace_id: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    custom_prompt_dict: Optional[dict] = None,
    litellm_metadata: Optional[dict] = None,
    disable_add_transform_inline_image_block: Optional[bool] = None,
    drop_params: Optional[bool] = None,
    prompt_id: Optional[str] = None,
    prompt_variables: Optional[dict] = None,
    async_call: Optional[bool] = None,
    **kwargs,
) -> dict:
    litellm_params = {
        "acompletion": acompletion,
        "api_key": api_key,
        "force_timeout": force_timeout,
        "logger_fn": logger_fn,
        "verbose": verbose,
        "custom_llm_provider": custom_llm_provider,
        "api_base": api_base,
        "litellm_call_id": litellm_call_id,
        "model_alias_map": model_alias_map,
        "completion_call_id": completion_call_id,
        "metadata": metadata,
        "model_info": model_info,
        "proxy_server_request": proxy_server_request,
        "preset_cache_key": preset_cache_key,
        "no-log": no_log,
        "stream_response": {},  # litellm_call_id: ModelResponse Dict
        "input_cost_per_token": input_cost_per_token,
        "input_cost_per_second": input_cost_per_second,
        "output_cost_per_token": output_cost_per_token,
        "output_cost_per_second": output_cost_per_second,
        "cooldown_time": cooldown_time,
        "text_completion": text_completion,
        "azure_ad_token_provider": azure_ad_token_provider,
        "user_continue_message": user_continue_message,
        "base_model": base_model
        or _get_base_model_from_litellm_call_metadata(metadata=metadata),
        "litellm_trace_id": litellm_trace_id,
        "hf_model_name": hf_model_name,
        "custom_prompt_dict": custom_prompt_dict,
        "litellm_metadata": litellm_metadata,
        "disable_add_transform_inline_image_block": disable_add_transform_inline_image_block,
        "drop_params": drop_params,
        "prompt_id": prompt_id,
        "prompt_variables": prompt_variables,
        "async_call": async_call,
    }
    return litellm_params


def _should_drop_param(k, additional_drop_params) -> bool:
    if (
        additional_drop_params is not None
        and isinstance(additional_drop_params, list)
        and k in additional_drop_params
    ):
        return True  # allow user to drop specific params for a model - e.g. vllm - logit bias

    return False


def _get_non_default_params(
    passed_params: dict, default_params: dict, additional_drop_params: Optional[bool]
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
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    temperature: Optional[int] = None,
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None,
    custom_llm_provider: Optional[str] = None,
    drop_params: Optional[bool] = None,
    **kwargs,
):
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
    for k in passed_params.keys():  # pass additional kwargs without modification
        if k not in default_params.keys():
            optional_params[k] = passed_params[k]
    return optional_params


def get_optional_params_image_gen(
    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[str] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    additional_drop_params: Optional[bool] = None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals()
    model = passed_params.pop("model", None)
    custom_llm_provider = passed_params.pop("custom_llm_provider")
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
    optional_params = {}

    ## raise exception if non-default value passed for non-openai/azure embedding calls
    def _check_valid_arg(supported_params):
        if len(non_default_params.keys()) > 0:
            keys = list(non_default_params.keys())
            for k in keys:
                if (
                    litellm.drop_params is True and k not in supported_params
                ):  # drop the unsupported non-default values
                    non_default_params.pop(k, None)
                elif k not in supported_params:
                    raise UnsupportedParamsError(
                        status_code=500,
                        message=f"Setting `{k}` is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
                    )
            return non_default_params

    if (
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
            else litellm.AmazonStabilityConfig
        )
        supported_params = config_class.get_supported_openai_params(model=model)
        _check_valid_arg(supported_params=supported_params)
        optional_params = config_class.map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
    elif custom_llm_provider == "vertex_ai":
        supported_params = ["n"]
        """
        All params here: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration?project=adroit-crow-413218
        """
        _check_valid_arg(supported_params=supported_params)
        if n is not None:
            optional_params["sampleCount"] = int(n)

    for k in passed_params.keys():
        if k not in default_params.keys():
            optional_params[k] = passed_params[k]
    return optional_params


def get_optional_params_embeddings(  # noqa: PLR0915
    # 2 optional params
    model: str,
    user: Optional[str] = None,
    encoding_format: Optional[str] = None,
    dimensions: Optional[int] = None,
    custom_llm_provider="",
    drop_params: Optional[bool] = None,
    additional_drop_params: Optional[bool] = None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals()
    custom_llm_provider = passed_params.pop("custom_llm_provider", None)
    special_params = passed_params.pop("kwargs")
    for k, v in special_params.items():
        passed_params[k] = v

    drop_params = passed_params.pop("drop_params", None)
    additional_drop_params = passed_params.pop("additional_drop_params", None)

    default_params = {"user": None, "encoding_format": None, "dimensions": None}

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

    non_default_params = _get_non_default_params(
        passed_params=passed_params,
        default_params=default_params,
        additional_drop_params=additional_drop_params,
    )
    ## raise exception if non-default value passed for non-openai/azure embedding calls
    if custom_llm_provider == "openai":
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
        final_params = {**optional_params, **kwargs}
        return final_params
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
        final_params = {**optional_params, **kwargs}
        return final_params
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
        return optional_params
    elif custom_llm_provider == "vertex_ai":
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
        final_params = {**optional_params, **kwargs}
        return final_params
    elif custom_llm_provider == "lm_studio":
        supported_params = (
            litellm.LmStudioEmbeddingConfig().get_supported_openai_params()
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.LmStudioEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
        final_params = {**optional_params, **kwargs}
        return final_params
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
        final_params = {**optional_params, **kwargs}
        return final_params
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
        final_params = {**optional_params, **kwargs}
        return final_params
    elif custom_llm_provider == "jina_ai":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="jina_ai",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.JinaAIEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
        final_params = {**optional_params, **kwargs}
        return final_params
    elif custom_llm_provider == "voyage":
        supported_params = get_supported_openai_params(
            model=model,
            custom_llm_provider="voyage",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.VoyageEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=drop_params if drop_params is not None else False,
        )
        final_params = {**optional_params, **kwargs}
        return final_params
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
        final_params = {**optional_params, **kwargs}
        return final_params

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
    final_params = {**non_default_params, **kwargs}
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
    additional_drop_params=None,
    messages: Optional[List[AllMessageValues]] = None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals().copy()
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

    default_params = {
        "functions": None,
        "function_call": None,
        "temperature": None,
        "top_p": None,
        "n": None,
        "stream": None,
        "stream_options": None,
        "stop": None,
        "max_tokens": None,
        "max_completion_tokens": None,
        "modalities": None,
        "prediction": None,
        "audio": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "logit_bias": None,
        "user": None,
        "model": None,
        "custom_llm_provider": "",
        "response_format": None,
        "seed": None,
        "tools": None,
        "tool_choice": None,
        "max_retries": None,
        "logprobs": None,
        "top_logprobs": None,
        "extra_headers": None,
        "api_version": None,
        "parallel_tool_calls": None,
        "drop_params": None,
        "additional_drop_params": None,
        "messages": None,
    }

    # filter out those parameters that were passed with non-default values
    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (
            k != "model"
            and k != "custom_llm_provider"
            and k != "api_version"
            and k != "drop_params"
            and k != "additional_drop_params"
            and k != "messages"
            and k in default_params
            and v != default_params[k]
            and _should_drop_param(k=k, additional_drop_params=additional_drop_params)
            is False
        )
    }

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

    if "response_format" in non_default_params:
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

    def _check_valid_arg(supported_params: List[str]):
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
                    message=f"{custom_llm_provider} does not support parameters: {unsupported_params}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n",
                )

    provider_config: Optional[BaseConfig] = None
    if custom_llm_provider is not None and custom_llm_provider in [
        provider.value for provider in LlmProviders
    ]:
        provider_config = ProviderConfigManager.get_provider_chat_config(
            model=model, provider=LlmProviders(custom_llm_provider)
        )
    supported_params = get_supported_openai_params(
        model=model, custom_llm_provider=custom_llm_provider
    )
    if supported_params is None:
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider="openai"
        )
    _check_valid_arg(supported_params=supported_params or [])
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
        optional_params = litellm.HuggingfaceConfig().map_openai_params(
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
    elif custom_llm_provider == "vertex_ai" and model in litellm.vertex_llama3_models:
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
    elif custom_llm_provider == "vertex_ai" and model in litellm.vertex_mistral_models:
        if "codestral" in model:
            optional_params = litellm.CodestralTextCompletionConfig().map_openai_params(
                model=model,
                non_default_params=non_default_params,
                optional_params=optional_params,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
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
    elif custom_llm_provider == "vertex_ai" and model in litellm.vertex_ai_ai21_models:
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
        base_model = litellm.AmazonConverseConfig()._get_base_model(model)
        if base_model in litellm.bedrock_converse_models:
            optional_params = litellm.AmazonConverseConfig().map_openai_params(
                model=model,
                non_default_params=non_default_params,
                optional_params=optional_params,
                drop_params=(
                    drop_params
                    if drop_params is not None and isinstance(drop_params, bool)
                    else False
                ),
                messages=messages,
            )

        elif "anthropic" in model:
            if "aws_bedrock_client" in passed_params:  # deprecated boto3.invoke route.
                if model.startswith("anthropic.claude-3"):
                    optional_params = (
                        litellm.AmazonAnthropicClaude3Config().map_openai_params(
                            non_default_params=non_default_params,
                            optional_params=optional_params,
                        )
                    )
            else:
                optional_params = litellm.AmazonAnthropicConfig().map_openai_params(
                    non_default_params=non_default_params,
                    optional_params=optional_params,
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
    elif custom_llm_provider == "azure":
        if litellm.AzureOpenAIO1Config().is_o1_model(model=model):
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
                if k not in default_params.keys():
                    extra_body[k] = passed_params[k]
            optional_params.setdefault("extra_body", {})
            optional_params["extra_body"] = {
                **optional_params["extra_body"],
                **extra_body,
            }

            optional_params["extra_body"] = _ensure_extra_body_is_safe(
                extra_body=optional_params["extra_body"]
            )
    else:
        # if user passed in non-default kwargs for specific providers/models, pass them along
        for k in passed_params.keys():
            if k not in default_params.keys():
                optional_params[k] = passed_params[k]
    print_verbose(f"Final returned optional params: {optional_params}")
    return optional_params


def get_non_default_params(passed_params: dict) -> dict:
    default_params = {
        "functions": None,
        "function_call": None,
        "temperature": None,
        "top_p": None,
        "n": None,
        "stream": None,
        "stream_options": None,
        "stop": None,
        "max_tokens": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "logit_bias": None,
        "user": None,
        "model": None,
        "custom_llm_provider": "",
        "response_format": None,
        "seed": None,
        "tools": None,
        "tool_choice": None,
        "max_retries": None,
        "logprobs": None,
        "top_logprobs": None,
        "extra_headers": None,
    }
    # filter out those parameters that were passed with non-default values
    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (
            k != "model"
            and k != "custom_llm_provider"
            and k in default_params
            and v != default_params[k]
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


def get_response_string(response_obj: ModelResponse) -> str:
    _choices: List[Union[Choices, StreamingChoices]] = response_obj.choices

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


def _strip_bedrock_region(model_name) -> str:
    return litellm.AmazonConverseConfig()._get_base_model(model_name)


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
    if custom_llm_provider and custom_llm_provider == "bedrock":
        strip_bedrock_region = _strip_bedrock_region(model_name=model)
        return strip_bedrock_region
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
        elif custom_llm_provider == "bedrock" and model_info[
            "litellm_provider"
        ].startswith("bedrock"):
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
        split_model = model.split("/")[1]
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


@lru_cache_wrapper(maxsize=16)
def _cached_get_model_info_helper(
    model: str, custom_llm_provider: Optional[str]
) -> ModelInfoBase:
    """
    _get_model_info_helper wrapped with lru_cache

    Speed Optimization to hit high RPS
    """
    return _get_model_info_helper(model=model, custom_llm_provider=custom_llm_provider)


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
                supports_assistant_prefill=None,
                supports_prompt_caching=None,
                supports_pdf_input=None,
            )
        elif custom_llm_provider == "ollama" or custom_llm_provider == "ollama_chat":
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
            provider_config: Optional[BaseLLMModelInfo] = None

            if combined_model_name in litellm.model_cost:
                key = combined_model_name
                _model_info = _get_model_info_from_model_cost(key=key)
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if _model_info is None and model in litellm.model_cost:
                key = model
                _model_info = _get_model_info_from_model_cost(key=key)
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if (
                _model_info is None
                and combined_stripped_model_name in litellm.model_cost
            ):
                key = combined_stripped_model_name
                _model_info = _get_model_info_from_model_cost(key=key)
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if _model_info is None and stripped_model_name in litellm.model_cost:
                key = stripped_model_name
                _model_info = _get_model_info_from_model_cost(key=key)
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None
            if _model_info is None and split_model in litellm.model_cost:
                key = split_model
                _model_info = _get_model_info_from_model_cost(key=key)
                if not _check_provider_match(
                    model_info=_model_info, custom_llm_provider=custom_llm_provider
                ):
                    _model_info = None

            if custom_llm_provider and custom_llm_provider in LlmProvidersSet:
                # Check if the provider string exists in LlmProviders enum
                provider_config = ProviderConfigManager.get_provider_model_info(
                    model=model, provider=LlmProviders(custom_llm_provider)
                )

            if _model_info is None and provider_config is not None:
                _model_info = cast(
                    Optional[Dict],
                    provider_config.get_model_info(
                        model=model, existing_model_info=_model_info
                    ),
                )
                key = "provider_specific_model_info"
            if _model_info is None or key is None:
                raise ValueError(
                    "This model isn't mapped yet. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                )

            ## PROVIDER-SPECIFIC INFORMATION
            if custom_llm_provider == "predibase":
                _model_info["supports_response_schema"] = True

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
                input_cost_per_query=_model_info.get("input_cost_per_query", None),
                input_cost_per_second=_model_info.get("input_cost_per_second", None),
                input_cost_per_audio_token=_model_info.get(
                    "input_cost_per_audio_token", None
                ),
                output_cost_per_token=_output_cost_per_token,
                output_cost_per_audio_token=_model_info.get(
                    "output_cost_per_audio_token", None
                ),
                output_cost_per_character=_model_info.get(
                    "output_cost_per_character", None
                ),
                output_cost_per_token_above_128k_tokens=_model_info.get(
                    "output_cost_per_token_above_128k_tokens", None
                ),
                output_cost_per_character_above_128k_tokens=_model_info.get(
                    "output_cost_per_character_above_128k_tokens", None
                ),
                output_cost_per_second=_model_info.get("output_cost_per_second", None),
                output_cost_per_image=_model_info.get("output_cost_per_image", None),
                output_vector_size=_model_info.get("output_vector_size", None),
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
                supports_vision=_model_info.get("supports_vision", False),
                supports_function_calling=_model_info.get(
                    "supports_function_calling", False
                ),
                supports_assistant_prefill=_model_info.get(
                    "supports_assistant_prefill", False
                ),
                supports_prompt_caching=_model_info.get(
                    "supports_prompt_caching", False
                ),
                supports_audio_input=_model_info.get("supports_audio_input", False),
                supports_audio_output=_model_info.get("supports_audio_output", False),
                supports_pdf_input=_model_info.get("supports_pdf_input", False),
                supports_embedding_image_input=_model_info.get(
                    "supports_embedding_image_input", False
                ),
                supports_native_streaming=_model_info.get(
                    "supports_native_streaming", None
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
            supports_prompt_caching: Optional[bool]
            supports_audio_input: Optional[bool]
            supports_audio_output: Optional[bool]
            supports_pdf_input: Optional[bool]
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
        elif custom_llm_provider == "gemini":
            if "GEMINI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
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

    if api_key is not None:
        new_missing_keys = []
        for key in missing_keys:
            if "api_key" not in key.lower():
                new_missing_keys.append(key)
        missing_keys = new_missing_keys

    if api_base is not None:
        new_missing_keys = []
        for key in missing_keys:
            if "api_base" not in key.lower():
                new_missing_keys.append(key)
        missing_keys = new_missing_keys

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
        num_tokens = anthropic_obj.count_tokens(text)
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


def type_to_response_format_param(
    response_format: Optional[Union[Type[BaseModel], dict]],
) -> Optional[dict]:
    """
    Re-implementation of openai's 'type_to_response_format_param' function

    Used for converting pydantic object to api schema.
    """
    if response_format is None:
        return None

    if isinstance(response_format, dict):
        return response_format

    # type checkers don't narrow the negation of a `TypeGuard` as it isn't
    # a safe default behaviour but we know that at this point the `response_format`
    # can only be a `type`
    if not _parsing._completions.is_basemodel_type(response_format):
        raise TypeError(f"Unsupported response_format type - {response_format}")

    return {
        "type": "json_schema",
        "json_schema": {
            "schema": _pydantic.to_strict_json_schema(response_format),
            "name": response_format.__name__,
            "strict": True,
        },
    }


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

    # If the API asks us to wait a certain amount of time (and it's a reasonable amount), just do what it says.
    if retry_after is not None and 0 < retry_after <= 60:
        return retry_after

    initial_retry_delay = 0.5
    max_retry_delay = 8.0
    nb_retries = max_retries - remaining_retries

    # Apply exponential backoff, but not more than the max.
    sleep_seconds = min(initial_retry_delay * pow(2.0, nb_retries), max_retry_delay)

    # Apply some jitter, plus-or-minus half a second.
    jitter = 1 - 0.25 * random.random()
    timeout = sleep_seconds * jitter
    return timeout if timeout >= min_timeout else min_timeout


# custom prompt helper function
def register_prompt_template(
    model: str,
    roles: dict,
    initial_prompt_value: str = "",
    final_prompt_value: str = "",
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
    model = get_llm_provider(model=model)[0]
    litellm.custom_prompt_dict[model] = {
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
            text_choices["text"] = chunk["choices"][0]["delta"]["content"]
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

    for message in messages:
        used_tokens = get_token_count(final_messages, model)
        available_tokens = max_tokens - used_tokens
        if available_tokens <= 3:
            break
        final_messages = attempt_message_addition(
            final_messages=final_messages,
            message=message,
            available_tokens=available_tokens,
            max_tokens=max_tokens,
            model=model,
        )

    return final_messages


def attempt_message_addition(
    final_messages, message, available_tokens, max_tokens, model
):
    temp_messages = [message] + final_messages
    temp_message_tokens = get_token_count(messages=temp_messages, model=model)

    if temp_message_tokens <= max_tokens:
        return temp_messages

    # if temp_message_tokens > max_tokens, try shortening temp_messages
    elif "function_call" not in message:
        # fit updated_message to be within temp_message_tokens - max_tokens (aka the amount temp_message_tokens is greate than max_tokens)
        updated_message = shorten_message_to_fit_limit(message, available_tokens, model)
        if can_add_message(updated_message, final_messages, max_tokens, model):
            return [updated_message] + final_messages

    return final_messages


def can_add_message(message, messages, max_tokens, model):
    if get_token_count(messages + [message], model) <= max_tokens:
        return True
    return False


def get_token_count(messages, model):
    return token_counter(model=model, messages=messages)


def shorten_message_to_fit_limit(message, tokens_needed, model: Optional[str]):
    """
    Shorten a message to fit within a token limit by removing characters from the middle.
    """

    # For OpenAI models, even blank messages cost 7 token,
    # and if the buffer is less than 3, the while loop will never end,
    # hence the value 10.
    if model is not None and "gpt" in model and tokens_needed <= 10:
        return message

    content = message["content"]

    while True:
        total_tokens = get_token_count([message], model)

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
        content = trimmed_content

    return message


# LiteLLM token trimmer
# this code is borrowed from https://github.com/KillianLucas/tokentrim/blob/main/tokentrim/tokentrim.py
# Credits for this code go to Killian Lucas
def trim_messages(
    messages,
    model: Optional[str] = None,
    trim_ratio: float = 0.75,
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
        # # Remove the collected tool messages from the original list
        if len(tool_messages):
            messages = messages[: -len(tool_messages)]

        current_tokens = token_counter(model=model or "", messages=messages)
        print_verbose(f"Current tokens: {current_tokens}, max tokens: {max_tokens}")

        # Do nothing if current tokens under messages
        if current_tokens < max_tokens:
            return messages

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

        final_messages = process_messages(
            messages=messages, max_tokens=max_tokens, model=model
        )

        # Add system message to the beginning of the final messages
        if system_message_event:
            final_messages = [system_message_event] + final_messages

        if len(tool_messages) > 0:
            final_messages.extend(tool_messages)

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
        return messages


def get_valid_models(check_provider_endpoint: bool = False) -> List[str]:
    """
    Returns a list of valid LLMs based on the set environment variables

    Args:
        check_provider_endpoint: If True, will check the provider's endpoint for valid models.

    Returns:
        A list of valid LLMs
    """
    try:
        # get keys set in .env
        environ_keys = os.environ.keys()
        valid_providers = []
        # for all valid providers, make a list of supported llms
        valid_models = []

        for provider in litellm.provider_list:
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

        for provider in valid_providers:
            provider_config = ProviderConfigManager.get_provider_model_info(
                model=None,
                provider=LlmProviders(provider),
            )

            if provider == "azure":
                valid_models.append("Azure-LLM")
            elif provider_config is not None and check_provider_endpoint:
                valid_models.extend(provider_config.get_models())
            else:
                models_for_provider = litellm.models_by_provider.get(provider, [])
                valid_models.extend(models_for_provider)
        return valid_models
    except Exception as e:
        verbose_logger.debug(f"Error getting valid models: {e}")
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


def _get_base_model_from_litellm_call_metadata(
    metadata: Optional[dict],
) -> Optional[str]:
    if metadata is None:
        return None

    if metadata is not None:
        model_info = metadata.get("model_info", {})

        if model_info is not None:
            base_model = model_info.get("base_model", None)
            if base_model is not None:
                return base_model
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
    def __init__(self, model_responses):
        self.model_responses = model_responses
        self.index = 0

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.model_responses):
            raise StopIteration
        model_response = self.model_responses[self.index]
        self.index += 1
        return model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.model_responses):
            raise StopAsyncIteration
        model_response = self.model_responses[self.index]
        self.index += 1
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


def validate_chat_completion_messages(messages: List[AllMessageValues]):
    """
    Ensures all messages are valid OpenAI chat completion messages.
    """
    # 1. convert all messages to dict
    messages = [
        cast(AllMessageValues, convert_to_dict(cast(dict, m))) for m in messages
    ]
    # 2. validate user messages
    return validate_chat_completion_user_messages(messages=messages)


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
            if "invalid content type" in str(e):
                raise Exception(
                    f"Invalid user message={m} at index {idx}. Please ensure all user messages are valid OpenAI chat completion messages."
                )
            else:
                raise e

    return messages


def validate_chat_completion_tool_choice(
    tool_choice: Optional[Union[dict, str]]
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
    ) -> BaseConfig:
        """
        Returns the provider config for a given provider.
        """
        if (
            provider == LlmProviders.OPENAI
            and litellm.openAIO1Config.is_model_o1_reasoning_model(model=model)
        ):
            return litellm.OpenAIO1Config()
        elif litellm.LlmProviders.DEEPSEEK == provider:
            return litellm.DeepSeekChatConfig()
        elif litellm.LlmProviders.GROQ == provider:
            return litellm.GroqChatConfig()
        elif litellm.LlmProviders.DATABRICKS == provider:
            return litellm.DatabricksConfig()
        elif litellm.LlmProviders.XAI == provider:
            return litellm.XAIChatConfig()
        elif litellm.LlmProviders.TEXT_COMPLETION_OPENAI == provider:
            return litellm.OpenAITextCompletionConfig()
        elif litellm.LlmProviders.COHERE_CHAT == provider:
            return litellm.CohereChatConfig()
        elif litellm.LlmProviders.COHERE == provider:
            return litellm.CohereConfig()
        elif litellm.LlmProviders.CLARIFAI == provider:
            return litellm.ClarifaiConfig()
        elif litellm.LlmProviders.ANTHROPIC == provider:
            return litellm.AnthropicConfig()
        elif litellm.LlmProviders.ANTHROPIC_TEXT == provider:
            return litellm.AnthropicTextConfig()
        elif litellm.LlmProviders.VERTEX_AI == provider:
            if "claude" in model:
                return litellm.VertexAIAnthropicConfig()
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
        elif (
            litellm.LlmProviders.CUSTOM == provider
            or litellm.LlmProviders.CUSTOM_OPENAI == provider
            or litellm.LlmProviders.OPENAI_LIKE == provider
            or litellm.LlmProviders.LITELLM_PROXY == provider
        ):
            return litellm.OpenAILikeChatConfig()
        elif litellm.LlmProviders.AIOHTTP_OPENAI == provider:
            return litellm.AiohttpOpenAIChatConfig()
        elif litellm.LlmProviders.HOSTED_VLLM == provider:
            return litellm.HostedVLLMChatConfig()
        elif litellm.LlmProviders.LM_STUDIO == provider:
            return litellm.LMStudioChatConfig()
        elif litellm.LlmProviders.GALADRIEL == provider:
            return litellm.GaladrielChatConfig()
        elif litellm.LlmProviders.REPLICATE == provider:
            return litellm.ReplicateConfig()
        elif litellm.LlmProviders.HUGGINGFACE == provider:
            return litellm.HuggingfaceConfig()
        elif litellm.LlmProviders.TOGETHER_AI == provider:
            return litellm.TogetherAIConfig()
        elif litellm.LlmProviders.OPENROUTER == provider:
            return litellm.OpenrouterConfig()
        elif litellm.LlmProviders.GEMINI == provider:
            return litellm.GoogleAIStudioGeminiConfig()
        elif (
            litellm.LlmProviders.AI21 == provider
            or litellm.LlmProviders.AI21_CHAT == provider
        ):
            return litellm.AI21ChatConfig()
        elif litellm.LlmProviders.AZURE == provider:
            if litellm.AzureOpenAIO1Config().is_o1_model(model=model):
                return litellm.AzureOpenAIO1Config()
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
        elif litellm.LlmProviders.BEDROCK == provider:
            base_model = litellm.AmazonConverseConfig()._get_base_model(model)
            if base_model in litellm.bedrock_converse_models:
                pass
            elif "amazon" in model:  # amazon titan llms
                return litellm.AmazonTitanConfig()
            elif "meta" in model:  # amazon / meta llms
                return litellm.AmazonLlamaConfig()
            elif "ai21" in model:  # ai21 llms
                return litellm.AmazonAI21Config()
            elif "cohere" in model:  # cohere models on bedrock
                return litellm.AmazonCohereConfig()
            elif "mistral" in model:  # mistral models on bedrock
                return litellm.AmazonMistralConfig()
        return litellm.OpenAIGPTConfig()

    @staticmethod
    def get_provider_embedding_config(
        model: str,
        provider: LlmProviders,
    ) -> BaseEmbeddingConfig:
        if litellm.LlmProviders.VOYAGE == provider:
            return litellm.VoyageEmbeddingConfig()
        elif litellm.LlmProviders.TRITON == provider:
            return litellm.TritonEmbeddingConfig()
        elif litellm.LlmProviders.WATSONX == provider:
            return litellm.IBMWatsonXEmbeddingConfig()
        raise ValueError(f"Provider {provider.value} does not support embedding config")

    @staticmethod
    def get_provider_rerank_config(
        model: str,
        provider: LlmProviders,
    ) -> BaseRerankConfig:
        if litellm.LlmProviders.COHERE == provider:
            return litellm.CohereRerankConfig()
        elif litellm.LlmProviders.AZURE_AI == provider:
            return litellm.AzureAIRerankConfig()
        elif litellm.LlmProviders.INFINITY == provider:
            return litellm.InfinityRerankConfig()
        return litellm.CohereRerankConfig()

    @staticmethod
    def get_provider_audio_transcription_config(
        model: str,
        provider: LlmProviders,
    ) -> Optional[BaseAudioTranscriptionConfig]:
        if litellm.LlmProviders.FIREWORKS_AI == provider:
            return litellm.FireworksAIAudioTranscriptionConfig()
        elif litellm.LlmProviders.DEEPGRAM == provider:
            return litellm.DeepgramAudioTranscriptionConfig()
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
        elif LlmProviders.LITELLM_PROXY == provider:
            return litellm.LiteLLMProxyChatConfig()
        elif LlmProviders.TOPAZ == provider:
            return litellm.TopazModelInfo()

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
    if (
        service_type == "prometheus"
        and litellm.disable_end_user_cost_tracking_prometheus_only
    ):
        return None
    return end_user_id


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
        return token_count >= 1024
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


def get_non_default_completion_params(kwargs: dict) -> dict:
    openai_params = litellm.OPENAI_CHAT_COMPLETION_PARAMS
    default_params = openai_params + all_litellm_params
    non_default_params = {
        k: v for k, v in kwargs.items() if k not in default_params
    }  # model-specific params - pass them straight to the model/provider
    return non_default_params
