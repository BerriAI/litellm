# What is this?
## Common Utility file for Logging handler
# Logging function -> log the exact model details + what's being sent | Non-Blocking
import copy
import datetime
import json
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime as dt_object
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

from pydantic import BaseModel

import litellm
from litellm import (
    _custom_logger_compatible_callbacks_literal,
    json_logs,
    log_raw_request_response,
    turn_off_message_logging,
)
from litellm._logging import _is_debugging_on, verbose_logger
from litellm.batches.batch_utils import _handle_completed_batch
from litellm.caching.caching import DualCache, InMemoryCache
from litellm.caching.caching_handler import LLMCachingHandler
from litellm.constants import (
    DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT,
    DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT,
)
from litellm.cost_calculator import (
    RealtimeAPITokenUsageProcessor,
    _select_model_name_for_cost_calc,
)
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.arize.arize import ArizeLogger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.mlflow import MlflowLogger
from litellm.integrations.pagerduty.pagerduty import PagerDutyAlerting
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.litellm_core_utils.model_param_helper import ModelParamHelper
from litellm.litellm_core_utils.redact_messages import (
    redact_message_input_output_from_custom_logger,
    redact_message_input_output_from_logging,
)
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.openai import (
    AllMessageValues,
    Batch,
    FineTuningJob,
    HttpxBinaryResponseContent,
    OpenAIFileObject,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
)
from litellm.types.rerank import RerankResponse
from litellm.types.router import SPECIAL_MODEL_INFO_PARAMS
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ImageResponse,
    LiteLLMBatch,
    LiteLLMLoggingBaseClass,
    LiteLLMRealtimeStreamLoggingObject,
    ModelResponse,
    ModelResponseStream,
    RawRequestTypedDict,
    StandardBuiltInToolsParams,
    StandardCallbackDynamicParams,
    StandardLoggingAdditionalHeaders,
    StandardLoggingHiddenParams,
    StandardLoggingMCPToolCall,
    StandardLoggingMetadata,
    StandardLoggingModelCostFailureDebugInformation,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
    StandardLoggingPayloadStatus,
    StandardLoggingPromptManagementMetadata,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
)
from litellm.utils import _get_base_model_from_metadata, executor, print_verbose

from ..integrations.argilla import ArgillaLogger
from ..integrations.arize.arize_phoenix import ArizePhoenixLogger
from ..integrations.athina import AthinaLogger
from ..integrations.azure_storage.azure_storage import AzureBlobStorageLogger
from ..integrations.braintrust_logging import BraintrustLogger
from ..integrations.custom_prompt_management import CustomPromptManagement
from ..integrations.datadog.datadog import DataDogLogger
from ..integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from ..integrations.dynamodb import DyanmoDBLogger
from ..integrations.galileo import GalileoObserve
from ..integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from ..integrations.gcs_pubsub.pub_sub import GcsPubSubLogger
from ..integrations.greenscale import GreenscaleLogger
from ..integrations.helicone import HeliconeLogger
from ..integrations.humanloop import HumanloopLogger
from ..integrations.lago import LagoLogger
from ..integrations.langfuse.langfuse import LangFuseLogger
from ..integrations.langfuse.langfuse_handler import LangFuseHandler
from ..integrations.langfuse.langfuse_prompt_management import LangfusePromptManagement
from ..integrations.langsmith import LangsmithLogger
from ..integrations.literal_ai import LiteralAILogger
from ..integrations.logfire_logger import LogfireLevel, LogfireLogger
from ..integrations.lunary import LunaryLogger
from ..integrations.openmeter import OpenMeterLogger
from ..integrations.opik.opik import OpikLogger
from ..integrations.prometheus import PrometheusLogger
from ..integrations.prompt_layer import PromptLayerLogger
from ..integrations.s3 import S3Logger
from ..integrations.supabase import Supabase
from ..integrations.traceloop import TraceloopLogger
from ..integrations.weights_biases import WeightsBiasesLogger
from .exception_mapping_utils import _get_response_headers
from .initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params as _initialize_standard_callback_dynamic_params,
)
from .specialty_caches.dynamic_logging_cache import DynamicLoggingCache

try:
    from ..proxy.enterprise.enterprise_callbacks.generic_api_callback import (
        GenericAPILogger,
    )
except Exception as e:
    verbose_logger.debug(
        f"[Non-Blocking] Unable to import GenericAPILogger - LiteLLM Enterprise Feature - {str(e)}"
    )

_in_memory_loggers: List[Any] = []

### GLOBAL VARIABLES ###

sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
posthog = None
slack_app = None
alerts_channel = None
heliconeLogger = None
athinaLogger = None
promptLayerLogger = None
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
supabaseClient = None
callback_list: Optional[List[str]] = []
user_logger_fn = None
additional_details: Optional[Dict[str, str]] = {}
local_cache: Optional[Dict[str, str]] = {}
last_fetched_at = None
last_fetched_at_keys = None


####
class ServiceTraceIDCache:
    def __init__(self) -> None:
        self.cache = InMemoryCache()

    def get_cache(self, litellm_call_id: str, service_name: str) -> Optional[str]:
        key_name = "{}:{}".format(service_name, litellm_call_id)
        response = self.cache.get_cache(key=key_name)
        return response

    def set_cache(self, litellm_call_id: str, service_name: str, trace_id: str) -> None:
        key_name = "{}:{}".format(service_name, litellm_call_id)
        self.cache.set_cache(key=key_name, value=trace_id)
        return None


in_memory_trace_id_cache = ServiceTraceIDCache()
in_memory_dynamic_logger_cache = DynamicLoggingCache()


class Logging(LiteLLMLoggingBaseClass):
    global supabaseClient, promptLayerLogger, weightsBiasesLogger, logfireLogger, capture_exception, add_breadcrumb, lunaryLogger, logfireLogger, prometheusLogger, slack_app
    custom_pricing: bool = False
    stream_options = None

    def __init__(
        self,
        model: str,
        messages,
        stream,
        call_type,
        start_time,
        litellm_call_id: str,
        function_id: str,
        litellm_trace_id: Optional[str] = None,
        dynamic_input_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None,
        dynamic_success_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None,
        dynamic_async_success_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None,
        dynamic_failure_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None,
        dynamic_async_failure_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = None,
        applied_guardrails: Optional[List[str]] = None,
        kwargs: Optional[Dict] = None,
        log_raw_request_response: bool = False,
    ):
        _input: Optional[str] = messages  # save original value of messages
        if messages is not None:
            if isinstance(messages, str):
                messages = [
                    {"role": "user", "content": messages}
                ]  # convert text completion input to the chat completion format
            elif (
                isinstance(messages, list)
                and len(messages) > 0
                and isinstance(messages[0], str)
            ):
                new_messages = []
                for m in messages:
                    new_messages.append({"role": "user", "content": m})
                messages = new_messages
        self.model = model
        self.messages = copy.deepcopy(messages)
        self.stream = stream
        self.start_time = start_time  # log the call start time
        self.call_type = call_type
        self.litellm_call_id = litellm_call_id
        self.litellm_trace_id = litellm_trace_id
        self.function_id = function_id
        self.streaming_chunks: List[Any] = []  # for generating complete stream response
        self.sync_streaming_chunks: List[Any] = (
            []
        )  # for generating complete stream response
        self.log_raw_request_response = log_raw_request_response

        # Initialize dynamic callbacks
        self.dynamic_input_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = dynamic_input_callbacks
        self.dynamic_success_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = dynamic_success_callbacks
        self.dynamic_async_success_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = dynamic_async_success_callbacks
        self.dynamic_failure_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = dynamic_failure_callbacks
        self.dynamic_async_failure_callbacks: Optional[
            List[Union[str, Callable, CustomLogger]]
        ] = dynamic_async_failure_callbacks

        # Process dynamic callbacks
        self.process_dynamic_callbacks()

        ## DYNAMIC LANGFUSE / GCS / logging callback KEYS ##
        self.standard_callback_dynamic_params: StandardCallbackDynamicParams = (
            self.initialize_standard_callback_dynamic_params(kwargs)
        )
        self.standard_built_in_tools_params: StandardBuiltInToolsParams = (
            self.initialize_standard_built_in_tools_params(kwargs)
        )
        ## TIME TO FIRST TOKEN LOGGING ##
        self.completion_start_time: Optional[datetime.datetime] = None
        self._llm_caching_handler: Optional[LLMCachingHandler] = None

        # INITIAL LITELLM_PARAMS
        litellm_params = {}
        if kwargs is not None:
            litellm_params = get_litellm_params(**kwargs)
            litellm_params = scrub_sensitive_keys_in_metadata(litellm_params)

        self.litellm_params = litellm_params

        self.model_call_details: Dict[str, Any] = {
            "litellm_trace_id": litellm_trace_id,
            "litellm_call_id": litellm_call_id,
            "input": _input,
            "litellm_params": litellm_params,
            "applied_guardrails": applied_guardrails,
            "model": model,
        }

    def process_dynamic_callbacks(self):
        """
        Initializes CustomLogger compatible callbacks in self.dynamic_* callbacks

        If a callback is in litellm._known_custom_logger_compatible_callbacks, it needs to be intialized and added to the respective dynamic_* callback list.
        """
        # Process input callbacks
        self.dynamic_input_callbacks = self._process_dynamic_callback_list(
            self.dynamic_input_callbacks, dynamic_callbacks_type="input"
        )

        # Process failure callbacks
        self.dynamic_failure_callbacks = self._process_dynamic_callback_list(
            self.dynamic_failure_callbacks, dynamic_callbacks_type="failure"
        )

        # Process async failure callbacks
        self.dynamic_async_failure_callbacks = self._process_dynamic_callback_list(
            self.dynamic_async_failure_callbacks, dynamic_callbacks_type="async_failure"
        )

        # Process success callbacks
        self.dynamic_success_callbacks = self._process_dynamic_callback_list(
            self.dynamic_success_callbacks, dynamic_callbacks_type="success"
        )

        # Process async success callbacks
        self.dynamic_async_success_callbacks = self._process_dynamic_callback_list(
            self.dynamic_async_success_callbacks, dynamic_callbacks_type="async_success"
        )

    def _process_dynamic_callback_list(
        self,
        callback_list: Optional[List[Union[str, Callable, CustomLogger]]],
        dynamic_callbacks_type: Literal[
            "input", "success", "failure", "async_success", "async_failure"
        ],
    ) -> Optional[List[Union[str, Callable, CustomLogger]]]:
        """
        Helper function to initialize CustomLogger compatible callbacks in self.dynamic_* callbacks

        - If a callback is in litellm._known_custom_logger_compatible_callbacks,
        replace the string with the initialized callback class.
        - If dynamic callback is a "success" callback that is a known_custom_logger_compatible_callbacks then add it to dynamic_async_success_callbacks
        - If dynamic callback is a "failure" callback that is a known_custom_logger_compatible_callbacks then add it to dynamic_failure_callbacks
        """
        if callback_list is None:
            return None

        processed_list: List[Union[str, Callable, CustomLogger]] = []
        for callback in callback_list:
            if (
                isinstance(callback, str)
                and callback in litellm._known_custom_logger_compatible_callbacks
            ):
                callback_class = _init_custom_logger_compatible_class(
                    callback, internal_usage_cache=None, llm_router=None  # type: ignore
                )
                if callback_class is not None:
                    processed_list.append(callback_class)

                    # If processing dynamic_success_callbacks, add to dynamic_async_success_callbacks
                    if dynamic_callbacks_type == "success":
                        if self.dynamic_async_success_callbacks is None:
                            self.dynamic_async_success_callbacks = []
                        self.dynamic_async_success_callbacks.append(callback_class)
                    elif dynamic_callbacks_type == "failure":
                        if self.dynamic_async_failure_callbacks is None:
                            self.dynamic_async_failure_callbacks = []
                        self.dynamic_async_failure_callbacks.append(callback_class)
            else:
                processed_list.append(callback)
        return processed_list

    def initialize_standard_callback_dynamic_params(
        self, kwargs: Optional[Dict] = None
    ) -> StandardCallbackDynamicParams:
        """
        Initialize the standard callback dynamic params from the kwargs

        checks if langfuse_secret_key, gcs_bucket_name in kwargs and sets the corresponding attributes in StandardCallbackDynamicParams
        """
        return _initialize_standard_callback_dynamic_params(kwargs)

    def initialize_standard_built_in_tools_params(
        self, kwargs: Optional[Dict] = None
    ) -> StandardBuiltInToolsParams:
        """
        Initialize the standard built-in tools params from the kwargs

        checks if web_search_options in kwargs or tools and sets the corresponding attribute in StandardBuiltInToolsParams
        """
        return StandardBuiltInToolsParams(
            web_search_options=StandardBuiltInToolCostTracking._get_web_search_options(
                kwargs or {}
            ),
            file_search=StandardBuiltInToolCostTracking._get_file_search_tool_call(
                kwargs or {}
            ),
        )

    def update_environment_variables(
        self,
        litellm_params: Dict,
        optional_params: Dict,
        model: Optional[str] = None,
        user: Optional[str] = None,
        **additional_params,
    ):
        self.optional_params = optional_params
        if model is not None:
            self.model = model
        self.user = user
        self.litellm_params = {
            **self.litellm_params,
            **scrub_sensitive_keys_in_metadata(litellm_params),
        }
        self.logger_fn = litellm_params.get("logger_fn", None)
        verbose_logger.debug(f"self.optional_params: {self.optional_params}")

        self.model_call_details.update(
            {
                "model": self.model,
                "messages": self.messages,
                "optional_params": self.optional_params,
                "litellm_params": self.litellm_params,
                "start_time": self.start_time,
                "stream": self.stream,
                "user": user,
                "call_type": str(self.call_type),
                "litellm_call_id": self.litellm_call_id,
                "completion_start_time": self.completion_start_time,
                "standard_callback_dynamic_params": self.standard_callback_dynamic_params,
                **self.optional_params,
                **additional_params,
            }
        )

        ## check if stream options is set ##  - used by CustomStreamWrapper for easy instrumentation
        if "stream_options" in additional_params:
            self.stream_options = additional_params["stream_options"]
        ## check if custom pricing set ##
        if (
            litellm_params.get("input_cost_per_token") is not None
            or litellm_params.get("input_cost_per_second") is not None
            or litellm_params.get("output_cost_per_token") is not None
            or litellm_params.get("output_cost_per_second") is not None
        ):
            self.custom_pricing = True

        if "custom_llm_provider" in self.model_call_details:
            self.custom_llm_provider = self.model_call_details["custom_llm_provider"]

    def should_run_prompt_management_hooks(
        self,
        non_default_params: Dict,
        prompt_id: Optional[str] = None,
    ) -> bool:
        """
        Return True if prompt management hooks should be run
        """
        if prompt_id:
            return True
        if AnthropicCacheControlHook.should_use_anthropic_cache_control_hook(
            non_default_params
        ):
            return True
        return False

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: Dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        prompt_management_logger: Optional[CustomLogger] = None,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        custom_logger = (
            prompt_management_logger
            or self.get_custom_logger_for_prompt_management(
                model=model, non_default_params=non_default_params
            )
        )

        if custom_logger:
            (
                model,
                messages,
                non_default_params,
            ) = custom_logger.get_chat_completion_prompt(
                model=model,
                messages=messages,
                non_default_params=non_default_params or {},
                prompt_id=prompt_id,
                prompt_variables=prompt_variables,
                dynamic_callback_params=self.standard_callback_dynamic_params,
            )
        self.messages = messages
        return model, messages, non_default_params

    def get_custom_logger_for_prompt_management(
        self, model: str, non_default_params: Dict
    ) -> Optional[CustomLogger]:
        """
        Get a custom logger for prompt management based on model name or available callbacks.

        Args:
            model: The model name to check for prompt management integration

        Returns:
            A CustomLogger instance if one is found, None otherwise
        """
        # First check if model starts with a known custom logger compatible callback
        for callback_name in litellm._known_custom_logger_compatible_callbacks:
            if model.startswith(callback_name):
                custom_logger = _init_custom_logger_compatible_class(
                    logging_integration=callback_name,
                    internal_usage_cache=None,
                    llm_router=None,
                )
                if custom_logger is not None:
                    self.model_call_details["prompt_integration"] = model.split("/")[0]
                    return custom_logger

        # Then check for any registered CustomPromptManagement loggers
        prompt_management_loggers = (
            litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=CustomPromptManagement
            )
        )

        if prompt_management_loggers:
            logger = prompt_management_loggers[0]
            self.model_call_details["prompt_integration"] = logger.__class__.__name__
            return logger

        if anthropic_cache_control_logger := AnthropicCacheControlHook.get_custom_logger_for_anthropic_cache_control_hook(
            non_default_params
        ):
            self.model_call_details["prompt_integration"] = (
                anthropic_cache_control_logger.__class__.__name__
            )
            return anthropic_cache_control_logger

        return None

    def get_custom_logger_for_anthropic_cache_control_hook(
        self, non_default_params: Dict
    ) -> Optional[CustomLogger]:
        if non_default_params.get("cache_control_injection_points", None):
            custom_logger = _init_custom_logger_compatible_class(
                logging_integration="anthropic_cache_control_hook",
                internal_usage_cache=None,
                llm_router=None,
            )
            return custom_logger
        return None

    def _get_raw_request_body(self, data: Optional[Union[dict, str]]) -> dict:
        if data is None:
            return {"error": "Received empty dictionary for raw request body"}
        if isinstance(data, str):
            try:
                return json.loads(data)
            except Exception:
                return {
                    "error": "Unable to parse raw request body. Got - {}".format(data)
                }
        return data

    def _get_masked_api_base(self, api_base: str) -> str:
        if "key=" in api_base:
            # Find the position of "key=" in the string
            key_index = api_base.find("key=") + 4
            # Mask the last 5 characters after "key="
            masked_api_base = api_base[:key_index] + "*" * 5 + api_base[-4:]
        else:
            masked_api_base = api_base
        return str(masked_api_base)

    def _pre_call(self, input, api_key, model=None, additional_args={}):
        """
        Common helper function across the sync + async pre-call function
        """

        self.model_call_details["input"] = input
        self.model_call_details["api_key"] = api_key
        self.model_call_details["additional_args"] = additional_args
        self.model_call_details["log_event_type"] = "pre_api_call"
        if (
            model
        ):  # if model name was changes pre-call, overwrite the initial model call name with the new one
            self.model_call_details["model"] = model
        self.model_call_details["litellm_params"]["api_base"] = (
            self._get_masked_api_base(additional_args.get("api_base", ""))
        )

    def pre_call(self, input, api_key, model=None, additional_args={}):  # noqa: PLR0915
        # Log the exact input to the LLM API
        litellm.error_logs["PRE_CALL"] = locals()
        try:
            self._pre_call(
                input=input,
                api_key=api_key,
                model=model,
                additional_args=additional_args,
            )

            # User Logging -> if you pass in a custom logging function
            self._print_llm_call_debugging_log(
                api_base=additional_args.get("api_base", ""),
                headers=additional_args.get("headers", {}),
                additional_args=additional_args,
            )
            # log raw request to provider (like LangFuse) -- if opted in.
            if (
                self.log_raw_request_response is True
                or log_raw_request_response is True
            ):
                _litellm_params = self.model_call_details.get("litellm_params", {})
                _metadata = _litellm_params.get("metadata", {}) or {}
                try:
                    # [Non-blocking Extra Debug Information in metadata]
                    if turn_off_message_logging is True:
                        _metadata["raw_request"] = (
                            "redacted by litellm. \
                            'litellm.turn_off_message_logging=True'"
                        )
                    else:
                        curl_command = self._get_request_curl_command(
                            api_base=additional_args.get("api_base", ""),
                            headers=additional_args.get("headers", {}),
                            additional_args=additional_args,
                            data=additional_args.get("complete_input_dict", {}),
                        )

                        _metadata["raw_request"] = str(curl_command)
                        # split up, so it's easier to parse in the UI
                        self.model_call_details["raw_request_typed_dict"] = (
                            RawRequestTypedDict(
                                raw_request_api_base=str(
                                    additional_args.get("api_base") or ""
                                ),
                                raw_request_body=self._get_raw_request_body(
                                    additional_args.get("complete_input_dict", {})
                                ),
                                raw_request_headers=self._get_masked_headers(
                                    additional_args.get("headers", {}) or {},
                                    ignore_sensitive_headers=True,
                                ),
                                error=None,
                            )
                        )
                except Exception as e:
                    self.model_call_details["raw_request_typed_dict"] = (
                        RawRequestTypedDict(
                            error=str(e),
                        )
                    )
                    _metadata["raw_request"] = (
                        "Unable to Log \
                        raw request: {}".format(
                            str(e)
                        )
                    )
            if self.logger_fn and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    verbose_logger.exception(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}".format(
                            str(e)
                        )
                    )

            self.model_call_details["api_call_start_time"] = datetime.datetime.now()
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made
            callbacks = litellm.input_callback + (self.dynamic_input_callbacks or [])
            for callback in callbacks:
                try:
                    if callback == "supabase" and supabaseClient is not None:
                        verbose_logger.debug("reaches supabase for logging!")
                        model = self.model_call_details["model"]
                        messages = self.model_call_details["input"]
                        verbose_logger.debug(f"supabaseClient: {supabaseClient}")
                        supabaseClient.input_log_event(
                            model=model,
                            messages=messages,
                            end_user=self.model_call_details.get("user", "default"),
                            litellm_call_id=self.litellm_params["litellm_call_id"],
                            print_verbose=print_verbose,
                        )
                    elif callback == "sentry" and add_breadcrumb:
                        try:
                            details_to_log = copy.deepcopy(self.model_call_details)
                        except Exception:
                            details_to_log = self.model_call_details
                        if litellm.turn_off_message_logging:
                            # make a copy of the _model_Call_details and log it
                            details_to_log.pop("messages", None)
                            details_to_log.pop("input", None)
                            details_to_log.pop("prompt", None)

                        add_breadcrumb(
                            category="litellm.llm_call",
                            message=f"Model Call Details pre-call: {details_to_log}",
                            level="info",
                        )

                    elif isinstance(callback, CustomLogger):  # custom logger class
                        callback.log_pre_api_call(
                            model=self.model,
                            messages=self.messages,
                            kwargs=self.model_call_details,
                        )
                    elif (
                        callable(callback) and customLogger is not None
                    ):  # custom logger functions
                        customLogger.log_input_event(
                            model=self.model,
                            messages=self.messages,
                            kwargs=self.model_call_details,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                except Exception as e:
                    verbose_logger.exception(
                        "litellm.Logging.pre_call(): Exception occured - {}".format(
                            str(e)
                        )
                    )
                    verbose_logger.debug(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}".format(
                    str(e)
                )
            )
            verbose_logger.error(
                f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
            )
            if capture_exception:  # log this error to sentry for debugging
                capture_exception(e)

    def _print_llm_call_debugging_log(
        self,
        api_base: str,
        headers: dict,
        additional_args: dict,
    ):
        """
        Internal debugging helper function

        Prints the RAW curl command sent from LiteLLM
        """
        if _is_debugging_on():
            if json_logs:
                masked_headers = self._get_masked_headers(headers)
                verbose_logger.debug(
                    "POST Request Sent from LiteLLM",
                    extra={"api_base": {api_base}, **masked_headers},
                )
            else:
                headers = additional_args.get("headers", {})
                if headers is None:
                    headers = {}
                data = additional_args.get("complete_input_dict", {})
                api_base = str(additional_args.get("api_base", ""))
                curl_command = self._get_request_curl_command(
                    api_base=api_base,
                    headers=headers,
                    additional_args=additional_args,
                    data=data,
                )
                verbose_logger.debug(f"\033[92m{curl_command}\033[0m\n")

    def _get_request_body(self, data: dict) -> str:
        return str(data)

    def _get_request_curl_command(
        self, api_base: str, headers: Optional[dict], additional_args: dict, data: dict
    ) -> str:
        masked_api_base = self._get_masked_api_base(api_base)
        if headers is None:
            headers = {}
        curl_command = "\n\nPOST Request Sent from LiteLLM:\n"
        curl_command += "curl -X POST \\\n"
        curl_command += f"{masked_api_base} \\\n"
        masked_headers = self._get_masked_headers(headers)
        formatted_headers = " ".join(
            [f"-H '{k}: {v}'" for k, v in masked_headers.items()]
        )
        curl_command += (
            f"{formatted_headers} \\\n" if formatted_headers.strip() != "" else ""
        )
        curl_command += f"-d '{self._get_request_body(data)}'\n"
        if additional_args.get("request_str", None) is not None:
            # print the sagemaker / bedrock client request
            curl_command = "\nRequest Sent from LiteLLM:\n"
            curl_command += additional_args.get("request_str", None)
        elif api_base == "":
            curl_command = str(self.model_call_details)
        return curl_command

    def _get_masked_headers(
        self, headers: dict, ignore_sensitive_headers: bool = False
    ) -> dict:
        """
        Internal debugging helper function

        Masks the headers of the request sent from LiteLLM
        """
        return _get_masked_values(
            headers, ignore_sensitive_values=ignore_sensitive_headers
        )

    def post_call(
        self, original_response, input=None, api_key=None, additional_args={}
    ):
        # Log the exact result from the LLM API, for streaming - log the type of response received
        litellm.error_logs["POST_CALL"] = locals()
        if isinstance(original_response, dict):
            original_response = json.dumps(original_response)
        try:
            self.model_call_details["input"] = input
            self.model_call_details["api_key"] = api_key
            self.model_call_details["original_response"] = original_response
            self.model_call_details["additional_args"] = additional_args
            self.model_call_details["log_event_type"] = "post_api_call"

            if json_logs:
                verbose_logger.debug(
                    "RAW RESPONSE:\n{}\n\n".format(
                        self.model_call_details.get(
                            "original_response", self.model_call_details
                        )
                    ),
                )
            else:
                print_verbose(
                    "RAW RESPONSE:\n{}\n\n".format(
                        self.model_call_details.get(
                            "original_response", self.model_call_details
                        )
                    )
                )
            if self.logger_fn and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    verbose_logger.exception(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}".format(
                            str(e)
                        )
                    )
            original_response = redact_message_input_output_from_logging(
                model_call_details=(
                    self.model_call_details
                    if hasattr(self, "model_call_details")
                    else {}
                ),
                result=original_response,
            )
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made

            callbacks = litellm.input_callback + (self.dynamic_input_callbacks or [])
            for callback in callbacks:
                try:
                    if callback == "sentry" and add_breadcrumb:
                        verbose_logger.debug("reaches sentry breadcrumbing")
                        try:
                            details_to_log = copy.deepcopy(self.model_call_details)
                        except Exception:
                            details_to_log = self.model_call_details
                        if litellm.turn_off_message_logging:
                            # make a copy of the _model_Call_details and log it
                            details_to_log.pop("messages", None)
                            details_to_log.pop("input", None)
                            details_to_log.pop("prompt", None)

                        add_breadcrumb(
                            category="litellm.llm_call",
                            message=f"Model Call Details post-call: {details_to_log}",
                            level="info",
                        )
                    elif isinstance(callback, CustomLogger):  # custom logger class
                        callback.log_post_api_call(
                            kwargs=self.model_call_details,
                            response_obj=None,
                            start_time=self.start_time,
                            end_time=None,
                        )
                except Exception as e:
                    verbose_logger.exception(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while post-call logging with integrations {}".format(
                            str(e)
                        )
                    )
                    verbose_logger.debug(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}".format(
                    str(e)
                )
            )

    def get_response_ms(self) -> float:
        return (
            self.model_call_details.get("end_time", datetime.datetime.now())
            - self.model_call_details.get("start_time", datetime.datetime.now())
        ).total_seconds() * 1000

    def _response_cost_calculator(
        self,
        result: Union[
            ModelResponse,
            ModelResponseStream,
            EmbeddingResponse,
            ImageResponse,
            TranscriptionResponse,
            TextCompletionResponse,
            HttpxBinaryResponseContent,
            RerankResponse,
            Batch,
            FineTuningJob,
            ResponsesAPIResponse,
            ResponseCompletedEvent,
            OpenAIFileObject,
            LiteLLMRealtimeStreamLoggingObject,
        ],
        cache_hit: Optional[bool] = None,
        litellm_model_name: Optional[str] = None,
        router_model_id: Optional[str] = None,
    ) -> Optional[float]:
        """
        Calculate response cost using result + logging object variables.

        used for consistent cost calculation across response headers + logging integrations.
        """

        ## RESPONSE COST ##
        custom_pricing = use_custom_pricing_for_model(
            litellm_params=(
                self.litellm_params if hasattr(self, "litellm_params") else None
            )
        )

        prompt = ""  # use for tts cost calc
        _input = self.model_call_details.get("input", None)
        if _input is not None and isinstance(_input, str):
            prompt = _input

        if cache_hit is None:
            cache_hit = self.model_call_details.get("cache_hit", False)

        try:
            response_cost_calculator_kwargs = {
                "response_object": result,
                "model": litellm_model_name or self.model,
                "cache_hit": cache_hit,
                "custom_llm_provider": self.model_call_details.get(
                    "custom_llm_provider", None
                ),
                "base_model": _get_base_model_from_metadata(
                    model_call_details=self.model_call_details
                ),
                "call_type": self.call_type,
                "optional_params": self.optional_params,
                "custom_pricing": custom_pricing,
                "prompt": prompt,
                "standard_built_in_tools_params": self.standard_built_in_tools_params,
                "router_model_id": router_model_id,
            }
        except Exception as e:  # error creating kwargs for cost calculation
            debug_info = StandardLoggingModelCostFailureDebugInformation(
                error_str=str(e),
                traceback_str=_get_traceback_str_for_error(str(e)),
            )
            verbose_logger.debug(
                f"response_cost_failure_debug_information: {debug_info}"
            )
            self.model_call_details["response_cost_failure_debug_information"] = (
                debug_info
            )
            return None

        try:
            response_cost = litellm.response_cost_calculator(
                **response_cost_calculator_kwargs
            )
            verbose_logger.debug(f"response_cost: {response_cost}")
            return response_cost
        except Exception as e:  # error calculating cost
            debug_info = StandardLoggingModelCostFailureDebugInformation(
                error_str=str(e),
                traceback_str=_get_traceback_str_for_error(str(e)),
                model=response_cost_calculator_kwargs["model"],
                cache_hit=response_cost_calculator_kwargs["cache_hit"],
                custom_llm_provider=response_cost_calculator_kwargs[
                    "custom_llm_provider"
                ],
                base_model=response_cost_calculator_kwargs["base_model"],
                call_type=response_cost_calculator_kwargs["call_type"],
                custom_pricing=response_cost_calculator_kwargs["custom_pricing"],
            )
            verbose_logger.debug(
                f"response_cost_failure_debug_information: {debug_info}"
            )
            self.model_call_details["response_cost_failure_debug_information"] = (
                debug_info
            )

        return None

    async def _response_cost_calculator_async(
        self,
        result: Union[
            ModelResponse,
            ModelResponseStream,
            EmbeddingResponse,
            ImageResponse,
            TranscriptionResponse,
            TextCompletionResponse,
            HttpxBinaryResponseContent,
            RerankResponse,
            Batch,
            FineTuningJob,
        ],
        cache_hit: Optional[bool] = None,
    ) -> Optional[float]:
        return self._response_cost_calculator(result=result, cache_hit=cache_hit)

    def should_run_callback(
        self, callback: litellm.CALLBACK_TYPES, litellm_params: dict, event_hook: str
    ) -> bool:
        if litellm.global_disable_no_log_param:
            return True

        if litellm_params.get("no-log", False) is True:
            # proxy cost tracking cal backs should run

            if not (
                isinstance(callback, CustomLogger)
                and "_PROXY_" in callback.__class__.__name__
            ):
                verbose_logger.debug(
                    f"no-log request, skipping logging for {event_hook} event"
                )
                return False
        return True

    def _update_completion_start_time(self, completion_start_time: datetime.datetime):
        self.completion_start_time = completion_start_time
        self.model_call_details["completion_start_time"] = self.completion_start_time

    def _success_handler_helper_fn(
        self,
        result=None,
        start_time=None,
        end_time=None,
        cache_hit=None,
        standard_logging_object: Optional[StandardLoggingPayload] = None,
    ):
        try:
            if start_time is None:
                start_time = self.start_time
            if end_time is None:
                end_time = datetime.datetime.now()
            if self.completion_start_time is None:
                self.completion_start_time = end_time
                self.model_call_details["completion_start_time"] = (
                    self.completion_start_time
                )
            self.model_call_details["log_event_type"] = "successful_api_call"
            self.model_call_details["end_time"] = end_time
            self.model_call_details["cache_hit"] = cache_hit

            if self.call_type == CallTypes.anthropic_messages.value:
                result = self._handle_anthropic_messages_response_logging(result=result)
            ## if model in model cost map - log the response cost
            ## else set cost to None

            logging_result = result

            if self.call_type == CallTypes.arealtime.value and isinstance(result, list):
                combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
                    results=result
                )
                logging_result = (
                    RealtimeAPITokenUsageProcessor.create_logging_realtime_object(
                        usage=combined_usage_object,
                        results=result,
                    )
                )

                # self.model_call_details[
                #     "response_cost"
                # ] = handle_realtime_stream_cost_calculation(
                #     results=result,
                #     combined_usage_object=combined_usage_object,
                #     custom_llm_provider=self.custom_llm_provider,
                #     litellm_model_name=self.model,
                # )
                # self.model_call_details["combined_usage_object"] = combined_usage_object
            if (
                standard_logging_object is None
                and result is not None
                and self.stream is not True
            ):
                if (
                    isinstance(logging_result, ModelResponse)
                    or isinstance(logging_result, ModelResponseStream)
                    or isinstance(logging_result, EmbeddingResponse)
                    or isinstance(logging_result, ImageResponse)
                    or isinstance(logging_result, TranscriptionResponse)
                    or isinstance(logging_result, TextCompletionResponse)
                    or isinstance(logging_result, HttpxBinaryResponseContent)  # tts
                    or isinstance(logging_result, RerankResponse)
                    or isinstance(logging_result, FineTuningJob)
                    or isinstance(logging_result, LiteLLMBatch)
                    or isinstance(logging_result, ResponsesAPIResponse)
                    or isinstance(logging_result, OpenAIFileObject)
                    or isinstance(logging_result, LiteLLMRealtimeStreamLoggingObject)
                ):
                    ## HIDDEN PARAMS ##
                    hidden_params = getattr(logging_result, "_hidden_params", {})
                    if hidden_params:
                        # add to metadata for logging
                        if self.model_call_details.get("litellm_params") is not None:
                            self.model_call_details["litellm_params"].setdefault(
                                "metadata", {}
                            )
                            if (
                                self.model_call_details["litellm_params"]["metadata"]
                                is None
                            ):
                                self.model_call_details["litellm_params"][
                                    "metadata"
                                ] = {}

                            self.model_call_details["litellm_params"]["metadata"][  # type: ignore
                                "hidden_params"
                            ] = getattr(
                                logging_result, "_hidden_params", {}
                            )
                    ## RESPONSE COST - Only calculate if not in hidden_params ##
                    if "response_cost" in hidden_params:
                        self.model_call_details["response_cost"] = hidden_params[
                            "response_cost"
                        ]
                    else:
                        self.model_call_details["response_cost"] = (
                            self._response_cost_calculator(result=logging_result)
                        )
                    ## STANDARDIZED LOGGING PAYLOAD

                    self.model_call_details["standard_logging_object"] = (
                        get_standard_logging_object_payload(
                            kwargs=self.model_call_details,
                            init_response_obj=logging_result,
                            start_time=start_time,
                            end_time=end_time,
                            logging_obj=self,
                            status="success",
                            standard_built_in_tools_params=self.standard_built_in_tools_params,
                        )
                    )
                elif isinstance(result, dict) or isinstance(result, list):
                    ## STANDARDIZED LOGGING PAYLOAD
                    self.model_call_details["standard_logging_object"] = (
                        get_standard_logging_object_payload(
                            kwargs=self.model_call_details,
                            init_response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            logging_obj=self,
                            status="success",
                            standard_built_in_tools_params=self.standard_built_in_tools_params,
                        )
                    )
            elif standard_logging_object is not None:
                self.model_call_details["standard_logging_object"] = (
                    standard_logging_object
                )
            else:  # streaming chunks + image gen.
                self.model_call_details["response_cost"] = None

            if (
                litellm.max_budget
                and self.stream is False
                and result is not None
                and isinstance(result, dict)
                and "content" in result
            ):
                time_diff = (end_time - start_time).total_seconds()
                float_diff = float(time_diff)
                litellm._current_cost += litellm.completion_cost(
                    model=self.model,
                    prompt="",
                    completion=getattr(result, "content", ""),
                    total_time=float_diff,
                    standard_built_in_tools_params=self.standard_built_in_tools_params,
                )

            return start_time, end_time, result
        except Exception as e:
            raise Exception(f"[Non-Blocking] LiteLLM.Success_Call Error: {str(e)}")

    def success_handler(  # noqa: PLR0915
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        verbose_logger.debug(
            f"Logging Details LiteLLM-Success Call: Cache_hit={cache_hit}"
        )
        start_time, end_time, result = self._success_handler_helper_fn(
            start_time=start_time,
            end_time=end_time,
            result=result,
            cache_hit=cache_hit,
            standard_logging_object=kwargs.get("standard_logging_object", None),
        )
        try:
            ## BUILD COMPLETE STREAMED RESPONSE
            complete_streaming_response: Optional[
                Union[ModelResponse, TextCompletionResponse, ResponsesAPIResponse]
            ] = None
            if "complete_streaming_response" in self.model_call_details:
                return  # break out of this.
            complete_streaming_response = self._get_assembled_streaming_response(
                result=result,
                start_time=start_time,
                end_time=end_time,
                is_async=False,
                streaming_chunks=self.sync_streaming_chunks,
            )
            if complete_streaming_response is not None:
                verbose_logger.debug(
                    "Logging Details LiteLLM-Success Call streaming complete"
                )
                self.model_call_details["complete_streaming_response"] = (
                    complete_streaming_response
                )
                self.model_call_details["response_cost"] = (
                    self._response_cost_calculator(result=complete_streaming_response)
                )
                ## STANDARDIZED LOGGING PAYLOAD
                self.model_call_details["standard_logging_object"] = (
                    get_standard_logging_object_payload(
                        kwargs=self.model_call_details,
                        init_response_obj=complete_streaming_response,
                        start_time=start_time,
                        end_time=end_time,
                        logging_obj=self,
                        status="success",
                        standard_built_in_tools_params=self.standard_built_in_tools_params,
                    )
                )
            callbacks = self.get_combined_callback_list(
                dynamic_success_callbacks=self.dynamic_success_callbacks,
                global_callbacks=litellm.success_callback,
            )

            ## REDACT MESSAGES ##
            result = redact_message_input_output_from_logging(
                model_call_details=(
                    self.model_call_details
                    if hasattr(self, "model_call_details")
                    else {}
                ),
                result=result,
            )
            ## LOGGING HOOK ##
            for callback in callbacks:
                if isinstance(callback, CustomLogger):
                    self.model_call_details, result = callback.logging_hook(
                        kwargs=self.model_call_details,
                        result=result,
                        call_type=self.call_type,
                    )

            for callback in callbacks:
                try:
                    litellm_params = self.model_call_details.get("litellm_params", {})
                    should_run = self.should_run_callback(
                        callback=callback,
                        litellm_params=litellm_params,
                        event_hook="success_handler",
                    )
                    if not should_run:
                        continue
                    if callback == "promptlayer" and promptLayerLogger is not None:
                        print_verbose("reaches promptlayer for logging!")
                        promptLayerLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "supabase" and supabaseClient is not None:
                        print_verbose("reaches supabase for logging!")
                        kwargs = self.model_call_details

                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                print_verbose("reaches supabase for streaming logging!")
                                result = kwargs["complete_streaming_response"]

                        model = kwargs["model"]
                        messages = kwargs["messages"]
                        optional_params = kwargs.get("optional_params", {})
                        litellm_params = kwargs.get("litellm_params", {})
                        supabaseClient.log_event(
                            model=model,
                            messages=messages,
                            end_user=optional_params.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=litellm_params.get(
                                "litellm_call_id", str(uuid.uuid4())
                            ),
                            print_verbose=print_verbose,
                        )
                    if callback == "wandb" and weightsBiasesLogger is not None:
                        print_verbose("reaches wandb for logging!")
                        weightsBiasesLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "logfire" and logfireLogger is not None:
                        verbose_logger.debug("reaches logfire for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v

                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                print_verbose("reaches logfire for streaming logging!")
                                result = kwargs["complete_streaming_response"]

                        logfireLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            level=LogfireLevel.INFO.value,  # type: ignore
                        )

                    if callback == "lunary" and lunaryLogger is not None:
                        print_verbose("reaches lunary for logging!")
                        model = self.model
                        kwargs = self.model_call_details

                        input = kwargs.get("messages", kwargs.get("input", None))

                        type = (
                            "embed"
                            if self.call_type == CallTypes.embedding.value
                            else "llm"
                        )

                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                result = kwargs["complete_streaming_response"]

                        lunaryLogger.log_event(
                            type=type,
                            kwargs=kwargs,
                            event="end",
                            model=model,
                            input=input,
                            user_id=kwargs.get("user", None),
                            # user_props=self.model_call_details.get("user_props", None),
                            extra=kwargs.get("optional_params", {}),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            run_id=self.litellm_call_id,
                            print_verbose=print_verbose,
                        )
                    if callback == "helicone" and heliconeLogger is not None:
                        print_verbose("reaches helicone for logging!")
                        model = self.model
                        messages = self.model_call_details["input"]
                        kwargs = self.model_call_details

                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                print_verbose("reaches helicone for streaming logging!")
                                result = kwargs["complete_streaming_response"]

                        heliconeLogger.log_success(
                            model=model,
                            messages=messages,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            kwargs=kwargs,
                        )
                    if callback == "langfuse":
                        global langFuseLogger
                        print_verbose("reaches langfuse for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose("reaches langfuse for streaming logging!")
                                result = kwargs["complete_streaming_response"]

                        langfuse_logger_to_use = LangFuseHandler.get_langfuse_logger_for_request(
                            globalLangfuseLogger=langFuseLogger,
                            standard_callback_dynamic_params=self.standard_callback_dynamic_params,
                            in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
                        )
                        if langfuse_logger_to_use is not None:
                            _response = langfuse_logger_to_use.log_event_on_langfuse(
                                kwargs=kwargs,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                                user_id=kwargs.get("user", None),
                            )
                            if _response is not None and isinstance(_response, dict):
                                _trace_id = _response.get("trace_id", None)
                                if _trace_id is not None:
                                    in_memory_trace_id_cache.set_cache(
                                        litellm_call_id=self.litellm_call_id,
                                        service_name="langfuse",
                                        trace_id=_trace_id,
                                    )
                    if callback == "generic":
                        global genericAPILogger
                        verbose_logger.debug("reaches langfuse for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose("reaches langfuse for streaming logging!")
                                result = kwargs["complete_streaming_response"]
                        if genericAPILogger is None:
                            genericAPILogger = GenericAPILogger()  # type: ignore
                        genericAPILogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "greenscale" and greenscaleLogger is not None:
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose(
                                    "reaches greenscale for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]

                        greenscaleLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "athina" and athinaLogger is not None:
                        deep_copy = {}
                        for k, v in self.model_call_details.items():
                            deep_copy[k] = v
                        athinaLogger.log_event(
                            kwargs=deep_copy,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "traceloop":
                        deep_copy = {}
                        for k, v in self.model_call_details.items():
                            if k != "original_response":
                                deep_copy[k] = v
                        traceloopLogger.log_event(
                            kwargs=deep_copy,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "s3":
                        global s3Logger
                        if s3Logger is None:
                            s3Logger = S3Logger()
                        if self.stream:
                            if "complete_streaming_response" in self.model_call_details:
                                print_verbose(
                                    "S3Logger Logger: Got Stream Event - Completed Stream Response"
                                )
                                s3Logger.log_event(
                                    kwargs=self.model_call_details,
                                    response_obj=self.model_call_details[
                                        "complete_streaming_response"
                                    ],
                                    start_time=start_time,
                                    end_time=end_time,
                                    print_verbose=print_verbose,
                                )
                            else:
                                print_verbose(
                                    "S3Logger Logger: Got Stream Event - No complete stream response as yet"
                                )
                        else:
                            s3Logger.log_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                                print_verbose=print_verbose,
                            )

                    if (
                        callback == "openmeter"
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        is not True
                    ):
                        global openMeterLogger
                        if openMeterLogger is None:
                            print_verbose("Instantiates openmeter client")
                            openMeterLogger = OpenMeterLogger()
                        if self.stream and complete_streaming_response is None:
                            openMeterLogger.log_stream_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            if self.stream and complete_streaming_response:
                                self.model_call_details["complete_response"] = (
                                    self.model_call_details.get(
                                        "complete_streaming_response", {}
                                    )
                                )
                                result = self.model_call_details["complete_response"]
                            openMeterLogger.log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )

                    if (
                        isinstance(callback, CustomLogger)
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        is not True
                        and self.call_type
                        != CallTypes.pass_through.value  # pass-through endpoints call async_log_success_event
                    ):  # custom logger class
                        if self.stream and complete_streaming_response is None:
                            callback.log_stream_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            if self.stream and complete_streaming_response:
                                self.model_call_details["complete_response"] = (
                                    self.model_call_details.get(
                                        "complete_streaming_response", {}
                                    )
                                )
                                result = self.model_call_details["complete_response"]

                            callback.log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                    if (
                        callable(callback) is True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        is not True
                        and customLogger is not None
                    ):  # custom logger functions
                        print_verbose(
                            "success callbacks: Running Custom Callback Function - {}".format(
                                callback
                            )
                        )

                        customLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )

                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging {}".format(
                    str(e)
                ),
            )

    async def async_success_handler(  # noqa: PLR0915
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        """
        Implementing async callbacks, to handle asyncio event loop issues when custom integrations need to use async functions.
        """
        print_verbose(
            "Logging Details LiteLLM-Async Success Call, cache_hit={}".format(cache_hit)
        )

        ## CALCULATE COST FOR BATCH JOBS
        if self.call_type == CallTypes.aretrieve_batch.value and isinstance(
            result, LiteLLMBatch
        ):
            response_cost, batch_usage, batch_models = await _handle_completed_batch(
                batch=result, custom_llm_provider=self.custom_llm_provider
            )

            result._hidden_params["response_cost"] = response_cost
            result._hidden_params["batch_models"] = batch_models
            result.usage = batch_usage

        start_time, end_time, result = self._success_handler_helper_fn(
            start_time=start_time,
            end_time=end_time,
            result=result,
            cache_hit=cache_hit,
            standard_logging_object=kwargs.get("standard_logging_object", None),
        )

        ## BUILD COMPLETE STREAMED RESPONSE
        if "async_complete_streaming_response" in self.model_call_details:
            return  # break out of this.
        complete_streaming_response: Optional[
            Union[ModelResponse, TextCompletionResponse, ResponsesAPIResponse]
        ] = self._get_assembled_streaming_response(
            result=result,
            start_time=start_time,
            end_time=end_time,
            is_async=True,
            streaming_chunks=self.streaming_chunks,
        )

        if complete_streaming_response is not None:
            print_verbose("Async success callbacks: Got a complete streaming response")

            self.model_call_details["async_complete_streaming_response"] = (
                complete_streaming_response
            )
            try:
                if self.model_call_details.get("cache_hit", False) is True:
                    self.model_call_details["response_cost"] = 0.0
                else:
                    # check if base_model set on azure
                    _get_base_model_from_metadata(
                        model_call_details=self.model_call_details
                    )
                    # base_model defaults to None if not set on model_info
                    self.model_call_details["response_cost"] = (
                        self._response_cost_calculator(
                            result=complete_streaming_response
                        )
                    )

                verbose_logger.debug(
                    f"Model={self.model}; cost={self.model_call_details['response_cost']}"
                )
            except litellm.NotFoundError:
                verbose_logger.warning(
                    f"Model={self.model} not found in completion cost map. Setting 'response_cost' to None"
                )
                self.model_call_details["response_cost"] = None

            ## STANDARDIZED LOGGING PAYLOAD
            self.model_call_details["standard_logging_object"] = (
                get_standard_logging_object_payload(
                    kwargs=self.model_call_details,
                    init_response_obj=complete_streaming_response,
                    start_time=start_time,
                    end_time=end_time,
                    logging_obj=self,
                    status="success",
                    standard_built_in_tools_params=self.standard_built_in_tools_params,
                )
            )
        callbacks = self.get_combined_callback_list(
            dynamic_success_callbacks=self.dynamic_async_success_callbacks,
            global_callbacks=litellm._async_success_callback,
        )

        result = redact_message_input_output_from_logging(
            model_call_details=(
                self.model_call_details if hasattr(self, "model_call_details") else {}
            ),
            result=result,
        )

        ## LOGGING HOOK ##

        for callback in callbacks:
            if isinstance(callback, CustomGuardrail):
                from litellm.types.guardrails import GuardrailEventHooks

                if (
                    callback.should_run_guardrail(
                        data=self.model_call_details,
                        event_type=GuardrailEventHooks.logging_only,
                    )
                    is not True
                ):
                    continue

                self.model_call_details, result = await callback.async_logging_hook(
                    kwargs=self.model_call_details,
                    result=result,
                    call_type=self.call_type,
                )
            elif isinstance(callback, CustomLogger):
                result = redact_message_input_output_from_custom_logger(
                    result=result, litellm_logging_obj=self, custom_logger=callback
                )
                self.model_call_details, result = await callback.async_logging_hook(
                    kwargs=self.model_call_details,
                    result=result,
                    call_type=self.call_type,
                )

        for callback in callbacks:
            # check if callback can run for this request
            litellm_params = self.model_call_details.get("litellm_params", {})
            should_run = self.should_run_callback(
                callback=callback,
                litellm_params=litellm_params,
                event_hook="async_success_handler",
            )
            if not should_run:
                continue
            try:
                if callback == "openmeter" and openMeterLogger is not None:
                    if self.stream is True:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):
                            await openMeterLogger.async_log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            await openMeterLogger.async_log_stream_event(  # [TODO]: move this to being an async log stream event function
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                    else:
                        await openMeterLogger.async_log_success_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                        )
                if isinstance(callback, CustomLogger):  # custom logger class
                    if self.stream is True:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):
                            await callback.async_log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            await callback.async_log_stream_event(  # [TODO]: move this to being an async log stream event function
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                    else:
                        await callback.async_log_success_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                        )
                if callable(callback):  # custom logger functions
                    global customLogger
                    if customLogger is None:
                        customLogger = CustomLogger()
                    if self.stream:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):
                            await customLogger.async_log_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                                print_verbose=print_verbose,
                                callback_func=callback,
                            )
                    else:
                        await customLogger.async_log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                if callback == "dynamodb":
                    global dynamoLogger
                    if dynamoLogger is None:
                        dynamoLogger = DyanmoDBLogger()
                    if self.stream:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):
                            print_verbose(
                                "DynamoDB Logger: Got Stream Event - Completed Stream Response"
                            )
                            await dynamoLogger._async_log_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                                print_verbose=print_verbose,
                            )
                        else:
                            print_verbose(
                                "DynamoDB Logger: Got Stream Event - No complete stream response as yet"
                            )
                    else:
                        await dynamoLogger._async_log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
            except Exception:
                verbose_logger.error(
                    f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging {traceback.format_exc()}"
                )
                pass

    def _failure_handler_helper_fn(
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = datetime.datetime.now()

        # on some exceptions, model_call_details is not always initialized, this ensures that we still log those exceptions
        if not hasattr(self, "model_call_details"):
            self.model_call_details = {}

        self.model_call_details["log_event_type"] = "failed_api_call"
        self.model_call_details["exception"] = exception
        self.model_call_details["traceback_exception"] = traceback_exception
        self.model_call_details["end_time"] = end_time
        self.model_call_details.setdefault("original_response", None)
        self.model_call_details["response_cost"] = 0

        if hasattr(exception, "headers") and isinstance(exception.headers, dict):
            self.model_call_details.setdefault("litellm_params", {})
            metadata = (
                self.model_call_details["litellm_params"].get("metadata", {}) or {}
            )
            metadata.update(exception.headers)

        ## STANDARDIZED LOGGING PAYLOAD

        self.model_call_details["standard_logging_object"] = (
            get_standard_logging_object_payload(
                kwargs=self.model_call_details,
                init_response_obj={},
                start_time=start_time,
                end_time=end_time,
                logging_obj=self,
                status="failure",
                error_str=str(exception),
                original_exception=exception,
                standard_built_in_tools_params=self.standard_built_in_tools_params,
            )
        )
        return start_time, end_time

    async def special_failure_handlers(self, exception: Exception):
        """
        Custom events, emitted for specific failures.

        Currently just for router model group rate limit error
        """
        from litellm.types.router import RouterErrors

        litellm_params: dict = self.model_call_details.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}

        ## BASE CASE ## check if rate limit error for model group size 1
        is_base_case = False
        if metadata.get("model_group_size") is not None:
            model_group_size = metadata.get("model_group_size")
            if isinstance(model_group_size, int) and model_group_size == 1:
                is_base_case = True
        ## check if special error ##
        if (
            RouterErrors.no_deployments_available.value not in str(exception)
            and is_base_case is False
        ):
            return

        ## get original model group ##

        model_group = metadata.get("model_group") or None
        for callback in litellm._async_failure_callback:
            if isinstance(callback, CustomLogger):  # custom logger class
                await callback.log_model_group_rate_limit_error(
                    exception=exception,
                    original_model_group=model_group,
                    kwargs=self.model_call_details,
                )  # type: ignore

    def failure_handler(  # noqa: PLR0915
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        verbose_logger.debug(
            f"Logging Details LiteLLM-Failure Call: {litellm.failure_callback}"
        )
        try:
            start_time, end_time = self._failure_handler_helper_fn(
                exception=exception,
                traceback_exception=traceback_exception,
                start_time=start_time,
                end_time=end_time,
            )
            callbacks = self.get_combined_callback_list(
                dynamic_success_callbacks=self.dynamic_failure_callbacks,
                global_callbacks=litellm.failure_callback,
            )

            result = None  # result sent to all loggers, init this to None incase it's not created

            result = redact_message_input_output_from_logging(
                model_call_details=(
                    self.model_call_details
                    if hasattr(self, "model_call_details")
                    else {}
                ),
                result=result,
            )
            for callback in callbacks:
                try:
                    if callback == "lunary" and lunaryLogger is not None:
                        print_verbose("reaches lunary for logging error!")

                        model = self.model

                        input = self.model_call_details["input"]

                        _type = (
                            "embed"
                            if self.call_type == CallTypes.embedding.value
                            else "llm"
                        )

                        lunaryLogger.log_event(
                            kwargs=self.model_call_details,
                            type=_type,
                            event="error",
                            user_id=self.model_call_details.get("user", "default"),
                            model=model,
                            input=input,
                            error=traceback_exception,
                            run_id=self.litellm_call_id,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "sentry":
                        print_verbose("sending exception to sentry")
                        if capture_exception:
                            capture_exception(exception)
                        else:
                            print_verbose(
                                f"capture exception not initialized: {capture_exception}"
                            )
                    elif callback == "supabase" and supabaseClient is not None:
                        print_verbose("reaches supabase for logging!")
                        print_verbose(f"supabaseClient: {supabaseClient}")
                        supabaseClient.log_event(
                            model=self.model if hasattr(self, "model") else "",
                            messages=self.messages,
                            end_user=self.model_call_details.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=self.model_call_details["litellm_call_id"],
                            print_verbose=print_verbose,
                        )
                    if (
                        callable(callback) and customLogger is not None
                    ):  # custom logger functions
                        customLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                    if (
                        isinstance(callback, CustomLogger)
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        is not True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        is not True
                    ):  # custom logger class
                        callback.log_failure_event(
                            start_time=start_time,
                            end_time=end_time,
                            response_obj=result,
                            kwargs=self.model_call_details,
                        )
                    if callback == "langfuse":
                        global langFuseLogger
                        verbose_logger.debug("reaches langfuse for logging failure")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        langfuse_logger_to_use = LangFuseHandler.get_langfuse_logger_for_request(
                            globalLangfuseLogger=langFuseLogger,
                            standard_callback_dynamic_params=self.standard_callback_dynamic_params,
                            in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
                        )
                        _response = langfuse_logger_to_use.log_event_on_langfuse(
                            start_time=start_time,
                            end_time=end_time,
                            response_obj=None,
                            user_id=kwargs.get("user", None),
                            status_message=str(exception),
                            level="ERROR",
                            kwargs=self.model_call_details,
                        )
                        if _response is not None and isinstance(_response, dict):
                            _trace_id = _response.get("trace_id", None)
                            if _trace_id is not None:
                                in_memory_trace_id_cache.set_cache(
                                    litellm_call_id=self.litellm_call_id,
                                    service_name="langfuse",
                                    trace_id=_trace_id,
                                )
                    if callback == "traceloop":
                        traceloopLogger.log_event(
                            start_time=start_time,
                            end_time=end_time,
                            response_obj=None,
                            user_id=self.model_call_details.get("user", None),
                            print_verbose=print_verbose,
                            status_message=str(exception),
                            level="ERROR",
                            kwargs=self.model_call_details,
                        )
                    if callback == "logfire" and logfireLogger is not None:
                        verbose_logger.debug("reaches logfire for failure logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        kwargs["exception"] = exception

                        logfireLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            level=LogfireLevel.ERROR.value,  # type: ignore
                            print_verbose=print_verbose,
                        )

                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure logging with integrations {str(e)}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure logging {}".format(
                    str(e)
                )
            )

    async def async_failure_handler(
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        """
        Implementing async callbacks, to handle asyncio event loop issues when custom integrations need to use async functions.
        """
        await self.special_failure_handlers(exception=exception)
        start_time, end_time = self._failure_handler_helper_fn(
            exception=exception,
            traceback_exception=traceback_exception,
            start_time=start_time,
            end_time=end_time,
        )

        callbacks = self.get_combined_callback_list(
            dynamic_success_callbacks=self.dynamic_async_failure_callbacks,
            global_callbacks=litellm._async_failure_callback,
        )

        result = None  # result sent to all loggers, init this to None incase it's not created

        for callback in callbacks:
            try:
                if isinstance(callback, CustomLogger):  # custom logger class
                    await callback.async_log_failure_event(
                        kwargs=self.model_call_details,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                    )  # type: ignore
                if (
                    callable(callback) and customLogger is not None
                ):  # custom logger functions
                    await customLogger.async_log_event(
                        kwargs=self.model_call_details,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                        callback_func=callback,
                    )
            except Exception as e:
                verbose_logger.exception(
                    "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure \
                        logging {}\nCallback={}".format(
                        str(e), callback
                    )
                )

    def _get_trace_id(self, service_name: Literal["langfuse"]) -> Optional[str]:
        """
        For the given service (e.g. langfuse), return the trace_id actually logged.

        Used for constructing the url in slack alerting.

        Returns:
            - str: The logged trace id
            - None: If trace id not yet emitted.
        """
        trace_id: Optional[str] = None
        if service_name == "langfuse":
            trace_id = in_memory_trace_id_cache.get_cache(
                litellm_call_id=self.litellm_call_id, service_name=service_name
            )

        return trace_id

    def _get_callback_object(self, service_name: Literal["langfuse"]) -> Optional[Any]:
        """
        Return dynamic callback object.

        Meant to solve issue when doing key-based/team-based logging
        """
        global langFuseLogger

        if service_name == "langfuse":
            if langFuseLogger is None or (
                (
                    self.standard_callback_dynamic_params.get("langfuse_public_key")
                    is not None
                    and self.standard_callback_dynamic_params.get("langfuse_public_key")
                    != langFuseLogger.public_key
                )
                or (
                    self.standard_callback_dynamic_params.get("langfuse_public_key")
                    is not None
                    and self.standard_callback_dynamic_params.get("langfuse_public_key")
                    != langFuseLogger.public_key
                )
                or (
                    self.standard_callback_dynamic_params.get("langfuse_host")
                    is not None
                    and self.standard_callback_dynamic_params.get("langfuse_host")
                    != langFuseLogger.langfuse_host
                )
            ):
                return LangFuseLogger(
                    langfuse_public_key=self.standard_callback_dynamic_params.get(
                        "langfuse_public_key"
                    ),
                    langfuse_secret=self.standard_callback_dynamic_params.get(
                        "langfuse_secret"
                    ),
                    langfuse_host=self.standard_callback_dynamic_params.get(
                        "langfuse_host"
                    ),
                )
            return langFuseLogger

        return None

    def handle_sync_success_callbacks_for_async_calls(
        self,
        result: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        """
        Handles calling success callbacks for Async calls.

        Why: Some callbacks - `langfuse`, `s3` are sync callbacks. We need to call them in the executor.
        """
        if self._should_run_sync_callbacks_for_async_calls() is False:
            return

        executor.submit(
            self.success_handler,
            result,
            start_time,
            end_time,
        )

    def _should_run_sync_callbacks_for_async_calls(self) -> bool:
        """
        Returns:
            - bool: True if sync callbacks should be run for async calls. eg. `langfuse`, `s3`
        """
        _combined_sync_callbacks = self.get_combined_callback_list(
            dynamic_success_callbacks=self.dynamic_success_callbacks,
            global_callbacks=litellm.success_callback,
        )
        _filtered_success_callbacks = self._remove_internal_custom_logger_callbacks(
            _combined_sync_callbacks
        )
        _filtered_success_callbacks = self._remove_internal_litellm_callbacks(
            _filtered_success_callbacks
        )
        return len(_filtered_success_callbacks) > 0

    def get_combined_callback_list(
        self, dynamic_success_callbacks: Optional[List], global_callbacks: List
    ) -> List:
        if dynamic_success_callbacks is None:
            return global_callbacks
        return list(set(dynamic_success_callbacks + global_callbacks))

    def _remove_internal_litellm_callbacks(self, callbacks: List) -> List:
        """
        Creates a filtered list of callbacks, excluding internal LiteLLM callbacks.

        Args:
            callbacks: List of callback functions/strings to filter

        Returns:
            List of filtered callbacks with internal ones removed
        """
        filtered = [
            cb for cb in callbacks if not self._is_internal_litellm_proxy_callback(cb)
        ]

        verbose_logger.debug(f"Filtered callbacks: {filtered}")
        return filtered

    def _get_callback_name(self, cb) -> str:
        """
        Helper to get the name of a callback function

        Args:
            cb: The callback function/string to get the name of

        Returns:
            The name of the callback
        """
        if hasattr(cb, "__name__"):
            return cb.__name__
        if hasattr(cb, "__func__"):
            return cb.__func__.__name__
        return str(cb)

    def _is_internal_litellm_proxy_callback(self, cb) -> bool:
        """Helper to check if a callback is internal"""
        INTERNAL_PREFIXES = [
            "_PROXY",
            "_service_logger.ServiceLogging",
            "sync_deployment_callback_on_success",
        ]
        if isinstance(cb, str):
            return False

        if not callable(cb):
            return True

        cb_name = self._get_callback_name(cb)
        return any(prefix in cb_name for prefix in INTERNAL_PREFIXES)

    def _remove_internal_custom_logger_callbacks(self, callbacks: List) -> List:
        """
        Removes internal custom logger callbacks from the list.
        """
        _new_callbacks = []
        for _c in callbacks:
            if isinstance(_c, CustomLogger):
                continue
            elif (
                isinstance(_c, str)
                and _c in litellm._known_custom_logger_compatible_callbacks
            ):
                continue
            _new_callbacks.append(_c)
        return _new_callbacks

    def _get_assembled_streaming_response(
        self,
        result: Union[
            ModelResponse,
            TextCompletionResponse,
            ModelResponseStream,
            ResponseCompletedEvent,
            Any,
        ],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        is_async: bool,
        streaming_chunks: List[Any],
    ) -> Optional[Union[ModelResponse, TextCompletionResponse, ResponsesAPIResponse]]:
        if isinstance(result, ModelResponse):
            return result
        elif isinstance(result, TextCompletionResponse):
            return result
        elif isinstance(result, ResponseCompletedEvent):
            return result.response
        return None

    def _handle_anthropic_messages_response_logging(self, result: Any) -> ModelResponse:
        """
        Handles logging for Anthropic messages responses.

        Args:
            result: The response object from the model call

        Returns:
            The the response object from the model call

        - For Non-streaming responses, we need to transform the response to a ModelResponse object.
        - For streaming responses, anthropic_messages handler calls success_handler with a assembled ModelResponse.
        """
        if self.stream and isinstance(result, ModelResponse):
            return result

        result = litellm.AnthropicConfig().transform_response(
            raw_response=self.model_call_details["httpx_response"],
            model_response=litellm.ModelResponse(),
            model=self.model,
            messages=[],
            logging_obj=self,
            optional_params={},
            api_key="",
            request_data={},
            encoding=litellm.encoding,
            json_mode=False,
            litellm_params={},
        )
        return result


def _get_masked_values(
    sensitive_object: dict,
    ignore_sensitive_values: bool = False,
    mask_all_values: bool = False,
    unmasked_length: int = 4,
    number_of_asterisks: Optional[int] = 4,
) -> dict:
    """
    Internal debugging helper function

    Masks the headers of the request sent from LiteLLM

    Args:
        masked_length: Optional length for the masked portion (number of *). If set, will use exactly this many *
                     regardless of original string length. The total length will be unmasked_length + masked_length.
    """
    sensitive_keywords = [
        "authorization",
        "token",
        "key",
        "secret",
    ]
    return {
        k: (
            (
                v[: unmasked_length // 2]
                + "*" * number_of_asterisks
                + v[-unmasked_length // 2 :]
            )
            if (
                isinstance(v, str)
                and len(v) > unmasked_length
                and number_of_asterisks is not None
            )
            else (
                (
                    v[: unmasked_length // 2]
                    + "*" * (len(v) - unmasked_length)
                    + v[-unmasked_length // 2 :]
                )
                if (isinstance(v, str) and len(v) > unmasked_length)
                else "*****"
            )
        )
        for k, v in sensitive_object.items()
        if not ignore_sensitive_values
        or not any(
            sensitive_keyword in k.lower() for sensitive_keyword in sensitive_keywords
        )
    }


def set_callbacks(callback_list, function_id=None):  # noqa: PLR0915
    """
    Globally sets the callback client
    """
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, traceloopLogger, athinaLogger, heliconeLogger, supabaseClient, lunaryLogger, promptLayerLogger, langFuseLogger, customLogger, weightsBiasesLogger, logfireLogger, dynamoLogger, s3Logger, dataDogLogger, prometheusLogger, greenscaleLogger, openMeterLogger

    try:
        for callback in callback_list:
            if callback == "sentry":
                try:
                    import sentry_sdk
                except ImportError:
                    print_verbose("Package 'sentry_sdk' is missing. Installing it...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "sentry_sdk"]
                    )
                    import sentry_sdk
                sentry_sdk_instance = sentry_sdk
                sentry_trace_rate = (
                    os.environ.get("SENTRY_API_TRACE_RATE")
                    if "SENTRY_API_TRACE_RATE" in os.environ
                    else "1.0"
                )
                sentry_sdk_instance.init(
                    dsn=os.environ.get("SENTRY_DSN"),
                    traces_sample_rate=float(sentry_trace_rate),  # type: ignore
                )
                capture_exception = sentry_sdk_instance.capture_exception
                add_breadcrumb = sentry_sdk_instance.add_breadcrumb
            elif callback == "posthog":
                try:
                    from posthog import Posthog
                except ImportError:
                    print_verbose("Package 'posthog' is missing. Installing it...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "posthog"]
                    )
                    from posthog import Posthog
                posthog = Posthog(
                    project_api_key=os.environ.get("POSTHOG_API_KEY"),
                    host=os.environ.get("POSTHOG_API_URL"),
                )
            elif callback == "slack":
                try:
                    from slack_bolt import App
                except ImportError:
                    print_verbose("Package 'slack_bolt' is missing. Installing it...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "slack_bolt"]
                    )
                    from slack_bolt import App
                slack_app = App(
                    token=os.environ.get("SLACK_API_TOKEN"),
                    signing_secret=os.environ.get("SLACK_API_SECRET"),
                )
                alerts_channel = os.environ["SLACK_API_CHANNEL"]
                print_verbose(f"Initialized Slack App: {slack_app}")
            elif callback == "traceloop":
                traceloopLogger = TraceloopLogger()
            elif callback == "athina":
                athinaLogger = AthinaLogger()
                print_verbose("Initialized Athina Logger")
            elif callback == "helicone":
                heliconeLogger = HeliconeLogger()
            elif callback == "lunary":
                lunaryLogger = LunaryLogger()
            elif callback == "promptlayer":
                promptLayerLogger = PromptLayerLogger()
            elif callback == "langfuse":
                langFuseLogger = LangFuseLogger(
                    langfuse_public_key=None, langfuse_secret=None, langfuse_host=None
                )
            elif callback == "openmeter":
                openMeterLogger = OpenMeterLogger()
            elif callback == "datadog":
                dataDogLogger = DataDogLogger()
            elif callback == "dynamodb":
                dynamoLogger = DyanmoDBLogger()
            elif callback == "s3":
                s3Logger = S3Logger()
            elif callback == "wandb":
                weightsBiasesLogger = WeightsBiasesLogger()
            elif callback == "logfire":
                logfireLogger = LogfireLogger()
            elif callback == "supabase":
                print_verbose("instantiating supabase")
                supabaseClient = Supabase()
            elif callback == "greenscale":
                greenscaleLogger = GreenscaleLogger()
                print_verbose("Initialized Greenscale Logger")
            elif callable(callback):
                customLogger = CustomLogger()
    except Exception as e:
        raise e


def _init_custom_logger_compatible_class(  # noqa: PLR0915
    logging_integration: _custom_logger_compatible_callbacks_literal,
    internal_usage_cache: Optional[DualCache],
    llm_router: Optional[
        Any
    ],  # expect litellm.Router, but typing errors due to circular import
    custom_logger_init_args: Optional[dict] = {},
) -> Optional[CustomLogger]:
    """
    Initialize a custom logger compatible class
    """
    try:
        custom_logger_init_args = custom_logger_init_args or {}
        if logging_integration == "lago":
            for callback in _in_memory_loggers:
                if isinstance(callback, LagoLogger):
                    return callback  # type: ignore

            lago_logger = LagoLogger()
            _in_memory_loggers.append(lago_logger)
            return lago_logger  # type: ignore
        elif logging_integration == "openmeter":
            for callback in _in_memory_loggers:
                if isinstance(callback, OpenMeterLogger):
                    return callback  # type: ignore

            _openmeter_logger = OpenMeterLogger()
            _in_memory_loggers.append(_openmeter_logger)
            return _openmeter_logger  # type: ignore
        elif logging_integration == "braintrust":
            for callback in _in_memory_loggers:
                if isinstance(callback, BraintrustLogger):
                    return callback  # type: ignore

            braintrust_logger = BraintrustLogger()
            _in_memory_loggers.append(braintrust_logger)
            return braintrust_logger  # type: ignore
        elif logging_integration == "langsmith":
            for callback in _in_memory_loggers:
                if isinstance(callback, LangsmithLogger):
                    return callback  # type: ignore

            _langsmith_logger = LangsmithLogger()
            _in_memory_loggers.append(_langsmith_logger)
            return _langsmith_logger  # type: ignore
        elif logging_integration == "argilla":
            for callback in _in_memory_loggers:
                if isinstance(callback, ArgillaLogger):
                    return callback  # type: ignore

            _argilla_logger = ArgillaLogger()
            _in_memory_loggers.append(_argilla_logger)
            return _argilla_logger  # type: ignore
        elif logging_integration == "literalai":
            for callback in _in_memory_loggers:
                if isinstance(callback, LiteralAILogger):
                    return callback  # type: ignore

            _literalai_logger = LiteralAILogger()
            _in_memory_loggers.append(_literalai_logger)
            return _literalai_logger  # type: ignore
        elif logging_integration == "prometheus":
            for callback in _in_memory_loggers:
                if isinstance(callback, PrometheusLogger):
                    return callback  # type: ignore

            _prometheus_logger = PrometheusLogger()
            _in_memory_loggers.append(_prometheus_logger)
            return _prometheus_logger  # type: ignore
        elif logging_integration == "datadog":
            for callback in _in_memory_loggers:
                if isinstance(callback, DataDogLogger):
                    return callback  # type: ignore

            _datadog_logger = DataDogLogger()
            _in_memory_loggers.append(_datadog_logger)
            return _datadog_logger  # type: ignore
        elif logging_integration == "datadog_llm_observability":
            _datadog_llm_obs_logger = DataDogLLMObsLogger()
            _in_memory_loggers.append(_datadog_llm_obs_logger)
            return _datadog_llm_obs_logger  # type: ignore
        elif logging_integration == "gcs_bucket":
            for callback in _in_memory_loggers:
                if isinstance(callback, GCSBucketLogger):
                    return callback  # type: ignore

            _gcs_bucket_logger = GCSBucketLogger()
            _in_memory_loggers.append(_gcs_bucket_logger)
            return _gcs_bucket_logger  # type: ignore
        elif logging_integration == "azure_storage":
            for callback in _in_memory_loggers:
                if isinstance(callback, AzureBlobStorageLogger):
                    return callback  # type: ignore

            _azure_storage_logger = AzureBlobStorageLogger()
            _in_memory_loggers.append(_azure_storage_logger)
            return _azure_storage_logger  # type: ignore
        elif logging_integration == "opik":
            for callback in _in_memory_loggers:
                if isinstance(callback, OpikLogger):
                    return callback  # type: ignore

            _opik_logger = OpikLogger()
            _in_memory_loggers.append(_opik_logger)
            return _opik_logger  # type: ignore
        elif logging_integration == "arize":
            from litellm.integrations.opentelemetry import (
                OpenTelemetry,
                OpenTelemetryConfig,
            )

            arize_config = ArizeLogger.get_arize_config()
            if arize_config.endpoint is None:
                raise ValueError(
                    "No valid endpoint found for Arize, please set 'ARIZE_ENDPOINT' to your GRPC endpoint or 'ARIZE_HTTP_ENDPOINT' to your HTTP endpoint"
                )
            otel_config = OpenTelemetryConfig(
                exporter=arize_config.protocol,
                endpoint=arize_config.endpoint,
            )

            os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
                f"space_key={arize_config.space_key},api_key={arize_config.api_key}"
            )
            for callback in _in_memory_loggers:
                if (
                    isinstance(callback, ArizeLogger)
                    and callback.callback_name == "arize"
                ):
                    return callback  # type: ignore
            _arize_otel_logger = ArizeLogger(config=otel_config, callback_name="arize")
            _in_memory_loggers.append(_arize_otel_logger)
            return _arize_otel_logger  # type: ignore
        elif logging_integration == "arize_phoenix":
            from litellm.integrations.opentelemetry import (
                OpenTelemetry,
                OpenTelemetryConfig,
            )

            arize_phoenix_config = ArizePhoenixLogger.get_arize_phoenix_config()
            otel_config = OpenTelemetryConfig(
                exporter=arize_phoenix_config.protocol,
                endpoint=arize_phoenix_config.endpoint,
            )

            # auth can be disabled on local deployments of arize phoenix
            if arize_phoenix_config.otlp_auth_headers is not None:
                os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
                    arize_phoenix_config.otlp_auth_headers
                )

            for callback in _in_memory_loggers:
                if (
                    isinstance(callback, OpenTelemetry)
                    and callback.callback_name == "arize_phoenix"
                ):
                    return callback  # type: ignore
            _otel_logger = OpenTelemetry(
                config=otel_config, callback_name="arize_phoenix"
            )
            _in_memory_loggers.append(_otel_logger)
            return _otel_logger  # type: ignore
        elif logging_integration == "otel":
            from litellm.integrations.opentelemetry import OpenTelemetry

            for callback in _in_memory_loggers:
                if isinstance(callback, OpenTelemetry):
                    return callback  # type: ignore
            otel_logger = OpenTelemetry(
                **_get_custom_logger_settings_from_proxy_server(
                    callback_name=logging_integration
                )
            )
            _in_memory_loggers.append(otel_logger)
            return otel_logger  # type: ignore

        elif logging_integration == "galileo":
            for callback in _in_memory_loggers:
                if isinstance(callback, GalileoObserve):
                    return callback  # type: ignore

            galileo_logger = GalileoObserve()
            _in_memory_loggers.append(galileo_logger)
            return galileo_logger  # type: ignore
        elif logging_integration == "logfire":
            if "LOGFIRE_TOKEN" not in os.environ:
                raise ValueError("LOGFIRE_TOKEN not found in environment variables")
            from litellm.integrations.opentelemetry import (
                OpenTelemetry,
                OpenTelemetryConfig,
            )

            otel_config = OpenTelemetryConfig(
                exporter="otlp_http",
                endpoint="https://logfire-api.pydantic.dev/v1/traces",
                headers=f"Authorization={os.getenv('LOGFIRE_TOKEN')}",
            )
            for callback in _in_memory_loggers:
                if isinstance(callback, OpenTelemetry):
                    return callback  # type: ignore
            _otel_logger = OpenTelemetry(config=otel_config)
            _in_memory_loggers.append(_otel_logger)
            return _otel_logger  # type: ignore
        elif logging_integration == "dynamic_rate_limiter":
            from litellm.proxy.hooks.dynamic_rate_limiter import (
                _PROXY_DynamicRateLimitHandler,
            )

            for callback in _in_memory_loggers:
                if isinstance(callback, _PROXY_DynamicRateLimitHandler):
                    return callback  # type: ignore

            if internal_usage_cache is None:
                raise Exception(
                    "Internal Error: Cache cannot be empty - internal_usage_cache={}".format(
                        internal_usage_cache
                    )
                )

            dynamic_rate_limiter_obj = _PROXY_DynamicRateLimitHandler(
                internal_usage_cache=internal_usage_cache
            )

            if llm_router is not None and isinstance(llm_router, litellm.Router):
                dynamic_rate_limiter_obj.update_variables(llm_router=llm_router)
            _in_memory_loggers.append(dynamic_rate_limiter_obj)
            return dynamic_rate_limiter_obj  # type: ignore
        elif logging_integration == "langtrace":
            if "LANGTRACE_API_KEY" not in os.environ:
                raise ValueError("LANGTRACE_API_KEY not found in environment variables")

            from litellm.integrations.opentelemetry import (
                OpenTelemetry,
                OpenTelemetryConfig,
            )

            otel_config = OpenTelemetryConfig(
                exporter="otlp_http",
                endpoint="https://langtrace.ai/api/trace",
            )
            os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
                f"api_key={os.getenv('LANGTRACE_API_KEY')}"
            )
            for callback in _in_memory_loggers:
                if (
                    isinstance(callback, OpenTelemetry)
                    and callback.callback_name == "langtrace"
                ):
                    return callback  # type: ignore
            _otel_logger = OpenTelemetry(config=otel_config, callback_name="langtrace")
            _in_memory_loggers.append(_otel_logger)
            return _otel_logger  # type: ignore

        elif logging_integration == "mlflow":
            for callback in _in_memory_loggers:
                if isinstance(callback, MlflowLogger):
                    return callback  # type: ignore

            _mlflow_logger = MlflowLogger()
            _in_memory_loggers.append(_mlflow_logger)
            return _mlflow_logger  # type: ignore
        elif logging_integration == "langfuse":
            for callback in _in_memory_loggers:
                if isinstance(callback, LangfusePromptManagement):
                    return callback

            langfuse_logger = LangfusePromptManagement()
            _in_memory_loggers.append(langfuse_logger)
            return langfuse_logger  # type: ignore
        elif logging_integration == "pagerduty":
            for callback in _in_memory_loggers:
                if isinstance(callback, PagerDutyAlerting):
                    return callback
            pagerduty_logger = PagerDutyAlerting(**custom_logger_init_args)
            _in_memory_loggers.append(pagerduty_logger)
            return pagerduty_logger  # type: ignore
        elif logging_integration == "anthropic_cache_control_hook":
            for callback in _in_memory_loggers:
                if isinstance(callback, AnthropicCacheControlHook):
                    return callback
            anthropic_cache_control_hook = AnthropicCacheControlHook()
            _in_memory_loggers.append(anthropic_cache_control_hook)
            return anthropic_cache_control_hook  # type: ignore
        elif logging_integration == "gcs_pubsub":
            for callback in _in_memory_loggers:
                if isinstance(callback, GcsPubSubLogger):
                    return callback
            _gcs_pubsub_logger = GcsPubSubLogger()
            _in_memory_loggers.append(_gcs_pubsub_logger)
            return _gcs_pubsub_logger  # type: ignore
        elif logging_integration == "humanloop":
            for callback in _in_memory_loggers:
                if isinstance(callback, HumanloopLogger):
                    return callback

            humanloop_logger = HumanloopLogger()
            _in_memory_loggers.append(humanloop_logger)
            return humanloop_logger  # type: ignore
    except Exception as e:
        verbose_logger.exception(
            f"[Non-Blocking Error] Error initializing custom logger: {e}"
        )
        return None


def get_custom_logger_compatible_class(  # noqa: PLR0915
    logging_integration: _custom_logger_compatible_callbacks_literal,
) -> Optional[CustomLogger]:
    try:
        if logging_integration == "lago":
            for callback in _in_memory_loggers:
                if isinstance(callback, LagoLogger):
                    return callback
        elif logging_integration == "openmeter":
            for callback in _in_memory_loggers:
                if isinstance(callback, OpenMeterLogger):
                    return callback
        elif logging_integration == "braintrust":
            for callback in _in_memory_loggers:
                if isinstance(callback, BraintrustLogger):
                    return callback
        elif logging_integration == "galileo":
            for callback in _in_memory_loggers:
                if isinstance(callback, GalileoObserve):
                    return callback
        elif logging_integration == "langsmith":
            for callback in _in_memory_loggers:
                if isinstance(callback, LangsmithLogger):
                    return callback
        elif logging_integration == "argilla":
            for callback in _in_memory_loggers:
                if isinstance(callback, ArgillaLogger):
                    return callback
        elif logging_integration == "literalai":
            for callback in _in_memory_loggers:
                if isinstance(callback, LiteralAILogger):
                    return callback
        elif logging_integration == "prometheus":
            for callback in _in_memory_loggers:
                if isinstance(callback, PrometheusLogger):
                    return callback
        elif logging_integration == "datadog":
            for callback in _in_memory_loggers:
                if isinstance(callback, DataDogLogger):
                    return callback
        elif logging_integration == "datadog_llm_observability":
            for callback in _in_memory_loggers:
                if isinstance(callback, DataDogLLMObsLogger):
                    return callback
        elif logging_integration == "gcs_bucket":
            for callback in _in_memory_loggers:
                if isinstance(callback, GCSBucketLogger):
                    return callback
        elif logging_integration == "azure_storage":
            for callback in _in_memory_loggers:
                if isinstance(callback, AzureBlobStorageLogger):
                    return callback
        elif logging_integration == "opik":
            for callback in _in_memory_loggers:
                if isinstance(callback, OpikLogger):
                    return callback
        elif logging_integration == "langfuse":
            for callback in _in_memory_loggers:
                if isinstance(callback, LangfusePromptManagement):
                    return callback
        elif logging_integration == "otel":
            from litellm.integrations.opentelemetry import OpenTelemetry

            for callback in _in_memory_loggers:
                if isinstance(callback, OpenTelemetry):
                    return callback
        elif logging_integration == "arize":
            if "ARIZE_SPACE_KEY" not in os.environ:
                raise ValueError("ARIZE_SPACE_KEY not found in environment variables")
            if "ARIZE_API_KEY" not in os.environ:
                raise ValueError("ARIZE_API_KEY not found in environment variables")
            for callback in _in_memory_loggers:
                if (
                    isinstance(callback, ArizeLogger)
                    and callback.callback_name == "arize"
                ):
                    return callback
        elif logging_integration == "logfire":
            if "LOGFIRE_TOKEN" not in os.environ:
                raise ValueError("LOGFIRE_TOKEN not found in environment variables")
            from litellm.integrations.opentelemetry import OpenTelemetry

            for callback in _in_memory_loggers:
                if isinstance(callback, OpenTelemetry):
                    return callback  # type: ignore

        elif logging_integration == "dynamic_rate_limiter":
            from litellm.proxy.hooks.dynamic_rate_limiter import (
                _PROXY_DynamicRateLimitHandler,
            )

            for callback in _in_memory_loggers:
                if isinstance(callback, _PROXY_DynamicRateLimitHandler):
                    return callback  # type: ignore

        elif logging_integration == "langtrace":
            from litellm.integrations.opentelemetry import OpenTelemetry

            if "LANGTRACE_API_KEY" not in os.environ:
                raise ValueError("LANGTRACE_API_KEY not found in environment variables")

            for callback in _in_memory_loggers:
                if (
                    isinstance(callback, OpenTelemetry)
                    and callback.callback_name == "langtrace"
                ):
                    return callback

        elif logging_integration == "mlflow":
            for callback in _in_memory_loggers:
                if isinstance(callback, MlflowLogger):
                    return callback
        elif logging_integration == "pagerduty":
            for callback in _in_memory_loggers:
                if isinstance(callback, PagerDutyAlerting):
                    return callback
        elif logging_integration == "anthropic_cache_control_hook":
            for callback in _in_memory_loggers:
                if isinstance(callback, AnthropicCacheControlHook):
                    return callback
        elif logging_integration == "gcs_pubsub":
            for callback in _in_memory_loggers:
                if isinstance(callback, GcsPubSubLogger):
                    return callback

        return None
    except Exception as e:
        verbose_logger.exception(
            f"[Non-Blocking Error] Error getting custom logger: {e}"
        )
        return None


def _get_custom_logger_settings_from_proxy_server(callback_name: str) -> Dict:
    """
    Get the settings for a custom logger from the proxy server config.yaml

    Proxy server config.yaml defines callback_settings as:

    callback_settings:
        otel:
            message_logging: False
    """
    from litellm.proxy.proxy_server import callback_settings

    if callback_settings:
        return dict(callback_settings.get(callback_name, {}))
    return {}


def use_custom_pricing_for_model(litellm_params: Optional[dict]) -> bool:
    """
    Check if the model uses custom pricing

    Returns True if any of `SPECIAL_MODEL_INFO_PARAMS` are present in `litellm_params` or `model_info`
    """
    if litellm_params is None:
        return False

    metadata: dict = litellm_params.get("metadata", {}) or {}
    model_info: dict = metadata.get("model_info", {}) or {}

    for _custom_cost_param in SPECIAL_MODEL_INFO_PARAMS:
        if litellm_params.get(_custom_cost_param, None) is not None:
            return True
        elif model_info.get(_custom_cost_param, None) is not None:
            return True

    return False


def is_valid_sha256_hash(value: str) -> bool:
    # Check if the value is a valid SHA-256 hash (64 hexadecimal characters)
    return bool(re.fullmatch(r"[a-fA-F0-9]{64}", value))


class StandardLoggingPayloadSetup:
    @staticmethod
    def cleanup_timestamps(
        start_time: Union[dt_object, float],
        end_time: Union[dt_object, float],
        completion_start_time: Union[dt_object, float],
    ) -> Tuple[float, float, float]:
        """
        Convert datetime objects to floats

        Args:
            start_time: Union[dt_object, float]
            end_time: Union[dt_object, float]
            completion_start_time: Union[dt_object, float]

        Returns:
            Tuple[float, float, float]: A tuple containing the start time, end time, and completion start time as floats.
        """

        if isinstance(start_time, datetime.datetime):
            start_time_float = start_time.timestamp()
        elif isinstance(start_time, float):
            start_time_float = start_time
        else:
            raise ValueError(
                f"start_time is required, got={start_time} of type {type(start_time)}"
            )

        if isinstance(end_time, datetime.datetime):
            end_time_float = end_time.timestamp()
        elif isinstance(end_time, float):
            end_time_float = end_time
        else:
            raise ValueError(
                f"end_time is required, got={end_time} of type {type(end_time)}"
            )

        if isinstance(completion_start_time, datetime.datetime):
            completion_start_time_float = completion_start_time.timestamp()
        elif isinstance(completion_start_time, float):
            completion_start_time_float = completion_start_time
        else:
            completion_start_time_float = end_time_float

        return start_time_float, end_time_float, completion_start_time_float

    @staticmethod
    def get_standard_logging_metadata(
        metadata: Optional[Dict[str, Any]],
        litellm_params: Optional[dict] = None,
        prompt_integration: Optional[str] = None,
        applied_guardrails: Optional[List[str]] = None,
        mcp_tool_call_metadata: Optional[StandardLoggingMCPToolCall] = None,
        usage_object: Optional[dict] = None,
    ) -> StandardLoggingMetadata:
        """
        Clean and filter the metadata dictionary to include only the specified keys in StandardLoggingMetadata.

        Args:
            metadata (Optional[Dict[str, Any]]): The original metadata dictionary.

        Returns:
            StandardLoggingMetadata: A StandardLoggingMetadata object containing the cleaned metadata.

        Note:
            - If the input metadata is None or not a dictionary, an empty StandardLoggingMetadata object is returned.
            - If 'user_api_key' is present in metadata and is a valid SHA256 hash, it's stored as 'user_api_key_hash'.
        """

        prompt_management_metadata: Optional[
            StandardLoggingPromptManagementMetadata
        ] = None
        if litellm_params is not None:
            prompt_id = cast(Optional[str], litellm_params.get("prompt_id", None))
            prompt_variables = cast(
                Optional[dict], litellm_params.get("prompt_variables", None)
            )

            if prompt_id is not None and prompt_integration is not None:
                prompt_management_metadata = StandardLoggingPromptManagementMetadata(
                    prompt_id=prompt_id,
                    prompt_variables=prompt_variables,
                    prompt_integration=prompt_integration,
                )

        # Initialize with default values
        clean_metadata = StandardLoggingMetadata(
            user_api_key_hash=None,
            user_api_key_alias=None,
            user_api_key_team_id=None,
            user_api_key_org_id=None,
            user_api_key_user_id=None,
            user_api_key_team_alias=None,
            user_api_key_user_email=None,
            spend_logs_metadata=None,
            requester_ip_address=None,
            requester_metadata=None,
            user_api_key_end_user_id=None,
            prompt_management_metadata=prompt_management_metadata,
            applied_guardrails=applied_guardrails,
            mcp_tool_call_metadata=mcp_tool_call_metadata,
            usage_object=usage_object,
        )
        if isinstance(metadata, dict):
            # Filter the metadata dictionary to include only the specified keys
            supported_keys = StandardLoggingMetadata.__annotations__.keys()
            for key in supported_keys:
                if key in metadata:
                    clean_metadata[key] = metadata[key]  # type: ignore

            if metadata.get("user_api_key") is not None:
                if is_valid_sha256_hash(str(metadata.get("user_api_key"))):
                    clean_metadata["user_api_key_hash"] = metadata.get(
                        "user_api_key"
                    )  # this is the hash
            _potential_requester_metadata = metadata.get(
                "metadata", None
            )  # check if user passed metadata in the sdk request - e.g. metadata for langsmith logging - https://docs.litellm.ai/docs/observability/langsmith_integration#set-langsmith-fields
            if (
                clean_metadata["requester_metadata"] is None
                and _potential_requester_metadata is not None
                and isinstance(_potential_requester_metadata, dict)
            ):
                clean_metadata["requester_metadata"] = _potential_requester_metadata
        return clean_metadata

    @staticmethod
    def get_usage_from_response_obj(
        response_obj: Optional[dict], combined_usage_object: Optional[Usage] = None
    ) -> Usage:
        ## BASE CASE ##
        if combined_usage_object is not None:
            return combined_usage_object
        if response_obj is None:
            return Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        usage = response_obj.get("usage", None) or {}
        if usage is None or (
            not isinstance(usage, dict) and not isinstance(usage, Usage)
        ):
            return Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        elif isinstance(usage, Usage):
            return usage
        elif isinstance(usage, dict):
            if ResponseAPILoggingUtils._is_response_api_usage(usage):
                return (
                    ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                        usage
                    )
                )
            return Usage(**usage)

        raise ValueError(f"usage is required, got={usage} of type {type(usage)}")

    @staticmethod
    def get_model_cost_information(
        base_model: Optional[str],
        custom_pricing: Optional[bool],
        custom_llm_provider: Optional[str],
        init_response_obj: Union[Any, BaseModel, dict],
    ) -> StandardLoggingModelInformation:
        model_cost_name = _select_model_name_for_cost_calc(
            model=None,
            completion_response=init_response_obj,  # type: ignore
            base_model=base_model,
            custom_pricing=custom_pricing,
        )
        if model_cost_name is None:
            model_cost_information = StandardLoggingModelInformation(
                model_map_key="", model_map_value=None
            )
        else:
            try:
                _model_cost_information = litellm.get_model_info(
                    model=model_cost_name, custom_llm_provider=custom_llm_provider
                )
                model_cost_information = StandardLoggingModelInformation(
                    model_map_key=model_cost_name,
                    model_map_value=_model_cost_information,
                )
            except Exception:
                verbose_logger.debug(  # keep in debug otherwise it will trigger on every call
                    "Model={} is not mapped in model cost map. Defaulting to None model_cost_information for standard_logging_payload".format(
                        model_cost_name
                    )
                )
                model_cost_information = StandardLoggingModelInformation(
                    model_map_key=model_cost_name, model_map_value=None
                )
        return model_cost_information

    @staticmethod
    def get_final_response_obj(
        response_obj: dict, init_response_obj: Union[Any, BaseModel, dict], kwargs: dict
    ) -> Optional[Union[dict, str, list]]:
        """
        Get final response object after redacting the message input/output from logging
        """
        if response_obj is not None:
            final_response_obj: Optional[Union[dict, str, list]] = response_obj
        elif isinstance(init_response_obj, list) or isinstance(init_response_obj, str):
            final_response_obj = init_response_obj
        else:
            final_response_obj = None

        modified_final_response_obj = redact_message_input_output_from_logging(
            model_call_details=kwargs,
            result=final_response_obj,
        )

        if modified_final_response_obj is not None and isinstance(
            modified_final_response_obj, BaseModel
        ):
            final_response_obj = modified_final_response_obj.model_dump()
        else:
            final_response_obj = modified_final_response_obj

        return final_response_obj

    @staticmethod
    def get_additional_headers(
        additiona_headers: Optional[dict],
    ) -> Optional[StandardLoggingAdditionalHeaders]:
        if additiona_headers is None:
            return None

        additional_logging_headers: StandardLoggingAdditionalHeaders = {}

        for key in StandardLoggingAdditionalHeaders.__annotations__.keys():
            _key = key.lower()
            _key = _key.replace("_", "-")
            if _key in additiona_headers:
                try:
                    additional_logging_headers[key] = int(additiona_headers[_key])  # type: ignore
                except (ValueError, TypeError):
                    verbose_logger.debug(
                        f"Could not convert {additiona_headers[_key]} to int for key {key}."
                    )
        return additional_logging_headers

    @staticmethod
    def get_hidden_params(
        hidden_params: Optional[dict],
    ) -> StandardLoggingHiddenParams:
        clean_hidden_params = StandardLoggingHiddenParams(
            model_id=None,
            cache_key=None,
            api_base=None,
            response_cost=None,
            additional_headers=None,
            litellm_overhead_time_ms=None,
            batch_models=None,
            litellm_model_name=None,
            usage_object=None,
        )
        if hidden_params is not None:
            for key in StandardLoggingHiddenParams.__annotations__.keys():
                if key in hidden_params:
                    if key == "additional_headers":
                        clean_hidden_params["additional_headers"] = (
                            StandardLoggingPayloadSetup.get_additional_headers(
                                hidden_params[key]
                            )
                        )
                    else:
                        clean_hidden_params[key] = hidden_params[key]  # type: ignore
        return clean_hidden_params

    @staticmethod
    def strip_trailing_slash(api_base: Optional[str]) -> Optional[str]:
        if api_base:
            return api_base.rstrip("/")
        return api_base

    @staticmethod
    def get_error_information(
        original_exception: Optional[Exception],
    ) -> StandardLoggingPayloadErrorInformation:
        error_status: str = str(getattr(original_exception, "status_code", ""))
        error_class: str = (
            str(original_exception.__class__.__name__) if original_exception else ""
        )
        _llm_provider_in_exception = getattr(original_exception, "llm_provider", "")

        # Get traceback information (first 100 lines)
        traceback_info = ""
        if original_exception:
            tb = getattr(original_exception, "__traceback__", None)
            if tb:
                import traceback

                tb_lines = traceback.format_tb(tb)
                traceback_info = "".join(tb_lines[:100])  # Limit to first 100 lines

        # Get additional error details
        error_message = str(original_exception)

        return StandardLoggingPayloadErrorInformation(
            error_code=error_status,
            error_class=error_class,
            llm_provider=_llm_provider_in_exception,
            traceback=traceback_info,
            error_message=error_message if original_exception else "",
        )

    @staticmethod
    def get_response_time(
        start_time_float: float,
        end_time_float: float,
        completion_start_time_float: float,
        stream: bool,
    ) -> float:
        """
        Get the response time for the LLM response

        Args:
            start_time_float: float - start time of the LLM call
            end_time_float: float - end time of the LLM call
            completion_start_time_float: float - time to first token of the LLM response (for streaming responses)
            stream: bool - True when a stream response is returned

        Returns:
            float: The response time for the LLM response
        """
        if stream is True:
            return completion_start_time_float - start_time_float
        else:
            return end_time_float - start_time_float


def get_standard_logging_object_payload(
    kwargs: Optional[dict],
    init_response_obj: Union[Any, BaseModel, dict],
    start_time: dt_object,
    end_time: dt_object,
    logging_obj: Logging,
    status: StandardLoggingPayloadStatus,
    error_str: Optional[str] = None,
    original_exception: Optional[Exception] = None,
    standard_built_in_tools_params: Optional[StandardBuiltInToolsParams] = None,
) -> Optional[StandardLoggingPayload]:
    try:
        kwargs = kwargs or {}

        hidden_params: Optional[dict] = None
        if init_response_obj is None:
            response_obj = {}
        elif isinstance(init_response_obj, BaseModel):
            response_obj = init_response_obj.model_dump()
            hidden_params = getattr(init_response_obj, "_hidden_params", None)
        elif isinstance(init_response_obj, dict):
            response_obj = init_response_obj
        else:
            response_obj = {}

        if original_exception is not None and hidden_params is None:
            response_headers = _get_response_headers(original_exception)
            if response_headers is not None:
                hidden_params = dict(
                    StandardLoggingHiddenParams(
                        additional_headers=StandardLoggingPayloadSetup.get_additional_headers(
                            dict(response_headers)
                        ),
                        model_id=None,
                        cache_key=None,
                        api_base=None,
                        response_cost=None,
                        litellm_overhead_time_ms=None,
                        batch_models=None,
                        litellm_model_name=None,
                        usage_object=None,
                    )
                )

        # standardize this function to be used across, s3, dynamoDB, langfuse logging
        litellm_params = kwargs.get("litellm_params", {})
        proxy_server_request = litellm_params.get("proxy_server_request") or {}

        metadata: dict = (
            litellm_params.get("litellm_metadata")
            or litellm_params.get("metadata", None)
            or {}
        )
        completion_start_time = kwargs.get("completion_start_time", end_time)
        call_type = kwargs.get("call_type")
        cache_hit = kwargs.get("cache_hit", False)
        usage = StandardLoggingPayloadSetup.get_usage_from_response_obj(
            response_obj=response_obj,
            combined_usage_object=cast(
                Optional[Usage], kwargs.get("combined_usage_object")
            ),
        )

        id = response_obj.get("id", kwargs.get("litellm_call_id"))

        _model_id = metadata.get("model_info", {}).get("id", "")
        _model_group = metadata.get("model_group", "")

        request_tags = (
            metadata.get("tags", [])
            if isinstance(metadata.get("tags", []), list)
            else []
        )

        # cleanup timestamps
        (
            start_time_float,
            end_time_float,
            completion_start_time_float,
        ) = StandardLoggingPayloadSetup.cleanup_timestamps(
            start_time=start_time,
            end_time=end_time,
            completion_start_time=completion_start_time,
        )
        response_time = StandardLoggingPayloadSetup.get_response_time(
            start_time_float=start_time_float,
            end_time_float=end_time_float,
            completion_start_time_float=completion_start_time_float,
            stream=kwargs.get("stream", False),
        )
        # clean up litellm hidden params
        clean_hidden_params = StandardLoggingPayloadSetup.get_hidden_params(
            hidden_params
        )
        # clean up litellm metadata
        clean_metadata = StandardLoggingPayloadSetup.get_standard_logging_metadata(
            metadata=metadata,
            litellm_params=litellm_params,
            prompt_integration=kwargs.get("prompt_integration", None),
            applied_guardrails=kwargs.get("applied_guardrails", None),
            mcp_tool_call_metadata=kwargs.get("mcp_tool_call_metadata", None),
            usage_object=usage.model_dump(),
        )

        _request_body = proxy_server_request.get("body", {})
        end_user_id = clean_metadata["user_api_key_end_user_id"] or _request_body.get(
            "user", None
        )  # maintain backwards compatibility with old request body check

        saved_cache_cost: float = 0.0
        if cache_hit is True:
            id = f"{id}_cache_hit{time.time()}"  # do not duplicate the request id
            saved_cache_cost = (
                logging_obj._response_cost_calculator(
                    result=init_response_obj, cache_hit=False  # type: ignore
                )
                or 0.0
            )

        ## Get model cost information ##
        base_model = _get_base_model_from_metadata(model_call_details=kwargs)
        custom_pricing = use_custom_pricing_for_model(litellm_params=litellm_params)

        model_cost_information = StandardLoggingPayloadSetup.get_model_cost_information(
            base_model=base_model,
            custom_pricing=custom_pricing,
            custom_llm_provider=kwargs.get("custom_llm_provider"),
            init_response_obj=init_response_obj,
        )
        response_cost: float = kwargs.get("response_cost", 0) or 0.0

        error_information = StandardLoggingPayloadSetup.get_error_information(
            original_exception=original_exception,
        )

        ## get final response object ##
        final_response_obj = StandardLoggingPayloadSetup.get_final_response_obj(
            response_obj=response_obj,
            init_response_obj=init_response_obj,
            kwargs=kwargs,
        )

        stream: Optional[bool] = None
        if (
            kwargs.get("complete_streaming_response") is not None
            or kwargs.get("async_complete_streaming_response") is not None
        ):
            stream = True

        payload: StandardLoggingPayload = StandardLoggingPayload(
            id=str(id),
            trace_id=kwargs.get("litellm_trace_id"),  # type: ignore
            call_type=call_type or "",
            cache_hit=cache_hit,
            stream=stream,
            status=status,
            custom_llm_provider=cast(Optional[str], kwargs.get("custom_llm_provider")),
            saved_cache_cost=saved_cache_cost,
            startTime=start_time_float,
            endTime=end_time_float,
            completionStartTime=completion_start_time_float,
            response_time=response_time,
            model=kwargs.get("model", "") or "",
            metadata=clean_metadata,
            cache_key=clean_hidden_params["cache_key"],
            response_cost=response_cost,
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            request_tags=request_tags,
            end_user=end_user_id or "",
            api_base=StandardLoggingPayloadSetup.strip_trailing_slash(
                litellm_params.get("api_base", "")
            )
            or "",
            model_group=_model_group,
            model_id=_model_id,
            requester_ip_address=clean_metadata.get("requester_ip_address", None),
            messages=kwargs.get("messages"),
            response=final_response_obj,
            model_parameters=ModelParamHelper.get_standard_logging_model_parameters(
                kwargs.get("optional_params", None) or {}
            ),
            hidden_params=clean_hidden_params,
            model_map_information=model_cost_information,
            error_str=error_str,
            error_information=error_information,
            response_cost_failure_debug_info=kwargs.get(
                "response_cost_failure_debug_information"
            ),
            guardrail_information=metadata.get(
                "standard_logging_guardrail_information", None
            ),
            standard_built_in_tools_params=standard_built_in_tools_params,
        )

        emit_standard_logging_payload(payload)
        return payload
    except Exception as e:
        verbose_logger.exception(
            "Error creating standard logging object - {}".format(str(e))
        )
        return None


def emit_standard_logging_payload(payload: StandardLoggingPayload):
    if os.getenv("LITELLM_PRINT_STANDARD_LOGGING_PAYLOAD"):
        verbose_logger.info(json.dumps(payload, indent=4))


def get_standard_logging_metadata(
    metadata: Optional[Dict[str, Any]],
) -> StandardLoggingMetadata:
    """
    Clean and filter the metadata dictionary to include only the specified keys in StandardLoggingMetadata.

    Args:
        metadata (Optional[Dict[str, Any]]): The original metadata dictionary.

    Returns:
        StandardLoggingMetadata: A StandardLoggingMetadata object containing the cleaned metadata.

    Note:
        - If the input metadata is None or not a dictionary, an empty StandardLoggingMetadata object is returned.
        - If 'user_api_key' is present in metadata and is a valid SHA256 hash, it's stored as 'user_api_key_hash'.
    """
    # Initialize with default values
    clean_metadata = StandardLoggingMetadata(
        user_api_key_hash=None,
        user_api_key_alias=None,
        user_api_key_team_id=None,
        user_api_key_org_id=None,
        user_api_key_user_id=None,
        user_api_key_user_email=None,
        user_api_key_team_alias=None,
        spend_logs_metadata=None,
        requester_ip_address=None,
        requester_metadata=None,
        user_api_key_end_user_id=None,
        prompt_management_metadata=None,
        applied_guardrails=None,
        mcp_tool_call_metadata=None,
        usage_object=None,
    )
    if isinstance(metadata, dict):
        # Filter the metadata dictionary to include only the specified keys
        clean_metadata = StandardLoggingMetadata(
            **{  # type: ignore
                key: metadata[key]
                for key in StandardLoggingMetadata.__annotations__.keys()
                if key in metadata
            }
        )

        if metadata.get("user_api_key") is not None:
            if is_valid_sha256_hash(str(metadata.get("user_api_key"))):
                clean_metadata["user_api_key_hash"] = metadata.get(
                    "user_api_key"
                )  # this is the hash
    return clean_metadata


def scrub_sensitive_keys_in_metadata(litellm_params: Optional[dict]):
    if litellm_params is None:
        litellm_params = {}

    metadata = litellm_params.get("metadata", {}) or {}

    ## check user_api_key_metadata for sensitive logging keys
    cleaned_user_api_key_metadata = {}
    if "user_api_key_metadata" in metadata and isinstance(
        metadata["user_api_key_metadata"], dict
    ):
        for k, v in metadata["user_api_key_metadata"].items():
            if k == "logging":  # prevent logging user logging keys
                cleaned_user_api_key_metadata[k] = (
                    "scrubbed_by_litellm_for_sensitive_keys"
                )
            else:
                cleaned_user_api_key_metadata[k] = v

        metadata["user_api_key_metadata"] = cleaned_user_api_key_metadata
        litellm_params["metadata"] = metadata

    return litellm_params


# integration helper function
def modify_integration(integration_name, integration_params):
    global supabaseClient
    if integration_name == "supabase":
        if "table_name" in integration_params:
            Supabase.supabase_table_name = integration_params["table_name"]


@lru_cache(maxsize=16)
def _get_traceback_str_for_error(error_str: str) -> str:
    """
    function wrapped with lru_cache to limit the number of times `traceback.format_exc()` is called
    """
    return traceback.format_exc()


from decimal import Decimal

# used for unit testing
from typing import Any, Dict, List, Optional, Union


def create_dummy_standard_logging_payload() -> StandardLoggingPayload:
    # First create the nested objects with proper typing
    model_info = StandardLoggingModelInformation(
        model_map_key="gpt-3.5-turbo", model_map_value=None
    )

    metadata = StandardLoggingMetadata(  # type: ignore
        user_api_key_hash=str("test_hash"),
        user_api_key_alias=str("test_alias"),
        user_api_key_team_id=str("test_team"),
        user_api_key_user_id=str("test_user"),
        user_api_key_team_alias=str("test_team_alias"),
        user_api_key_org_id=None,
        spend_logs_metadata=None,
        requester_ip_address=str("127.0.0.1"),
        requester_metadata=None,
        user_api_key_end_user_id=str("test_end_user"),
    )

    hidden_params = StandardLoggingHiddenParams(
        model_id=None,
        cache_key=None,
        api_base=None,
        response_cost=None,
        additional_headers=None,
        litellm_overhead_time_ms=None,
        batch_models=None,
        litellm_model_name=None,
        usage_object=None,
    )

    # Convert numeric values to appropriate types
    response_cost = Decimal("0.1")
    start_time = Decimal("1234567890.0")
    end_time = Decimal("1234567891.0")
    completion_start_time = Decimal("1234567890.5")
    saved_cache_cost = Decimal("0.0")

    # Create messages and response with proper typing
    messages: List[Dict[str, str]] = [{"role": "user", "content": "Hello, world!"}]
    response: Dict[str, List[Dict[str, Dict[str, str]]]] = {
        "choices": [{"message": {"content": "Hi there!"}}]
    }

    # Main payload initialization
    return StandardLoggingPayload(  # type: ignore
        id=str("test_id"),
        call_type=str("completion"),
        stream=bool(False),
        response_cost=response_cost,
        response_cost_failure_debug_info=None,
        status=str("success"),
        total_tokens=int(
            DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT
            + DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT
        ),
        prompt_tokens=int(DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT),
        completion_tokens=int(DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT),
        startTime=start_time,
        endTime=end_time,
        completionStartTime=completion_start_time,
        model_map_information=model_info,
        model=str("gpt-3.5-turbo"),
        model_id=str("model-123"),
        model_group=str("openai-gpt"),
        custom_llm_provider=str("openai"),
        api_base=str("https://api.openai.com"),
        metadata=metadata,
        cache_hit=bool(False),
        cache_key=None,
        saved_cache_cost=saved_cache_cost,
        request_tags=[],
        end_user=None,
        requester_ip_address=str("127.0.0.1"),
        messages=messages,
        response=response,
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=hidden_params,
    )
