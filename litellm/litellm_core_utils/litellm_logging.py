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
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

import litellm
from litellm import (
    json_logs,
    log_raw_request_response,
    turn_off_message_logging,
    verbose_logger,
)
from litellm.caching import DualCache, InMemoryCache, S3Cache
from litellm.cost_calculator import _select_model_name_for_cost_calc
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.redact_messages import (
    redact_message_input_output_from_custom_logger,
    redact_message_input_output_from_logging,
)
from litellm.proxy._types import CommonProxyErrors
from litellm.rerank_api.types import RerankResponse
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import SPECIAL_MODEL_INFO_PARAMS
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    StandardCallbackDynamicParams,
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelCostFailureDebugInformation,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
    StandardLoggingPayloadStatus,
    StandardPassThroughResponseObject,
    TextCompletionResponse,
    TranscriptionResponse,
)
from litellm.utils import (
    _get_base_model_from_metadata,
    print_verbose,
    prompt_token_calculator,
)

from ..integrations.aispend import AISpendLogger
from ..integrations.athina import AthinaLogger
from ..integrations.berrispend import BerriSpendLogger
from ..integrations.braintrust_logging import BraintrustLogger
from ..integrations.clickhouse import ClickhouseLogger
from ..integrations.datadog.datadog import DataDogLogger
from ..integrations.dynamodb import DyanmoDBLogger
from ..integrations.galileo import GalileoObserve
from ..integrations.gcs_bucket import GCSBucketLogger
from ..integrations.greenscale import GreenscaleLogger
from ..integrations.helicone import HeliconeLogger
from ..integrations.lago import LagoLogger
from ..integrations.langfuse import LangFuseLogger
from ..integrations.langsmith import LangsmithLogger
from ..integrations.litedebugger import LiteDebugger
from ..integrations.literal_ai import LiteralAILogger
from ..integrations.logfire_logger import LogfireLevel, LogfireLogger
from ..integrations.lunary import LunaryLogger
from ..integrations.openmeter import OpenMeterLogger
from ..integrations.prometheus import PrometheusLogger
from ..integrations.prometheus_services import PrometheusServicesLogger
from ..integrations.prompt_layer import PromptLayerLogger
from ..integrations.s3 import S3Logger
from ..integrations.supabase import Supabase
from ..integrations.traceloop import TraceloopLogger
from ..integrations.weights_biases import WeightsBiasesLogger

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
clickHouseLogger = None
greenscaleLogger = None
lunaryLogger = None
aispendLogger = None
berrispendLogger = None
supabaseClient = None
liteDebuggerClient = None
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


import hashlib


class DynamicLoggingCache:
    """
    Prevent memory leaks caused by initializing new logging clients on each request.

    Relevant Issue: https://github.com/BerriAI/litellm/issues/5695
    """

    def __init__(self) -> None:
        self.cache = InMemoryCache()

    def get_cache_key(self, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True)
        cache_key = hashlib.sha256(args_str.encode("utf-8")).hexdigest()
        return cache_key

    def get_cache(self, credentials: dict, service_name: str) -> Optional[Any]:
        key_name = self.get_cache_key(
            args={**credentials, "service_name": service_name}
        )
        response = self.cache.get_cache(key=key_name)
        return response

    def set_cache(self, credentials: dict, service_name: str, logging_obj: Any) -> None:
        key_name = self.get_cache_key(
            args={**credentials, "service_name": service_name}
        )
        self.cache.set_cache(key=key_name, value=logging_obj)
        return None


in_memory_trace_id_cache = ServiceTraceIDCache()
in_memory_dynamic_logger_cache = DynamicLoggingCache()


class Logging:
    global supabaseClient, liteDebuggerClient, promptLayerLogger, weightsBiasesLogger, logfireLogger, capture_exception, add_breadcrumb, lunaryLogger, logfireLogger, prometheusLogger, slack_app
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
        kwargs: Optional[Dict] = None,
    ):
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
        self.function_id = function_id
        self.streaming_chunks: List[Any] = []  # for generating complete stream response
        self.sync_streaming_chunks: List[Any] = (
            []
        )  # for generating complete stream response
        self.model_call_details: Dict[Any, Any] = {}

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

        ## TIME TO FIRST TOKEN LOGGING ##
        self.completion_start_time: Optional[datetime.datetime] = None

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
        from litellm.secret_managers.main import get_secret_str

        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        if kwargs:
            _supported_callback_params = (
                StandardCallbackDynamicParams.__annotations__.keys()
            )
            for param in _supported_callback_params:
                if param in kwargs:
                    _param_value = kwargs.pop(param)
                    if "os.environ/" in _param_value:
                        _param_value = get_secret_str(secret_name=_param_value)
                    standard_callback_dynamic_params[param] = _param_value  # type: ignore
        return standard_callback_dynamic_params

    def update_environment_variables(
        self, model, user, optional_params, litellm_params, **additional_params
    ):
        self.optional_params = optional_params
        self.model = model
        self.user = user
        self.litellm_params = scrub_sensitive_keys_in_metadata(litellm_params)
        self.logger_fn = litellm_params.get("logger_fn", None)
        verbose_logger.debug(f"self.optional_params: {self.optional_params}")

        self.model_call_details = {
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

    def pre_call(self, input, api_key, model=None, additional_args={}):
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
            headers = additional_args.get("headers", {})
            if headers is None:
                headers = {}
            data = additional_args.get("complete_input_dict", {})
            api_base = str(additional_args.get("api_base", ""))
            query_params = additional_args.get("query_params", {})
            if "key=" in api_base:
                # Find the position of "key=" in the string
                key_index = api_base.find("key=") + 4
                # Mask the last 5 characters after "key="
                masked_api_base = api_base[:key_index] + "*" * 5 + api_base[-4:]
            else:
                masked_api_base = api_base
            self.model_call_details["litellm_params"]["api_base"] = masked_api_base
            masked_headers = {
                k: (
                    (v[:-44] + "*" * 44)
                    if (isinstance(v, str) and len(v) > 44)
                    else "*****"
                )
                for k, v in headers.items()
            }
            formatted_headers = " ".join(
                [f"-H '{k}: {v}'" for k, v in masked_headers.items()]
            )

            verbose_logger.debug(f"PRE-API-CALL ADDITIONAL ARGS: {additional_args}")

            curl_command = "\n\nPOST Request Sent from LiteLLM:\n"
            curl_command += "curl -X POST \\\n"
            curl_command += f"{api_base} \\\n"
            curl_command += (
                f"{formatted_headers} \\\n" if formatted_headers.strip() != "" else ""
            )
            curl_command += f"-d '{str(data)}'\n"
            if additional_args.get("request_str", None) is not None:
                # print the sagemaker / bedrock client request
                curl_command = "\nRequest Sent from LiteLLM:\n"
                curl_command += additional_args.get("request_str", None)
            elif api_base == "":
                curl_command = self.model_call_details

            if json_logs:
                verbose_logger.debug(
                    "POST Request Sent from LiteLLM",
                    extra={"api_base": {api_base}, **masked_headers},
                )
            else:
                print_verbose(f"\033[92m{curl_command}\033[0m\n", log_level="DEBUG")
            # log raw request to provider (like LangFuse) -- if opted in.
            if log_raw_request_response is True:
                _litellm_params = self.model_call_details.get("litellm_params", {})
                _metadata = _litellm_params.get("metadata", {}) or {}
                try:
                    # [Non-blocking Extra Debug Information in metadata]
                    if (
                        turn_off_message_logging is not None
                        and turn_off_message_logging is True
                    ):
                        _metadata["raw_request"] = (
                            "redacted by litellm. \
                            'litellm.turn_off_message_logging=True'"
                        )
                    else:
                        _metadata["raw_request"] = str(curl_command)
                except Exception as e:
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
                litellm_logging_obj=self, result=original_response
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

    def _response_cost_calculator(
        self,
        result: Union[
            ModelResponse,
            EmbeddingResponse,
            ImageResponse,
            TranscriptionResponse,
            TextCompletionResponse,
            HttpxBinaryResponseContent,
            RerankResponse,
        ],
        cache_hit: Optional[bool] = None,
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

        if cache_hit is None:
            cache_hit = self.model_call_details.get("cache_hit", False)

        try:
            response_cost_calculator_kwargs = {
                "response_object": result,
                "model": self.model,
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
            }
        except Exception as e:  # error creating kwargs for cost calculation
            self.model_call_details["response_cost_failure_debug_information"] = (
                StandardLoggingModelCostFailureDebugInformation(
                    error_str=str(e),
                    traceback_str=traceback.format_exc(),
                )
            )
            return None

        try:
            response_cost = litellm.response_cost_calculator(
                **response_cost_calculator_kwargs
            )

            return response_cost
        except Exception as e:  # error calculating cost
            self.model_call_details["response_cost_failure_debug_information"] = (
                StandardLoggingModelCostFailureDebugInformation(
                    error_str=str(e),
                    traceback_str=traceback.format_exc(),
                    model=response_cost_calculator_kwargs["model"],
                    cache_hit=response_cost_calculator_kwargs["cache_hit"],
                    custom_llm_provider=response_cost_calculator_kwargs[
                        "custom_llm_provider"
                    ],
                    base_model=response_cost_calculator_kwargs["base_model"],
                    call_type=response_cost_calculator_kwargs["call_type"],
                    custom_pricing=response_cost_calculator_kwargs["custom_pricing"],
                )
            )

        return None

    def _success_handler_helper_fn(
        self, result=None, start_time=None, end_time=None, cache_hit=None
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
            ## if model in model cost map - log the response cost
            ## else set cost to None
            if (
                result is not None and self.stream is not True
            ):  # handle streaming separately
                if (
                    isinstance(result, ModelResponse)
                    or isinstance(result, EmbeddingResponse)
                    or isinstance(result, ImageResponse)
                    or isinstance(result, TranscriptionResponse)
                    or isinstance(result, TextCompletionResponse)
                    or isinstance(result, HttpxBinaryResponseContent)  # tts
                    or isinstance(result, RerankResponse)
                ):
                    ## RESPONSE COST ##
                    self.model_call_details["response_cost"] = (
                        self._response_cost_calculator(result=result)
                    )

                    ## HIDDEN PARAMS ##
                    if hasattr(result, "_hidden_params"):
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

                            self.model_call_details["litellm_params"]["metadata"][
                                "hidden_params"
                            ] = getattr(result, "_hidden_params", {})
                    ## STANDARDIZED LOGGING PAYLOAD

                    self.model_call_details["standard_logging_object"] = (
                        get_standard_logging_object_payload(
                            kwargs=self.model_call_details,
                            init_response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            logging_obj=self,
                            status="success",
                        )
                    )
                elif isinstance(result, dict):  # pass-through endpoints
                    ## STANDARDIZED LOGGING PAYLOAD
                    self.model_call_details["standard_logging_object"] = (
                        get_standard_logging_object_payload(
                            kwargs=self.model_call_details,
                            init_response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            logging_obj=self,
                            status="success",
                        )
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
                )

            return start_time, end_time, result
        except Exception as e:
            raise Exception(f"[Non-Blocking] LiteLLM.Success_Call Error: {str(e)}")

    def success_handler(
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        print_verbose(f"Logging Details LiteLLM-Success Call: Cache_hit={cache_hit}")
        start_time, end_time, result = self._success_handler_helper_fn(
            start_time=start_time,
            end_time=end_time,
            result=result,
            cache_hit=cache_hit,
        )
        # print(f"original response in success handler: {self.model_call_details['original_response']}")
        try:
            verbose_logger.debug(f"success callbacks: {litellm.success_callback}")
            ## BUILD COMPLETE STREAMED RESPONSE
            complete_streaming_response = None
            if "complete_streaming_response" in self.model_call_details:
                return  # break out of this.
            if self.stream and isinstance(result, ModelResponse):
                if (
                    result.choices[0].finish_reason is not None
                ):  # if it's the last chunk
                    self.sync_streaming_chunks.append(result)
                    # print_verbose(f"final set of received chunks: {self.sync_streaming_chunks}")
                    try:
                        complete_streaming_response = litellm.stream_chunk_builder(
                            self.sync_streaming_chunks,
                            messages=self.model_call_details.get("messages", None),
                            start_time=start_time,
                            end_time=end_time,
                        )
                    except Exception as e:
                        verbose_logger.exception(
                            "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while building complete streaming response in success logging {}".format(
                                str(e)
                            )
                        )
                        complete_streaming_response = None
                else:
                    self.sync_streaming_chunks.append(result)

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
                    )
                )
            if self.dynamic_success_callbacks is not None and isinstance(
                self.dynamic_success_callbacks, list
            ):
                callbacks = self.dynamic_success_callbacks
                ## keep the internal functions ##
                for callback in litellm.success_callback:
                    if (
                        isinstance(callback, CustomLogger)
                        and "_PROXY_" in callback.__class__.__name__
                    ):
                        callbacks.append(callback)
            else:
                callbacks = litellm.success_callback

            result = redact_message_input_output_from_logging(
                result=result, litellm_logging_obj=self
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
                    if litellm_params.get("no-log", False) is True:
                        # proxy cost tracking cal backs should run
                        if not (
                            isinstance(callback, CustomLogger)
                            and "_PROXY_" in callback.__class__.__name__
                        ):
                            print_verbose("no-log request, skipping logging")
                            continue
                    if callback == "lite_debugger" and liteDebuggerClient is not None:
                        print_verbose("reaches lite_debugger for logging!")
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        print_verbose(
                            f"liteDebuggerClient details function {self.call_type} and stream set to {self.stream}"
                        )
                        liteDebuggerClient.log_event(
                            end_user=kwargs.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=self.litellm_call_id,
                            print_verbose=print_verbose,
                            call_type=self.call_type,
                            stream=self.stream,
                        )
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

                        temp_langfuse_logger = langFuseLogger
                        if langFuseLogger is None or (
                            (
                                self.standard_callback_dynamic_params.get(
                                    "langfuse_public_key"
                                )
                                is not None
                                and self.standard_callback_dynamic_params.get(
                                    "langfuse_public_key"
                                )
                                != langFuseLogger.public_key
                            )
                            or (
                                self.standard_callback_dynamic_params.get(
                                    "langfuse_secret"
                                )
                                is not None
                                and self.standard_callback_dynamic_params.get(
                                    "langfuse_secret"
                                )
                                != langFuseLogger.secret_key
                            )
                            or (
                                self.standard_callback_dynamic_params.get(
                                    "langfuse_host"
                                )
                                is not None
                                and self.standard_callback_dynamic_params.get(
                                    "langfuse_host"
                                )
                                != langFuseLogger.langfuse_host
                            )
                        ):
                            credentials = {
                                "langfuse_public_key": self.standard_callback_dynamic_params.get(
                                    "langfuse_public_key"
                                ),
                                "langfuse_secret": self.standard_callback_dynamic_params.get(
                                    "langfuse_secret"
                                ),
                                "langfuse_host": self.standard_callback_dynamic_params.get(
                                    "langfuse_host"
                                ),
                            }
                            temp_langfuse_logger = (
                                in_memory_dynamic_logger_cache.get_cache(
                                    credentials=credentials, service_name="langfuse"
                                )
                            )
                            if temp_langfuse_logger is None:
                                temp_langfuse_logger = LangFuseLogger(
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
                                in_memory_dynamic_logger_cache.set_cache(
                                    credentials=credentials,
                                    service_name="langfuse",
                                    logging_obj=temp_langfuse_logger,
                                )
                        if temp_langfuse_logger is not None:
                            _response = temp_langfuse_logger.log_event(
                                kwargs=kwargs,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                                user_id=kwargs.get("user", None),
                                print_verbose=print_verbose,
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
                    if callback == "clickhouse":
                        global clickHouseLogger
                        verbose_logger.debug("reaches clickhouse for success logging!")
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
                                    "reaches clickhouse for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]
                        if clickHouseLogger is None:
                            clickHouseLogger = ClickhouseLogger()
                        clickHouseLogger.log_event(
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
                    if callback == "cache" and litellm.cache is not None:
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        print_verbose("success_callback: reaches cache for logging!")
                        kwargs = self.model_call_details
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                print_verbose(
                                    f"success_callback: reaches cache for logging, there is no complete_streaming_response. Kwargs={kwargs}\n\n"
                                )
                                pass
                            else:
                                print_verbose(
                                    "success_callback: reaches cache for logging, there is a complete_streaming_response. Adding to cache"
                                )
                                result = kwargs["complete_streaming_response"]
                                # only add to cache once we have a complete streaming response
                                litellm.cache.add_cache(result, **kwargs)
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
                            "success callbacks: Running Custom Callback Function"
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

    async def async_success_handler(
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        """
        Implementing async callbacks, to handle asyncio event loop issues when custom integrations need to use async functions.
        """
        print_verbose(
            "Logging Details LiteLLM-Async Success Call, cache_hit={}".format(cache_hit)
        )
        start_time, end_time, result = self._success_handler_helper_fn(
            start_time=start_time, end_time=end_time, result=result, cache_hit=cache_hit
        )
        ## BUILD COMPLETE STREAMED RESPONSE
        complete_streaming_response = None
        if "async_complete_streaming_response" in self.model_call_details:
            return  # break out of this.
        if self.stream:
            if result.choices[0].finish_reason is not None:  # if it's the last chunk
                self.streaming_chunks.append(result)
                # verbose_logger.debug(f"final set of received chunks: {self.streaming_chunks}")
                try:
                    complete_streaming_response = litellm.stream_chunk_builder(
                        self.streaming_chunks,
                        messages=self.model_call_details.get("messages", None),
                        start_time=start_time,
                        end_time=end_time,
                    )
                except Exception as e:
                    verbose_logger.exception(
                        "Error occurred building stream chunk in success logging: {}".format(
                            str(e)
                        )
                    )
                    complete_streaming_response = None
            else:
                self.streaming_chunks.append(result)
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
                )
            )
        if self.dynamic_async_success_callbacks is not None and isinstance(
            self.dynamic_async_success_callbacks, list
        ):
            callbacks = self.dynamic_async_success_callbacks
            ## keep the internal functions ##
            for callback in litellm._async_success_callback:
                callback_name = ""
                if isinstance(callback, CustomLogger):
                    callback_name = callback.__class__.__name__
                if callable(callback):
                    callback_name = callback.__name__
                if "_PROXY_" in callback_name:
                    callbacks.append(callback)
        else:
            callbacks = litellm._async_success_callback

        result = redact_message_input_output_from_logging(
            result=result, litellm_logging_obj=self
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
            if litellm_params.get("no-log", False) is True:
                # proxy cost tracking cal backs should run
                if not (
                    isinstance(callback, CustomLogger)
                    and "_PROXY_" in callback.__class__.__name__
                ):
                    print_verbose("no-log request, skipping logging")
                    continue
            try:
                if kwargs.get("no-log", False) is True:
                    print_verbose("no-log request, skipping logging")
                    continue
                if (
                    callback == "cache"
                    and litellm.cache is not None
                    and self.model_call_details.get("litellm_params", {}).get(
                        "acompletion", False
                    )
                    is True
                ):
                    # set_cache once complete streaming response is built
                    print_verbose("async success_callback: reaches cache for logging!")
                    kwargs = self.model_call_details
                    if self.stream:
                        if "async_complete_streaming_response" not in kwargs:
                            print_verbose(
                                f"async success_callback: reaches cache for logging, there is no async_complete_streaming_response. Kwargs={kwargs}\n\n"
                            )
                            pass
                        else:
                            print_verbose(
                                "async success_callback: reaches cache for logging, there is a async_complete_streaming_response. Adding to cache"
                            )
                            result = kwargs["async_complete_streaming_response"]
                            # only add to cache once we have a complete streaming response
                            if litellm.cache is not None and not isinstance(
                                litellm.cache.cache, S3Cache
                            ):
                                await litellm.cache.async_add_cache(result, **kwargs)
                            else:
                                litellm.cache.add_cache(result, **kwargs)
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

    def failure_handler(
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
            callbacks = []  # init this to empty incase it's not created

            if self.dynamic_failure_callbacks is not None and isinstance(
                self.dynamic_failure_callbacks, list
            ):
                callbacks = self.dynamic_failure_callbacks
                ## keep the internal functions ##
                for callback in litellm.failure_callback:
                    if (
                        isinstance(callback, CustomLogger)
                        and "_PROXY_" in callback.__class__.__name__
                    ):
                        callbacks.append(callback)
            else:
                callbacks = litellm.failure_callback

            result = None  # result sent to all loggers, init this to None incase it's not created

            result = redact_message_input_output_from_logging(
                result=result, litellm_logging_obj=self
            )
            for callback in callbacks:
                try:
                    if callback == "lite_debugger" and liteDebuggerClient is not None:
                        pass
                    elif callback == "lunary" and lunaryLogger is not None:
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
                        if langFuseLogger is None or (
                            (
                                self.standard_callback_dynamic_params.get(
                                    "langfuse_public_key"
                                )
                                is not None
                                and self.standard_callback_dynamic_params.get(
                                    "langfuse_public_key"
                                )
                                != langFuseLogger.public_key
                            )
                            or (
                                self.standard_callback_dynamic_params.get(
                                    "langfuse_public_key"
                                )
                                is not None
                                and self.standard_callback_dynamic_params.get(
                                    "langfuse_public_key"
                                )
                                != langFuseLogger.public_key
                            )
                            or (
                                self.standard_callback_dynamic_params.get(
                                    "langfuse_host"
                                )
                                is not None
                                and self.standard_callback_dynamic_params.get(
                                    "langfuse_host"
                                )
                                != langFuseLogger.langfuse_host
                            )
                        ):
                            langFuseLogger = LangFuseLogger(
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
                        _response = langFuseLogger.log_event(
                            start_time=start_time,
                            end_time=end_time,
                            response_obj=None,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
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

        callbacks = []  # init this to empty incase it's not created

        if self.dynamic_async_failure_callbacks is not None and isinstance(
            self.dynamic_async_failure_callbacks, list
        ):
            callbacks = self.dynamic_async_failure_callbacks
            ## keep the internal functions ##
            for callback in litellm._async_failure_callback:
                if (
                    isinstance(callback, CustomLogger)
                    and "_PROXY_" in callback.__class__.__name__
                ):
                    callbacks.append(callback)
        else:
            callbacks = litellm._async_failure_callback

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
                    "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success \
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


def set_callbacks(callback_list, function_id=None):
    """
    Globally sets the callback client
    """
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, traceloopLogger, athinaLogger, heliconeLogger, aispendLogger, berrispendLogger, supabaseClient, liteDebuggerClient, lunaryLogger, promptLayerLogger, langFuseLogger, customLogger, weightsBiasesLogger, logfireLogger, dynamoLogger, s3Logger, dataDogLogger, prometheusLogger, greenscaleLogger, openMeterLogger

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
            elif callback == "aispend":
                aispendLogger = AISpendLogger()
            elif callback == "berrispend":
                berrispendLogger = BerriSpendLogger()
            elif callback == "supabase":
                print_verbose("instantiating supabase")
                supabaseClient = Supabase()
            elif callback == "greenscale":
                greenscaleLogger = GreenscaleLogger()
                print_verbose("Initialized Greenscale Logger")
            elif callback == "lite_debugger":
                print_verbose("instantiating lite_debugger")
                if function_id:
                    liteDebuggerClient = LiteDebugger(email=function_id)
                elif litellm.token:
                    liteDebuggerClient = LiteDebugger(email=litellm.token)
                elif litellm.email:
                    liteDebuggerClient = LiteDebugger(email=litellm.email)
                else:
                    liteDebuggerClient = LiteDebugger(email=str(uuid.uuid4()))
            elif callable(callback):
                customLogger = CustomLogger()
    except Exception as e:
        raise e


def _init_custom_logger_compatible_class(
    logging_integration: litellm._custom_logger_compatible_callbacks_literal,
    internal_usage_cache: Optional[DualCache],
    llm_router: Optional[
        Any
    ],  # expect litellm.Router, but typing errors due to circular import
    premium_user: Optional[bool] = None,
) -> Optional[CustomLogger]:
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

        if premium_user:
            _prometheus_logger = PrometheusLogger()
            _in_memory_loggers.append(_prometheus_logger)
            return _prometheus_logger  # type: ignore
        elif premium_user is False:
            verbose_logger.warning(
                f"🚨🚨🚨 Prometheus Metrics is on LiteLLM Enterprise\n🚨 {CommonProxyErrors.not_premium_user.value}"
            )
            return None
        else:
            return None
    elif logging_integration == "datadog":
        for callback in _in_memory_loggers:
            if isinstance(callback, DataDogLogger):
                return callback  # type: ignore

        _datadog_logger = DataDogLogger()
        _in_memory_loggers.append(_datadog_logger)
        return _datadog_logger  # type: ignore
    elif logging_integration == "gcs_bucket":
        for callback in _in_memory_loggers:
            if isinstance(callback, GCSBucketLogger):
                return callback  # type: ignore

        _gcs_bucket_logger = GCSBucketLogger()
        _in_memory_loggers.append(_gcs_bucket_logger)
        return _gcs_bucket_logger  # type: ignore
    elif logging_integration == "arize":
        if "ARIZE_SPACE_KEY" not in os.environ:
            raise ValueError("ARIZE_SPACE_KEY not found in environment variables")
        if "ARIZE_API_KEY" not in os.environ:
            raise ValueError("ARIZE_API_KEY not found in environment variables")
        from litellm.integrations.opentelemetry import (
            OpenTelemetry,
            OpenTelemetryConfig,
        )

        arize_endpoint = (
            os.environ.get("ARIZE_ENDPOINT", None) or "https://otlp.arize.com/v1"
        )
        otel_config = OpenTelemetryConfig(
            exporter="otlp_grpc",
            endpoint=arize_endpoint,
        )
        os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
            f"space_key={os.getenv('ARIZE_SPACE_KEY')},api_key={os.getenv('ARIZE_API_KEY')}"
        )
        for callback in _in_memory_loggers:
            if (
                isinstance(callback, OpenTelemetry)
                and callback.callback_name == "arize"
            ):
                return callback  # type: ignore
        _otel_logger = OpenTelemetry(config=otel_config, callback_name="arize")
        _in_memory_loggers.append(_otel_logger)
        return _otel_logger  # type: ignore

    elif logging_integration == "otel":
        from litellm.integrations.opentelemetry import OpenTelemetry

        for callback in _in_memory_loggers:
            if isinstance(callback, OpenTelemetry):
                return callback  # type: ignore

        otel_logger = OpenTelemetry()
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


def get_custom_logger_compatible_class(
    logging_integration: litellm._custom_logger_compatible_callbacks_literal,
) -> Optional[CustomLogger]:
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
    elif logging_integration == "gcs_bucket":
        for callback in _in_memory_loggers:
            if isinstance(callback, GCSBucketLogger):
                return callback
    elif logging_integration == "otel":
        from litellm.integrations.opentelemetry import OpenTelemetry

        for callback in _in_memory_loggers:
            if isinstance(callback, OpenTelemetry):
                return callback
    elif logging_integration == "arize":
        from litellm.integrations.opentelemetry import OpenTelemetry

        if "ARIZE_SPACE_KEY" not in os.environ:
            raise ValueError("ARIZE_SPACE_KEY not found in environment variables")
        if "ARIZE_API_KEY" not in os.environ:
            raise ValueError("ARIZE_API_KEY not found in environment variables")
        for callback in _in_memory_loggers:
            if (
                isinstance(callback, OpenTelemetry)
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
    return None


def use_custom_pricing_for_model(litellm_params: Optional[dict]) -> bool:
    if litellm_params is None:
        return False
    for k, v in litellm_params.items():
        if k in SPECIAL_MODEL_INFO_PARAMS and v is not None:
            return True
    metadata: Optional[dict] = litellm_params.get("metadata", {})
    if metadata is None:
        return False
    model_info: Optional[dict] = metadata.get("model_info", {})
    if model_info is not None:
        for k, v in model_info.items():
            if k in SPECIAL_MODEL_INFO_PARAMS:
                return True

    return False


def is_valid_sha256_hash(value: str) -> bool:
    # Check if the value is a valid SHA-256 hash (64 hexadecimal characters)
    return bool(re.fullmatch(r"[a-fA-F0-9]{64}", value))


def get_standard_logging_object_payload(
    kwargs: Optional[dict],
    init_response_obj: Union[Any, BaseModel, dict],
    start_time: dt_object,
    end_time: dt_object,
    logging_obj: Logging,
    status: StandardLoggingPayloadStatus,
    error_str: Optional[str] = None,
) -> Optional[StandardLoggingPayload]:
    try:
        if kwargs is None:
            kwargs = {}

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

        # standardize this function to be used across, s3, dynamoDB, langfuse logging
        litellm_params = kwargs.get("litellm_params", {})
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        metadata = (
            litellm_params.get("metadata", {}) or {}
        )  # if litellm_params['metadata'] == None
        completion_start_time = kwargs.get("completion_start_time", end_time)
        call_type = kwargs.get("call_type")
        cache_hit = kwargs.get("cache_hit", False)
        usage = response_obj.get("usage", None) or {}
        if type(usage) is litellm.Usage:
            usage = dict(usage)
        id = response_obj.get("id", kwargs.get("litellm_call_id"))

        _model_id = metadata.get("model_info", {}).get("id", "")
        _model_group = metadata.get("model_group", "")

        request_tags = (
            metadata.get("tags", [])
            if isinstance(metadata.get("tags", []), list)
            else []
        )

        # cleanup timestamps
        if isinstance(start_time, datetime.datetime):
            start_time_float = start_time.timestamp()
        elif isinstance(start_time, float):
            start_time_float = start_time
        if isinstance(end_time, datetime.datetime):
            end_time_float = end_time.timestamp()
        elif isinstance(end_time, float):
            end_time_float = end_time
        if isinstance(completion_start_time, datetime.datetime):
            completion_start_time_float = completion_start_time.timestamp()
        elif isinstance(completion_start_time, float):
            completion_start_time_float = completion_start_time
        else:
            completion_start_time_float = end_time_float
        # clean up litellm hidden params
        clean_hidden_params = StandardLoggingHiddenParams(
            model_id=None,
            cache_key=None,
            api_base=None,
            response_cost=None,
            additional_headers=None,
        )
        if hidden_params is not None:
            clean_hidden_params = StandardLoggingHiddenParams(
                **{  # type: ignore
                    key: hidden_params[key]
                    for key in StandardLoggingHiddenParams.__annotations__.keys()
                    if key in hidden_params
                }
            )
        # clean up litellm metadata
        clean_metadata = get_standard_logging_metadata(metadata=metadata)

        if litellm.cache is not None:
            cache_key = litellm.cache.get_cache_key(**kwargs)
        else:
            cache_key = None

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
            custom_llm_provider = kwargs.get("custom_llm_provider", None)

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

        response_cost: float = kwargs.get("response_cost", 0) or 0.0

        if response_obj is not None:
            final_response_obj: Optional[Union[dict, str, list]] = response_obj
        elif isinstance(init_response_obj, list) or isinstance(init_response_obj, str):
            final_response_obj = init_response_obj
        else:
            final_response_obj = None

        if litellm.turn_off_message_logging:
            final_response_obj = "redacted-by-litellm"

        payload: StandardLoggingPayload = StandardLoggingPayload(
            id=str(id),
            call_type=call_type or "",
            cache_hit=cache_hit,
            status=status,
            saved_cache_cost=saved_cache_cost,
            startTime=start_time_float,
            endTime=end_time_float,
            completionStartTime=completion_start_time_float,
            model=kwargs.get("model", "") or "",
            metadata=clean_metadata,
            cache_key=cache_key,
            response_cost=response_cost,
            total_tokens=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            request_tags=request_tags,
            end_user=end_user_id or "",
            api_base=litellm_params.get("api_base", ""),
            model_group=_model_group,
            model_id=_model_id,
            requester_ip_address=clean_metadata.get("requester_ip_address", None),
            messages=kwargs.get("messages"),
            response=final_response_obj,
            model_parameters=kwargs.get("optional_params", None),
            hidden_params=clean_hidden_params,
            model_map_information=model_cost_information,
            error_str=error_str,
            response_cost_failure_debug_info=kwargs.get(
                "response_cost_failure_debug_information"
            ),
        )

        return payload
    except Exception as e:
        verbose_logger.exception(
            "Error creating standard logging object - {}".format(str(e))
        )
        return None


def get_standard_logging_metadata(
    metadata: Optional[Dict[str, Any]]
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
        user_api_key_user_id=None,
        user_api_key_team_alias=None,
        spend_logs_metadata=None,
        requester_ip_address=None,
        requester_metadata=None,
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
