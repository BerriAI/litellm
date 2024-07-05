# What is this?
## Common Utility file for Logging handler
# Logging function -> log the exact model details + what's being sent | Non-Blocking
import copy
import datetime
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional

import litellm
from litellm import (
    json_logs,
    log_raw_request_response,
    turn_off_message_logging,
    verbose_logger,
)
from litellm.caching import DualCache, InMemoryCache, S3Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.redact_messages import (
    redact_message_input_output_from_logging,
)
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    TextCompletionResponse,
    TranscriptionResponse,
)
from litellm.utils import (
    _get_base_model_from_metadata,
    add_breadcrumb,
    capture_exception,
    customLogger,
    langsmithLogger,
    liteDebuggerClient,
    logfireLogger,
    lunaryLogger,
    print_verbose,
    prometheusLogger,
    prompt_token_calculator,
    promptLayerLogger,
    supabaseClient,
    weightsBiasesLogger,
)

from ..integrations.aispend import AISpendLogger
from ..integrations.athina import AthinaLogger
from ..integrations.berrispend import BerriSpendLogger
from ..integrations.clickhouse import ClickhouseLogger
from ..integrations.custom_logger import CustomLogger
from ..integrations.datadog import DataDogLogger
from ..integrations.dynamodb import DyanmoDBLogger
from ..integrations.greenscale import GreenscaleLogger
from ..integrations.helicone import HeliconeLogger
from ..integrations.lago import LagoLogger
from ..integrations.langfuse import LangFuseLogger
from ..integrations.langsmith import LangsmithLogger
from ..integrations.litedebugger import LiteDebugger
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


in_memory_trace_id_cache = ServiceTraceIDCache()


class Logging:
    global supabaseClient, liteDebuggerClient, promptLayerLogger, weightsBiasesLogger, langsmithLogger, logfireLogger, capture_exception, add_breadcrumb, lunaryLogger, logfireLogger, prometheusLogger, slack_app
    custom_pricing: bool = False
    stream_options = None

    def __init__(
        self,
        model,
        messages,
        stream,
        call_type,
        start_time,
        litellm_call_id,
        function_id,
        dynamic_success_callbacks=None,
        dynamic_failure_callbacks=None,
        dynamic_async_success_callbacks=None,
        langfuse_public_key=None,
        langfuse_secret=None,
        langfuse_host=None,
    ):
        if call_type not in [item.value for item in CallTypes]:
            allowed_values = ", ".join([item.value for item in CallTypes])
            raise ValueError(
                f"Invalid call_type {call_type}. Allowed values: {allowed_values}"
            )
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
        self.streaming_chunks = []  # for generating complete stream response
        self.sync_streaming_chunks = []  # for generating complete stream response
        self.model_call_details = {}
        self.dynamic_input_callbacks = []  # [TODO] callbacks set for just that call
        self.dynamic_failure_callbacks = dynamic_failure_callbacks
        self.dynamic_success_callbacks = (
            dynamic_success_callbacks  # callbacks set for just that call
        )
        self.dynamic_async_success_callbacks = (
            dynamic_async_success_callbacks  # callbacks set for just that call
        )
        ## DYNAMIC LANGFUSE KEYS ##
        self.langfuse_public_key = langfuse_public_key
        self.langfuse_secret = langfuse_secret
        self.langfuse_host = langfuse_host
        ## TIME TO FIRST TOKEN LOGGING ##
        self.completion_start_time: Optional[datetime.datetime] = None

    def update_environment_variables(
        self, model, user, optional_params, litellm_params, **additional_params
    ):
        self.optional_params = optional_params
        self.model = model
        self.user = user
        self.litellm_params = litellm_params
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
            if "key=" in api_base:
                # Find the position of "key=" in the string
                key_index = api_base.find("key=") + 4
                # Mask the last 5 characters after "key="
                masked_api_base = (
                    api_base[:key_index] + "*" * 5 + api_base[key_index + 5 :]
                )
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
                try:
                    # [Non-blocking Extra Debug Information in metadata]
                    _litellm_params = self.model_call_details.get("litellm_params", {})
                    _metadata = _litellm_params.get("metadata", {}) or {}
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
                    verbose_logger.error(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}\n{}".format(
                            str(e), traceback.format_exc()
                        )
                    )
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made
            callbacks = litellm.input_callback + self.dynamic_input_callbacks
            for callback in callbacks:
                try:
                    if callback == "supabase":
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
                        except:
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
                    elif callable(callback):  # custom logger functions
                        customLogger.log_input_event(
                            model=self.model,
                            messages=self.messages,
                            kwargs=self.model_call_details,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                except Exception as e:
                    verbose_logger.error(
                        "litellm.Logging.pre_call(): Exception occured - {}\n{}".format(
                            str(e), traceback.format_exc()
                        )
                    )
                    verbose_logger.debug(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception:
            verbose_logger.error(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}\n{}".format(
                    str(e), traceback.format_exc()
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
                    verbose_logger.debug(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}\n{}".format(
                            str(e), traceback.format_exc()
                        )
                    )
            original_response = redact_message_input_output_from_logging(
                litellm_logging_obj=self, result=original_response
            )
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made

            callbacks = litellm.input_callback + self.dynamic_input_callbacks
            for callback in callbacks:
                try:
                    if callback == "sentry" and add_breadcrumb:
                        verbose_logger.debug("reaches sentry breadcrumbing")
                        try:
                            details_to_log = copy.deepcopy(self.model_call_details)
                        except:
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
                    verbose_logger.error(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while post-call logging with integrations {}\n{}".format(
                            str(e), traceback.format_exc()
                        )
                    )
                    verbose_logger.debug(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            verbose_logger.error(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {}\n{}".format(
                    str(e), traceback.format_exc()
                )
            )

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
            verbose_logger.debug(f"Model={self.model};")
            if (
                result is not None
                and (
                    isinstance(result, ModelResponse)
                    or isinstance(result, EmbeddingResponse)
                    or isinstance(result, ImageResponse)
                    or isinstance(result, TranscriptionResponse)
                    or isinstance(result, TextCompletionResponse)
                )
                and self.stream != True
            ):  # handle streaming separately
                self.model_call_details["response_cost"] = (
                    litellm.response_cost_calculator(
                        response_object=result,
                        model=self.model,
                        cache_hit=self.model_call_details.get("cache_hit", False),
                        custom_llm_provider=self.model_call_details.get(
                            "custom_llm_provider", None
                        ),
                        base_model=_get_base_model_from_metadata(
                            model_call_details=self.model_call_details
                        ),
                        call_type=self.call_type,
                        optional_params=self.optional_params,
                    )
                )
            else:  # streaming chunks + image gen.
                self.model_call_details["response_cost"] = None

            if (
                litellm.max_budget
                and self.stream == False
                and result is not None
                and "content" in result
            ):
                time_diff = (end_time - start_time).total_seconds()
                float_diff = float(time_diff)
                litellm._current_cost += litellm.completion_cost(
                    model=self.model,
                    prompt="",
                    completion=result["content"],
                    total_time=float_diff,
                )

            return start_time, end_time, result
        except Exception as e:
            raise Exception(f"[Non-Blocking] LiteLLM.Success_Call Error: {str(e)}")

    def success_handler(
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
        )
        # print(f"original response in success handler: {self.model_call_details['original_response']}")
        try:
            verbose_logger.debug(f"success callbacks: {litellm.success_callback}")
            ## BUILD COMPLETE STREAMED RESPONSE
            complete_streaming_response = None
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
                        verbose_logger.error(
                            "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while building complete streaming response in success logging {}\n{}".format(
                                str(e), traceback.format_exc()
                            ),
                            log_level="ERROR",
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
                    litellm.response_cost_calculator(
                        response_object=complete_streaming_response,
                        model=self.model,
                        cache_hit=self.model_call_details.get("cache_hit", False),
                        custom_llm_provider=self.model_call_details.get(
                            "custom_llm_provider", None
                        ),
                        base_model=_get_base_model_from_metadata(
                            model_call_details=self.model_call_details
                        ),
                        call_type=self.call_type,
                        optional_params=self.optional_params,
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

            for callback in callbacks:
                try:
                    litellm_params = self.model_call_details.get("litellm_params", {})
                    if litellm_params.get("no-log", False) == True:
                        # proxy cost tracking cal backs should run
                        if not (
                            isinstance(callback, CustomLogger)
                            and "_PROXY_" in callback.__class__.__name__
                        ):
                            print_verbose("no-log request, skipping logging")
                            continue
                    if callback == "lite_debugger":
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
                    if callback == "promptlayer":
                        print_verbose("reaches promptlayer for logging!")
                        promptLayerLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "supabase":
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
                    if callback == "wandb":
                        print_verbose("reaches wandb for logging!")
                        weightsBiasesLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "langsmith":
                        print_verbose("reaches langsmith for logging!")
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                print_verbose(
                                    "reaches langsmith for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]
                        langsmithLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "logfire":
                        global logfireLogger
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
                            level=LogfireLevel.INFO.value,
                        )

                    if callback == "lunary":
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
                    if callback == "helicone":
                        print_verbose("reaches helicone for logging!")
                        model = self.model
                        messages = self.model_call_details["input"]
                        heliconeLogger.log_success(
                            model=model,
                            messages=messages,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
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
                        if langFuseLogger is None or (
                            (
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                            or (
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                            or (
                                self.langfuse_host is not None
                                and self.langfuse_host != langFuseLogger.langfuse_host
                            )
                        ):
                            langFuseLogger = LangFuseLogger(
                                langfuse_public_key=self.langfuse_public_key,
                                langfuse_secret=self.langfuse_secret,
                                langfuse_host=self.langfuse_host,
                            )
                        _response = langFuseLogger.log_event(
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
                    if callback == "datadog":
                        global dataDogLogger
                        verbose_logger.debug("reaches datadog for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"datadog: is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose("reaches datadog for streaming logging!")
                                result = kwargs["complete_streaming_response"]
                        dataDogLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "prometheus":
                        verbose_logger.debug("reaches prometheus for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"prometheus: is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose(
                                    "reaches prometheus for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]
                        prometheusLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
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
                            genericAPILogger = GenericAPILogger()
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
                    if callback == "greenscale":
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
                    if callback == "athina":
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
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        == False
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
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        == False
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
                        callable(callback) == True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        == False
                    ):  # custom logger functions
                        print_verbose(
                            f"success callbacks: Running Custom Callback Function"
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
            verbose_logger.error(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging {}\n{}".format(
                    str(e), traceback.format_exc()
                ),
            )

    async def async_success_handler(
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        """
        Implementing async callbacks, to handle asyncio event loop issues when custom integrations need to use async functions.
        """
        print_verbose("Logging Details LiteLLM-Async Success Call")
        start_time, end_time, result = self._success_handler_helper_fn(
            start_time=start_time, end_time=end_time, result=result, cache_hit=cache_hit
        )
        ## BUILD COMPLETE STREAMED RESPONSE
        complete_streaming_response = None
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
                    print_verbose(
                        "Error occurred building stream chunk in success logging: {}\n{}".format(
                            str(e), traceback.format_exc()
                        ),
                        log_level="ERROR",
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
                    base_model = _get_base_model_from_metadata(
                        model_call_details=self.model_call_details
                    )
                    # base_model defaults to None if not set on model_info
                    self.model_call_details["response_cost"] = litellm.completion_cost(
                        completion_response=complete_streaming_response,
                        model=base_model,
                    )
                verbose_logger.debug(
                    f"Model={self.model}; cost={self.model_call_details['response_cost']}"
                )
            except litellm.NotFoundError as e:
                verbose_logger.error(
                    f"Model={self.model} not found in completion cost map. Setting 'response_cost' to None"
                )
                self.model_call_details["response_cost"] = None

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

        for callback in callbacks:
            # check if callback can run for this request
            litellm_params = self.model_call_details.get("litellm_params", {})
            if litellm_params.get("no-log", False) == True:
                # proxy cost tracking cal backs should run
                if not (
                    isinstance(callback, CustomLogger)
                    and "_PROXY_" in callback.__class__.__name__
                ):
                    print_verbose("no-log request, skipping logging")
                    continue
            try:
                if kwargs.get("no-log", False) == True:
                    print_verbose("no-log request, skipping logging")
                    continue
                if callback == "cache" and litellm.cache is not None:
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
                if callback == "openmeter":
                    global openMeterLogger
                    if self.stream == True:
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
                    if self.stream == True:
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
            except Exception as e:
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
        return start_time, end_time

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
                    if callback == "lite_debugger":
                        print_verbose("reaches lite_debugger for logging!")
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        result = {
                            "model": self.model,
                            "created": time.time(),
                            "error": traceback_exception,
                            "usage": {
                                "prompt_tokens": prompt_token_calculator(
                                    self.model, messages=self.messages
                                ),
                                "completion_tokens": 0,
                            },
                        }
                        liteDebuggerClient.log_event(
                            model=self.model,
                            messages=self.messages,
                            end_user=self.model_call_details.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=self.litellm_call_id,
                            print_verbose=print_verbose,
                            call_type=self.call_type,
                            stream=self.stream,
                        )
                    if callback == "lunary":
                        print_verbose("reaches lunary for logging error!")

                        model = self.model

                        input = self.model_call_details["input"]

                        _type = (
                            "embed"
                            if self.call_type == CallTypes.embedding.value
                            else "llm"
                        )

                        lunaryLogger.log_event(
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
                    elif callback == "supabase":
                        print_verbose("reaches supabase for logging!")
                        print_verbose(f"supabaseClient: {supabaseClient}")
                        result = {
                            "model": model,
                            "created": time.time(),
                            "error": traceback_exception,
                            "usage": {
                                "prompt_tokens": prompt_token_calculator(
                                    model, messages=self.messages
                                ),
                                "completion_tokens": 0,
                            },
                        }
                        supabaseClient.log_event(
                            model=self.model,
                            messages=self.messages,
                            end_user=self.model_call_details.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=self.model_call_details["litellm_call_id"],
                            print_verbose=print_verbose,
                        )
                    if callable(callback):  # custom logger functions
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
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
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
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                            or (
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                            or (
                                self.langfuse_host is not None
                                and self.langfuse_host != langFuseLogger.langfuse_host
                            )
                        ):
                            langFuseLogger = LangFuseLogger(
                                langfuse_public_key=self.langfuse_public_key,
                                langfuse_secret=self.langfuse_secret,
                                langfuse_host=self.langfuse_host,
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
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                            status_message=str(exception),
                            level="ERROR",
                            kwargs=self.model_call_details,
                        )
                    if callback == "prometheus":
                        global prometheusLogger
                        verbose_logger.debug("reaches prometheus for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        kwargs["exception"] = str(exception)
                        prometheusLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )

                    if callback == "logfire":
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
                            level=LogfireLevel.ERROR.value,
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
            verbose_logger.error(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure logging {}\n{}".format(
                    str(e), traceback.format_exc()
                )
            )

    async def async_failure_handler(
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        """
        Implementing async callbacks, to handle asyncio event loop issues when custom integrations need to use async functions.
        """
        start_time, end_time = self._failure_handler_helper_fn(
            exception=exception,
            traceback_exception=traceback_exception,
            start_time=start_time,
            end_time=end_time,
        )
        result = None  # result sent to all loggers, init this to None incase it's not created
        for callback in litellm._async_failure_callback:
            try:
                if isinstance(callback, CustomLogger):  # custom logger class
                    await callback.async_log_failure_event(
                        kwargs=self.model_call_details,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                    )  # type: ignore
                if callable(callback):  # custom logger functions
                    await customLogger.async_log_event(
                        kwargs=self.model_call_details,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                        callback_func=callback,
                    )
            except Exception as e:
                verbose_logger.error(
                    "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success \
                        logging {}\n{}\nCallback={}".format(
                        str(e), traceback.format_exc(), callback
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


def set_callbacks(callback_list, function_id=None):
    """
    Globally sets the callback client
    """
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, traceloopLogger, athinaLogger, heliconeLogger, aispendLogger, berrispendLogger, supabaseClient, liteDebuggerClient, lunaryLogger, promptLayerLogger, langFuseLogger, customLogger, weightsBiasesLogger, langsmithLogger, logfireLogger, dynamoLogger, s3Logger, dataDogLogger, prometheusLogger, greenscaleLogger, openMeterLogger

    try:
        for callback in callback_list:
            print_verbose(f"init callback list: {callback}")
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
                    traces_sample_rate=float(sentry_trace_rate),
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
            elif callback == "prometheus":
                if prometheusLogger is None:
                    prometheusLogger = PrometheusLogger()
            elif callback == "dynamodb":
                dynamoLogger = DyanmoDBLogger()
            elif callback == "s3":
                s3Logger = S3Logger()
            elif callback == "wandb":
                weightsBiasesLogger = WeightsBiasesLogger()
            elif callback == "langsmith":
                langsmithLogger = LangsmithLogger()
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
) -> CustomLogger:
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
