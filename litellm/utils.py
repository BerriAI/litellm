import sys
import dotenv, json, traceback, threading
import subprocess, os
import litellm, openai
import random, uuid, requests
import datetime, time
import tiktoken
import uuid
import aiohttp

encoding = tiktoken.get_encoding("cl100k_base")
import importlib.metadata
from .integrations.traceloop import TraceloopLogger
from .integrations.helicone import HeliconeLogger
from .integrations.aispend import AISpendLogger
from .integrations.berrispend import BerriSpendLogger
from .integrations.supabase import Supabase
from .integrations.llmonitor import LLMonitorLogger
from .integrations.prompt_layer import PromptLayerLogger
from .integrations.custom_logger import CustomLogger
from .integrations.langfuse import LangFuseLogger
from .integrations.litedebugger import LiteDebugger
from openai.error import OpenAIError as OriginalError
from openai.openai_object import OpenAIObject
from .exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
    ContextWindowExceededError,
    Timeout,
    APIConnectionError,
    APIError,
    BudgetExceededError
)
from typing import cast, List, Dict, Union, Optional
from .caching import Cache


####### ENVIRONMENT VARIABLES ####################
dotenv.load_dotenv()  # Loading env variables using dotenv
sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
posthog = None
slack_app = None
alerts_channel = None
heliconeLogger = None
promptLayerLogger = None
customLogger = None
langFuseLogger = None
llmonitorLogger = None
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

def _generate_id(): # private helper function
    return 'chatcmpl-' + str(uuid.uuid4())

class Message(OpenAIObject):
    def __init__(self, content="default", role="assistant", logprobs=None, **params):
        super(Message, self).__init__(**params)
        self.content = content
        self.role = role
        self.logprobs = logprobs

class Delta(OpenAIObject):
    def __init__(self, content=None, logprobs=None, role=None, **params):
        super(Delta, self).__init__(**params)
        if content is not None:
            self.content = content
        if role:
            self.role = role


class Choices(OpenAIObject):
    def __init__(self, finish_reason="stop", index=0, message=Message(), **params):
        super(Choices, self).__init__(**params)
        self.finish_reason = finish_reason
        self.index = index
        self.message = message

class StreamingChoices(OpenAIObject):
    def __init__(self, finish_reason=None, index=0, delta: Optional[Delta]=None, **params):
        super(StreamingChoices, self).__init__(**params)
        self.finish_reason = finish_reason
        self.index = index
        if delta:
            self.delta = delta
        else:
            self.delta = Delta()

class ModelResponse(OpenAIObject):
    def __init__(self, id=None, choices=None, created=None, model=None, usage=None, stream=False, **params):
        if stream:
            self.object = "chat.completion.chunk"
            self.choices = [StreamingChoices()]
        else:
            if model in litellm.open_ai_embedding_models:
                self.object = "embedding"
            else:
                self.object = "chat.completion"
            self.choices = self.choices = choices if choices else [Choices()]
        if id is None:
            self.id = _generate_id()
        else:
            self.id = id
        if created is None:
            self.created = int(time.time())
        else:
            self.created = created
        self.model = model
        self.usage = (
            usage
            if usage
            else {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )
        super(ModelResponse, self).__init__(**params)

    def to_dict_recursive(self):
        d = super().to_dict_recursive()
        d["choices"] = [choice.to_dict_recursive() for choice in self.choices]
        return d


############################################################
def print_verbose(print_statement):
    if litellm.set_verbose:
        print(f"LiteLLM: {print_statement}")

####### LOGGING ###################
from enum import Enum

class CallTypes(Enum):
    embedding = 'embedding'
    completion = 'completion'

# Logging function -> log the exact model details + what's being sent | Non-Blocking
class Logging:
    global supabaseClient, liteDebuggerClient

    def __init__(self, model, messages, stream, call_type, start_time, litellm_call_id, function_id):
        if call_type not in [item.value for item in CallTypes]:
            allowed_values = ", ".join([item.value for item in CallTypes])
            raise ValueError(f"Invalid call_type {call_type}. Allowed values: {allowed_values}")
        self.model = model
        self.messages = messages
        self.stream = stream
        self.start_time = start_time # log the call start time
        self.call_type = call_type
        self.litellm_call_id = litellm_call_id
        self.function_id = function_id
    
    def update_environment_variables(self, model, user, optional_params, litellm_params):
        self.optional_params = optional_params
        self.model = model
        self.user = user
        self.litellm_params = litellm_params
        self.logger_fn = litellm_params["logger_fn"]
        print_verbose(f"self.optional_params: {self.optional_params}")
        self.model_call_details = {
            "model": self.model,
            "messages": self.messages,
            "optional_params": self.optional_params,
            "litellm_params": self.litellm_params,
        }

    def pre_call(self, input, api_key, model=None, additional_args={}):
        # Log the exact input to the LLM API
        print_verbose(f"Logging Details Pre-API Call for call id {self.litellm_call_id}")
        try:
            # print_verbose(f"logging pre call for model: {self.model} with call type: {self.call_type}")
            self.model_call_details["input"] = input
            self.model_call_details["api_key"] = api_key
            self.model_call_details["additional_args"] = additional_args

            if (
                model
            ):  # if model name was changes pre-call, overwrite the initial model call name with the new one
                self.model_call_details["model"] = model

            # User Logging -> if you pass in a custom logging function
            print_verbose(f"model call details: {self.model_call_details}")
            if self.logger_fn and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
                    )

            if litellm.max_budget and self.stream:
                start_time = self.start_time
                end_time = self.start_time # no time has passed as the call hasn't been made yet
                time_diff = (end_time - start_time).total_seconds()
                float_diff = float(time_diff)
                litellm._current_cost += litellm.completion_cost(model=self.model, prompt="".join(message["content"] for message in self.messages), completion="", total_time=float_diff)

            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made
            for callback in litellm.input_callback:
                try:
                    if callback == "supabase":
                        print_verbose("reaches supabase for logging!")
                        model = self.model_call_details["model"]
                        messages = self.model_call_details["input"]
                        print(f"supabaseClient: {supabaseClient}")
                        supabaseClient.input_log_event(
                            model=model,
                            messages=messages,
                            end_user=litellm._thread_context.user,
                            litellm_call_id=self.litellm_params["litellm_call_id"],
                            print_verbose=print_verbose,
                        )

                    elif callback == "lite_debugger":
                        print_verbose(f"reaches litedebugger for logging! - model_call_details {self.model_call_details}")
                        model = self.model_call_details["model"]
                        messages = self.model_call_details["input"]
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        liteDebuggerClient.input_log_event(
                            model=model,
                            messages=messages,
                            end_user=litellm._thread_context.user,
                            litellm_call_id=self.litellm_params["litellm_call_id"],
                            litellm_params=self.model_call_details["litellm_params"],
                            optional_params=self.model_call_details["optional_params"],
                            print_verbose=print_verbose,
                            call_type=self.call_type
                        )
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while input logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
            )
            print_verbose(
                f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
            )
            if capture_exception:  # log this error to sentry for debugging
                capture_exception(e)

    def post_call(self, original_response, input=None, api_key=None,  additional_args={}):
        # Log the exact result from the LLM API, for streaming - log the type of response received
        try:
            self.model_call_details["input"] = input
            self.model_call_details["api_key"] = api_key
            self.model_call_details["original_response"] = original_response
            self.model_call_details["additional_args"] = additional_args

            # User Logging -> if you pass in a custom logging function
            print_verbose(f"model call details: {self.model_call_details}")
            print_verbose(
                f"Logging Details Post-API Call: logger_fn - {self.logger_fn} | callable(logger_fn) - {callable(self.logger_fn)}"
            )
            if self.logger_fn and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
                    )
            
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made
            for callback in litellm.input_callback:
                try:
                    if callback == "lite_debugger":
                        print_verbose("reaches litedebugger for post-call logging!")
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        liteDebuggerClient.post_call_log_event(
                            original_response=original_response,
                            litellm_call_id=self.litellm_params["litellm_call_id"],
                            print_verbose=print_verbose,
                            call_type = self.call_type, 
                            stream = self.stream,
                        )
                except:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while post-call logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
            )
            pass

    
    def success_handler(self, result, start_time=None, end_time=None):
        print_verbose(
                f"Logging Details LiteLLM-Success Call"
            )
        try:
            if start_time is None:
                start_time = self.start_time
            if end_time is None:
                end_time = datetime.datetime.now()
            print_verbose(f"success callbacks: {litellm.success_callback}")

            if litellm.max_budget and self.stream:
                time_diff = (end_time - start_time).total_seconds()
                float_diff = float(time_diff)
                litellm._current_cost += litellm.completion_cost(model=self.model, prompt="", completion=result["content"], total_time=float_diff)

            for callback in litellm.success_callback:
                try:
                    if callback == "lite_debugger":
                        print_verbose("reaches lite_debugger for logging!")
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        print_verbose(f"liteDebuggerClient details function {self.call_type} and stream set to {self.stream}")
                        liteDebuggerClient.log_event(
                            end_user=litellm._thread_context.user,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=self.litellm_call_id,
                            print_verbose=print_verbose,
                            call_type = self.call_type, 
                            stream = self.stream,
                        )
                    if callback == "api_manager":
                        print_verbose("reaches api manager for updating model cost")
                        litellm.apiManager.update_cost(completion_obj=result, user=self.user)
                    if callback == "cache":
                        # print("entering logger first time")
                        # print(self.litellm_params["stream_response"])
                        if litellm.cache != None and self.model_call_details.get('optional_params', {}).get('stream', False) == True:
                            litellm_call_id = self.litellm_params["litellm_call_id"]
                            if litellm_call_id in self.litellm_params["stream_response"]:
                                # append for the given call_id
                                if self.litellm_params["stream_response"][litellm_call_id]["choices"][0]["message"]["content"] == "default":
                                    self.litellm_params["stream_response"][litellm_call_id]["choices"][0]["message"]["content"] = result["content"] # handle first try
                                else:
                                    self.litellm_params["stream_response"][litellm_call_id]["choices"][0]["message"]["content"] += result["content"]
                            else: # init a streaming response for this call id
                                new_model_response = ModelResponse(choices=[Choices(message=Message(content="default"))])
                                #print("creating new model response")
                                #print(new_model_response)
                                self.litellm_params["stream_response"][litellm_call_id] = new_model_response
                            #print("adding to cache for", litellm_call_id)                              
                            litellm.cache.add_cache(self.litellm_params["stream_response"][litellm_call_id], **self.model_call_details)

                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging {traceback.format_exc()}"
            )
            pass

    def failure_handler(self, exception, traceback_exception, start_time=None, end_time=None):
        print_verbose(
                f"Logging Details LiteLLM-Failure Call"
            )
        try:
            if start_time is None:
                start_time = self.start_time
            if end_time is None:
                end_time = datetime.datetime.now()

            for callback in litellm.failure_callback:
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
                                end_user=litellm._thread_context.user,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                                litellm_call_id=self.litellm_call_id,
                                print_verbose=print_verbose,
                                call_type = self.call_type, 
                                stream = self.stream,
                            )
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure logging {traceback.format_exc()}"
            )
            pass


def exception_logging(
    additional_args={},
    logger_fn=None,
    exception=None,
):
    try:
        model_call_details = {}
        if exception:
            model_call_details["exception"] = exception
        model_call_details["additional_args"] = additional_args
        # User Logging -> if you pass in a custom logging function or want to use sentry breadcrumbs
        print_verbose(
            f"Logging Details: logger_fn - {logger_fn} | callable(logger_fn) - {callable(logger_fn)}"
        )
        if logger_fn and callable(logger_fn):
            try:
                logger_fn(
                    model_call_details
                )  # Expectation: any logger function passed in by the user should accept a dict object
            except Exception as e:
                print(
                    f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
                )
    except Exception as e:
        print(
            f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
        )
        pass


####### CLIENT ###################
# make it easy to log if completion/embedding runs succeeded or failed + see what happened | Non-Blocking
def client(original_function):
    global liteDebuggerClient, get_all_keys

    def function_setup(
        start_time, *args, **kwargs
    ):  # just run once to check if user wants to send their data anywhere - PostHog/Sentry/Slack/etc.
        try:
            global callback_list, add_breadcrumb, user_logger_fn, Logging
            function_id = kwargs["id"] if "id" in kwargs else None
            if litellm.use_client or ("use_client" in kwargs and kwargs["use_client"] == True): 
                print_verbose(f"litedebugger initialized")
                if "lite_debugger" not in litellm.input_callback:
                    litellm.input_callback.append("lite_debugger")
                if "lite_debugger" not in litellm.success_callback:
                    litellm.success_callback.append("lite_debugger")
                if "lite_debugger" not in litellm.failure_callback:
                    litellm.failure_callback.append("lite_debugger")
            if (
                len(litellm.input_callback) > 0
                or len(litellm.success_callback) > 0
                or len(litellm.failure_callback) > 0
            ) and len(callback_list) == 0:
                callback_list = list(
                    set(
                        litellm.input_callback
                        + litellm.success_callback
                        + litellm.failure_callback
                    )
                )
                set_callbacks(
                    callback_list=callback_list,
                    function_id=function_id
                )
            if add_breadcrumb:
                add_breadcrumb(
                    category="litellm.llm_call",
                    message=f"Positional Args: {args}, Keyword Args: {kwargs}",
                    level="info",
                )
            if "logger_fn" in kwargs:
                user_logger_fn = kwargs["logger_fn"]
            # CRASH REPORTING TELEMETRY
            crash_reporting(*args, **kwargs)
            # INIT LOGGER - for user-specified integrations
            model = args[0] if len(args) > 0 else kwargs["model"]
            call_type = original_function.__name__
            if call_type == CallTypes.completion.value:
                messages = args[1] if len(args) > 1 else kwargs["messages"]
            elif call_type == CallTypes.embedding.value:
                messages = args[1] if len(args) > 1 else kwargs["input"]
            stream = True if "stream" in kwargs and kwargs["stream"] == True else False
            logging_obj = Logging(model=model, messages=messages, stream=stream, litellm_call_id=kwargs["litellm_call_id"], function_id=function_id, call_type=call_type, start_time=start_time)
            return logging_obj
        except Exception as e:  # DO NOT BLOCK running the function because of this
            print_verbose(f"[Non-Blocking] {traceback.format_exc()}; args - {args}; kwargs - {kwargs}")
            print(e)
        pass
    
    def crash_reporting(*args, **kwargs):
        if litellm.telemetry:
            try:
                model = args[0] if len(args) > 0 else kwargs["model"]
                exception = kwargs["exception"] if "exception" in kwargs else None
                custom_llm_provider = (
                    kwargs["custom_llm_provider"]
                    if "custom_llm_provider" in kwargs
                    else None
                )
                safe_crash_reporting(
                    model=model,
                    exception=exception,
                    custom_llm_provider=custom_llm_provider,
                )  # log usage-crash details. Do not log any user details. If you want to turn this off, set `litellm.telemetry=False`.
            except:
                # [Non-Blocking Error]
                pass

    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = None
        litellm_call_id = str(uuid.uuid4())
        kwargs["litellm_call_id"] = litellm_call_id
        try:
            model = args[0] if len(args) > 0 else kwargs["model"]
        except:
            raise ValueError("model param not passed in.")

        try:
            logging_obj = function_setup(start_time, *args, **kwargs)
            kwargs["litellm_logging_obj"] = logging_obj

            # [OPTIONAL] CHECK BUDGET 
            if litellm.max_budget:
                if litellm._current_cost > litellm.max_budget:
                    raise BudgetExceededError(current_cost=litellm._current_cost, max_budget=litellm.max_budget)

            # [OPTIONAL] CHECK CACHE
            # remove this after deprecating litellm.caching
            if (litellm.caching or litellm.caching_with_models) and litellm.cache is None:
                litellm.cache = Cache() 

            if kwargs.get("caching", False): # allow users to control returning cached responses from the completion function
                # checking cache
                if (litellm.cache != None or litellm.caching or litellm.caching_with_models):
                    print_verbose(f"LiteLLM: Checking Cache")
                    cached_result = litellm.cache.get_cache(*args, **kwargs)
                    if cached_result != None:
                        return cached_result

            # MODEL CALL
            result = original_function(*args, **kwargs)
            end_time = datetime.datetime.now()
            if "stream" in kwargs and kwargs["stream"] == True:
                # TODO: Add to cache for streaming
                return result
        

            # [OPTIONAL] ADD TO CACHE
            if litellm.caching or litellm.caching_with_models or litellm.cache != None: # user init a cache object
                litellm.cache.add_cache(result, *args, **kwargs)
            
            # [OPTIONAL] Return LiteLLM call_id
            if litellm.use_client == True:
                result['litellm_call_id'] = litellm_call_id

            # LOG SUCCESS - handle streaming success logging in the _next_ object, remove `handle_success` once it's deprecated
            logging_obj.success_handler(result, start_time, end_time)
            # threading.Thread(target=logging_obj.success_handler, args=(result, start_time, end_time)).start()
            my_thread = threading.Thread(
                target=handle_success, args=(args, kwargs, result, start_time, end_time)
            )  # don't interrupt execution of main thread
            my_thread.start()
            # RETURN RESULT
            return result
        except Exception as e:
            traceback_exception = traceback.format_exc()
            crash_reporting(*args, **kwargs, exception=traceback_exception)
            end_time = datetime.datetime.now()
            # LOG FAILURE - handle streaming failure logging in the _next_ object, remove `handle_failure` once it's deprecated
            threading.Thread(target=logging_obj.failure_handler, args=(e, traceback_exception, start_time, end_time)).start()
            my_thread = threading.Thread(
                target=handle_failure,
                args=(e, traceback_exception, start_time, end_time, args, kwargs),
            )  # don't interrupt execution of main thread
            my_thread.start()
            if hasattr(e, "message"):
                if (
                    liteDebuggerClient and liteDebuggerClient.dashboard_url != None
                ):  # make it easy to get to the debugger logs if you've initialized it
                    e.message += f"\n Check the log in your dashboard - {liteDebuggerClient.dashboard_url}"
            raise e
    return wrapper


####### USAGE CALCULATOR ################


# Extract the number of billion parameters from the model name
# only used for together_computer LLMs
def get_model_params_and_category(model_name):
    import re
    params_match = re.search(r'(\d+b)', model_name) # catch all decimals like 3b, 70b, etc    
    category = None
    if params_match != None:
        params_match = params_match.group(1)
        params_match = params_match.replace("b", "")
        params_billion = float(params_match)
        # Determine the category based on the number of parameters
        if params_billion <= 3.0:
            category = "together-ai-up-to-3b"
        elif params_billion <= 7.0:
            category = "together-ai-3.1b-7b"
        elif params_billion <= 20.0:
            category = "together-ai-7.1b-20b"
        elif params_billion <= 40.0:
            category = "together-ai-20.1b-40b"
        elif params_billion <= 70.0:
            category = "together-ai-40.1b-70b"
        return category

    return None

def get_replicate_completion_pricing(completion_response=None, total_time=0.0):
    # see https://replicate.com/pricing
    a100_40gb_price_per_second_public = 0.001150
    # for all litellm currently supported LLMs, almost all requests go to a100_80gb
    a100_80gb_price_per_second_public = 0.001400 # assume all calls sent to A100 80GB for now
    if total_time == 0.0:
        start_time = completion_response['created']
        end_time = completion_response["ended"]
        total_time = end_time - start_time

    return a100_80gb_price_per_second_public*total_time


def token_counter(model="", text=None, messages = None):
    # Args:
    # text: raw text string passed to model
    # messages: List of Dicts passed to completion, messages = [{"role": "user", "content": "hello"}]
    # use tiktoken or anthropic's tokenizer depending on the model
    if text == None:
        if messages != None:
            text = " ".join([message["content"] for message in messages])
    num_tokens = 0

    if model != None and "claude" in model:
        try:
            import anthropic
        except Exception:
            # if importing anthropic fails
            # don't raise an exception
            num_tokens = len(encoding.encode(text))
            return num_tokens

        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        anthropic = Anthropic()
        num_tokens = anthropic.count_tokens(text)
    else:
        num_tokens = len(encoding.encode(text))
    return num_tokens


def cost_per_token(model="gpt-3.5-turbo", prompt_tokens=0, completion_tokens=0):
    # given
    prompt_tokens_cost_usd_dollar = 0
    completion_tokens_cost_usd_dollar = 0
    model_cost_ref = litellm.model_cost
    if model in model_cost_ref:
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref[model]["output_cost_per_token"] * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    else:
        # calculate average input cost
        input_cost_sum = 0
        output_cost_sum = 0
        model_cost_ref = litellm.model_cost
        for model in model_cost_ref:
            input_cost_sum += model_cost_ref[model]["input_cost_per_token"]
            output_cost_sum += model_cost_ref[model]["output_cost_per_token"]
        avg_input_cost = input_cost_sum / len(model_cost_ref.keys())
        avg_output_cost = output_cost_sum / len(model_cost_ref.keys())
        prompt_tokens_cost_usd_dollar = avg_input_cost * prompt_tokens
        completion_tokens_cost_usd_dollar = avg_output_cost * completion_tokens
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar


def completion_cost(
        completion_response=None,
        model="gpt-3.5-turbo", 
        prompt="", 
        completion="",
        total_time=0.0, # used for replicate
    ):
    try:
        # Handle Inputs to completion_cost
        prompt_tokens = 0
        completion_tokens = 0
        if completion_response != None:
            # get input/output tokens from completion_response
            prompt_tokens = completion_response['usage']['prompt_tokens']
            completion_tokens = completion_response['usage']['completion_tokens']
            model = completion_response['model'] # get model from completion_response
        else:
            prompt_tokens = token_counter(model=model, text=prompt)
            completion_tokens = token_counter(model=model, text=completion)
        
        # Calculate cost based on prompt_tokens, completion_tokens
        if "togethercomputer" in model:
            # together ai prices based on size of llm
            # get_model_params_and_category takes a model name and returns the category of LLM size it is in model_prices_and_context_window.json 
            model = get_model_params_and_category(model)
        # replicate llms are calculate based on time for request running
        # see https://replicate.com/pricing
        elif (
            model in litellm.replicate_models or
            "replicate" in model
        ):
            return get_replicate_completion_pricing(completion_response, total_time)
        prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar
    except:
        return 0.0 # this should not block a users execution path

####### HELPER FUNCTIONS ################
def get_litellm_params(
    return_async=False,
    api_key=None,
    force_timeout=600,
    azure=False,
    logger_fn=None,
    verbose=False,
    hugging_face=False,
    replicate=False,
    together_ai=False,
    custom_llm_provider=None,
    api_base=None,
    litellm_call_id=None,
    model_alias_map=None,
    completion_call_id=None
):
    litellm_params = {
        "return_async": return_async,
        "api_key": api_key,
        "force_timeout": force_timeout,
        "logger_fn": logger_fn,
        "verbose": verbose,
        "custom_llm_provider": custom_llm_provider,
        "api_base": api_base,
        "litellm_call_id": litellm_call_id,
        "model_alias_map": model_alias_map,
        "completion_call_id": completion_call_id,
        "stream_response": {} # litellm_call_id: ModelResponse Dict
    }

    return litellm_params


def get_optional_params(  # use the openai defaults
    # 12 optional params
    functions=[],
    function_call="",
    temperature=1,
    top_p=1,
    n=1,
    stream=False,
    stop=None,
    max_tokens=float("inf"),
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
    num_beams=1,
    remove_input=False, # for nlp_cloud
    user="",
    deployment_id=None,
    model=None,
    custom_llm_provider="",
    top_k=40,
    return_full_text=False,
    task=None
):
    optional_params = {}
    if model in litellm.anthropic_models:
        # handle anthropic params
        if stream:
            optional_params["stream"] = stream
        if stop != None:
            optional_params["stop_sequences"] = stop
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_p != 1:
            optional_params["top_p"] = top_p
        return optional_params
    elif model in litellm.cohere_models:
        # handle cohere params
        if stream:
            optional_params["stream"] = stream
        if temperature != 1:
            optional_params["temperature"] = temperature
        if max_tokens != float("inf"):
            optional_params["max_tokens"] = max_tokens
        if logit_bias != {}:
            optional_params["logit_bias"] = logit_bias
        return optional_params
    elif custom_llm_provider == "replicate":
        if stream:
            optional_params["stream"] = stream
            return optional_params
        if max_tokens != float("inf"):
            if "vicuna" in model or "flan" in model:
                optional_params["max_length"] = max_tokens
            else:
                optional_params["max_new_tokens"] = max_tokens
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_p != 1:
            optional_params["top_p"] = top_p
        if top_k != 40:
            optional_params["top_k"] = top_k
        if stop != None:
            optional_params["stop_sequences"] = stop
    elif custom_llm_provider == "huggingface":
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_p != 1:
            optional_params["top_p"] = top_p
        if n != 1:
            optional_params["n"] = n
        if stream:
            optional_params["stream"] = stream
        if stop != None:
            optional_params["stop"] = stop
        if max_tokens != float("inf"):
            optional_params["max_new_tokens"] = max_tokens
        if presence_penalty != 0:
            optional_params["repetition_penalty"] = presence_penalty
        optional_params["return_full_text"] = return_full_text
        optional_params["details"] = True
        optional_params["task"] = task
    elif custom_llm_provider == "together_ai":
        if stream:
            optional_params["stream_tokens"] = stream
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_p != 1:
            optional_params["top_p"] = top_p
        if top_k != 40:
            optional_params["top_k"] = top_k
        if max_tokens != float("inf"):
            optional_params["max_tokens"] = max_tokens
        if frequency_penalty != 0:
            optional_params["frequency_penalty"] = frequency_penalty # TODO: Check if should be repetition penalty
        if stop != None:
            optional_params["stop"] = stop #TG AI expects a list, example ["\n\n\n\n","&lt;|endoftext|&gt;"]
    elif (
        model in litellm.vertex_chat_models or model in litellm.vertex_code_chat_models
    ):  # chat-bison has diff args from chat-bison@001, ty Google :) 
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_p != 1:
            optional_params["top_p"] = top_p
        if max_tokens != float("inf"):
            optional_params["max_output_tokens"] = max_tokens
    elif model in litellm.vertex_text_models:
        # required params for all text vertex calls
        # temperature=0.2, top_p=0.1, top_k=20
        # always set temperature, top_p, top_k else, text bison fails
        optional_params["temperature"] = temperature
        optional_params["top_p"] = top_p
        optional_params["top_k"] = top_k
        if max_tokens != float("inf"):
            optional_params["max_output_tokens"] = max_tokens
    elif model in model in litellm.vertex_code_text_models:
        optional_params["temperature"] = temperature
        if max_tokens != float("inf"):
            optional_params["max_output_tokens"] = max_tokens
    elif custom_llm_provider == "baseten":
        optional_params["temperature"] = temperature
        optional_params["stream"] = stream
        if top_p != 1:
            optional_params["top_p"] = top_p
        optional_params["top_k"] = top_k
        optional_params["num_beams"] = num_beams
        if max_tokens != float("inf"):
            optional_params["max_new_tokens"] = max_tokens
    elif custom_llm_provider == "sagemaker":
        if "llama-2" in model:
            # llama-2 models on sagemaker support the following args
            """
            max_new_tokens: Model generates text until the output length (excluding the input context length) reaches max_new_tokens. If specified, it must be a positive integer.
            temperature: Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.
            top_p: In each step of text generation, sample from the smallest possible set of words with cumulative probability top_p. If specified, it must be a float between 0 and 1.
            return_full_text: If True, input text will be part of the output generated text. If specified, it must be boolean. The default value for it is False.
            """
            if max_tokens != float("inf"):
                optional_params["max_new_tokens"] = max_tokens
            if temperature != 1:
                optional_params["temperature"] = temperature
            if top_p != 1:
                optional_params["top_p"] = top_p
    elif custom_llm_provider == "bedrock":
        if "ai21" in model or "anthropic" in model:
            # params "maxTokens":200,"temperature":0,"topP":250,"stop_sequences":[],
            # https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=j2-ultra
            if max_tokens != float("inf"):
                optional_params["maxTokens"] = max_tokens
            if temperature != 1:
                optional_params["temperature"] = temperature
            if stop != None:
                optional_params["stop_sequences"] = stop
            if top_p != 1:
                optional_params["topP"] = top_p

        elif "amazon" in model: # amazon titan llms
            # see https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=titan-large
            if max_tokens != float("inf"):
                optional_params["maxTokenCount"] = max_tokens
            if temperature != 1:
                optional_params["temperature"] = temperature
            if stop != None:
                optional_params["stopSequences"] = stop
            if top_p != 1:
                optional_params["topP"] = top_p

    elif model in litellm.aleph_alpha_models:
        if max_tokens != float("inf"):
            optional_params["maximum_tokens"] = max_tokens
        if stream:
            optional_params["stream"] = stream
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_k != 40:
            optional_params["top_k"] = top_k
        if top_p != 1:
            optional_params["top_p"] = top_p
        if presence_penalty != 0:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty != 0:
            optional_params["frequency_penalty"] = frequency_penalty
        if n != 1:
            optional_params["n"] = n
        if stop != None:
            optional_params["stop_sequences"] = stop
    elif model in litellm.nlp_cloud_models or custom_llm_provider == "nlp_cloud":
        if max_tokens != float("inf"):
            optional_params["max_length"] = max_tokens
        if stream:
            optional_params["stream"] = stream
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_k != 40:
            optional_params["top_k"] = top_k
        if top_p != 1:
            optional_params["top_p"] = top_p
        if presence_penalty != 0:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty != 0:
            optional_params["frequency_penalty"] = frequency_penalty
        if num_beams != 1:
            optional_params["num_beams"] = num_beams
        if n != 1:
            optional_params["num_return_sequences"] = n
        if remove_input == True:
            optional_params["remove_input"] = True
        if stop != None:
            optional_params["stop_sequences"] = stop
    else:  # assume passing in params for openai/azure openai
        if functions != []:
            optional_params["functions"] = functions
        if function_call != "":
            optional_params["function_call"] = function_call
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_p != 1:
            optional_params["top_p"] = top_p
        if n != 1:
            optional_params["n"] = n
        if stream:
            optional_params["stream"] = stream
        if stop != None:
            optional_params["stop"] = stop
        if max_tokens != float("inf"):
            optional_params["max_tokens"] = max_tokens
        if presence_penalty != 0:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty != 0:
            optional_params["frequency_penalty"] = frequency_penalty
        if logit_bias != {}:
            optional_params["logit_bias"] = logit_bias
        if user != "":
            optional_params["user"] = user
        if deployment_id != None:
            optional_params["deployment_id"] = deployment_id
        return optional_params
    return optional_params

def get_llm_provider(model: str, custom_llm_provider: Optional[str] = None):
    try:
        # check if llm provider provided
        if custom_llm_provider:
            return model, custom_llm_provider

        # check if llm provider part of model name
        if model.split("/",1)[0] in litellm.provider_list:
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            return model, custom_llm_provider

        # check if model in known model provider list 
        ## openai - chatcompletion + text completion
        if model in litellm.open_ai_chat_completion_models:
            custom_llm_provider = "openai"
        elif model in litellm.open_ai_text_completion_models:
            custom_llm_provider = "text-completion-openai"
        ## anthropic 
        elif model in litellm.anthropic_models:
            custom_llm_provider = "anthropic"
        ## cohere
        elif model in litellm.cohere_models:
            custom_llm_provider = "cohere"
        ## replicate
        elif model in litellm.replicate_models:
            custom_llm_provider = "replicate"
        ## openrouter
        elif model in litellm.openrouter_models:
            custom_llm_provider = "openrouter"
        ## vertex - text + chat models
        elif model in litellm.vertex_chat_models or model in litellm.vertex_text_models:
            custom_llm_provider = "vertex_ai"
        ## huggingface 
        elif model in litellm.huggingface_models:
            custom_llm_provider = "huggingface"
        ## ai21 
        elif model in litellm.ai21_models:
            custom_llm_provider = "ai21"
        ## together_ai 
        elif model in litellm.together_ai_models:
            custom_llm_provider = "together_ai"
        ## aleph_alpha 
        elif model in litellm.aleph_alpha_models:
            custom_llm_provider = "aleph_alpha"
        ## baseten 
        elif model in litellm.baseten_models:
            custom_llm_provider = "baseten"
        ## nlp_cloud
        elif model in litellm.nlp_cloud_models:
            custom_llm_provider = "nlp_cloud"
        
        if custom_llm_provider is None or custom_llm_provider=="":
            raise ValueError(f"LLM Provider NOT provided. Pass in the LLM provider you are trying to call. E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/{model}',..)` Learn more: https://docs.litellm.ai/docs/providers")
        return model, custom_llm_provider
    except Exception as e: 
        raise e

def get_max_tokens(model: str):
    try:
        return litellm.model_cost[model]
    except:
        raise Exception("This model isn't mapped yet. Add it here - https://github.com/BerriAI/litellm/blob/main/cookbook/community-resources/max_tokens.json")
    

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

def validate_environment():
        api_key = None
        if "OPENAI_API_KEY" in os.environ:
            api_key = os.getenv("OPENAI_API_KEY")
        elif "ANTHROPIC_API_KEY" in os.environ:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif "REPLICATE_API_KEY" in os.environ:
            api_key = os.getenv("REPLICATE_API_KEY")
        elif "AZURE_API_KEY" in os.environ:
            api_key = os.getenv("AZURE_API_KEY")
        elif "COHERE_API_KEY" in os.environ:
            api_key = os.getenv("COHERE_API_KEY")
        elif "TOGETHERAI_API_KEY" in os.environ:
            api_key = os.getenv("TOGETHERAI_API_KEY")
        elif "BASETEN_API_KEY" in os.environ:
            api_key = os.getenv("BASETEN_API_KEY")
        elif "AI21_API_KEY" in os.environ:
            api_key = os.getenv("AI21_API_KEY")
        elif "OPENROUTER_API_KEY" in os.environ:
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif "ALEPHALPHA_API_KEY" in os.environ:
            api_key = os.getenv("ALEPHALPHA_API_KEY")
        return api_key

def set_callbacks(callback_list, function_id=None):
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, traceloopLogger, heliconeLogger, aispendLogger, berrispendLogger, supabaseClient, liteDebuggerClient, llmonitorLogger, promptLayerLogger, langFuseLogger, customLogger
    try:
        for callback in callback_list:
            print_verbose(f"callback: {callback}")
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
                    dsn=os.environ.get("SENTRY_API_URL"),
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
            elif callback == "helicone":
                heliconeLogger = HeliconeLogger()
            elif callback == "llmonitor":
                llmonitorLogger = LLMonitorLogger()
            elif callback == "promptlayer":
                promptLayerLogger = PromptLayerLogger()
            elif callback == "langfuse":
                langFuseLogger = LangFuseLogger()
            elif callback == "aispend":
                aispendLogger = AISpendLogger()
            elif callback == "berrispend":
                berrispendLogger = BerriSpendLogger()
            elif callback == "supabase":
                print_verbose(f"instantiating supabase")
                supabaseClient = Supabase()
            elif callback == "lite_debugger":
                print_verbose(f"instantiating lite_debugger")
                if function_id:
                    liteDebuggerClient = LiteDebugger(email=function_id)
                elif litellm.token:
                    liteDebuggerClient = LiteDebugger(email=litellm.token)
                elif litellm.email:
                    liteDebuggerClient = LiteDebugger(email=litellm.email)
                else:
                    liteDebuggerClient = LiteDebugger(email=str(uuid.uuid4()))
            elif callable(callback):
                customLogger = CustomLogger(callback_func=callback)
    except Exception as e:
        raise e


def handle_failure(exception, traceback_exception, start_time, end_time, args, kwargs):
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, aispendLogger, berrispendLogger, supabaseClient, liteDebuggerClient, llmonitorLogger
    try:
        # print_verbose(f"handle_failure args: {args}")
        # print_verbose(f"handle_failure kwargs: {kwargs}")

        success_handler = additional_details.pop("success_handler", None)
        failure_handler = additional_details.pop("failure_handler", None)

        additional_details["Event_Name"] = additional_details.pop(
            "failed_event_name", "litellm.failed_query"
        )
        print_verbose(f"self.failure_callback: {litellm.failure_callback}")
        for callback in litellm.failure_callback:
            try:
                if callback == "slack":
                    slack_msg = ""
                    if len(kwargs) > 0:
                        for key in kwargs:
                            slack_msg += f"{key}: {kwargs[key]}\n"
                    if len(args) > 0:
                        for i, arg in enumerate(args):
                            slack_msg += f"LiteLLM_Args_{str(i)}: {arg}"
                    for detail in additional_details:
                        slack_msg += f"{detail}: {additional_details[detail]}\n"
                    slack_msg += f"Traceback: {traceback_exception}"
                    slack_app.client.chat_postMessage(
                        channel=alerts_channel, text=slack_msg
                    )
                elif callback == "sentry":
                    capture_exception(exception)
                elif callback == "posthog":
                    print_verbose(
                        f"inside posthog, additional_details: {len(additional_details.keys())}"
                    )
                    ph_obj = {}
                    if len(kwargs) > 0:
                        ph_obj = kwargs
                    if len(args) > 0:
                        for i, arg in enumerate(args):
                            ph_obj["litellm_args_" + str(i)] = arg
                    for detail in additional_details:
                        ph_obj[detail] = additional_details[detail]
                    event_name = additional_details["Event_Name"]
                    print_verbose(f"ph_obj: {ph_obj}")
                    print_verbose(f"PostHog Event Name: {event_name}")
                    if "user_id" in additional_details:
                        posthog.capture(
                            additional_details["user_id"], event_name, ph_obj
                        )
                    else:  # PostHog calls require a unique id to identify a user - https://posthog.com/docs/libraries/python
                        unique_id = str(uuid.uuid4())
                        posthog.capture(unique_id, event_name)
                        print_verbose(f"successfully logged to PostHog!")
                elif callback == "berrispend":
                    print_verbose("reaches berrispend for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = args[1] if len(args) > 1 else kwargs["messages"]
                    result = {
                        "model": model,
                        "created": time.time(),
                        "error": traceback_exception,
                        "usage": {
                            "prompt_tokens": prompt_token_calculator(
                                model, messages=messages
                            ),
                            "completion_tokens": 0,
                        },
                    }
                    berrispendLogger.log_event(
                        model=model,
                        messages=messages,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "aispend":
                    print_verbose("reaches aispend for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = args[1] if len(args) > 1 else kwargs["messages"]
                    result = {
                        "model": model,
                        "created": time.time(),
                        "usage": {
                            "prompt_tokens": prompt_token_calculator(
                                model, messages=messages
                            ),
                            "completion_tokens": 0,
                        },
                    }
                    aispendLogger.log_event(
                        model=model,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "llmonitor":
                    print_verbose("reaches llmonitor for logging error!")

                    model = args[0] if len(args) > 0 else kwargs["model"]

                    input = (
                        args[1]
                        if len(args) > 1
                        else kwargs.get("messages", kwargs.get("input", None))
                    )

                    type = "embed" if "input" in kwargs else "llm"

                    llmonitorLogger.log_event(
                        type=type,
                        event="error",
                        user_id=litellm._thread_context.user,
                        model=model,
                        input=input,
                        error=traceback_exception,
                        run_id=kwargs["litellm_call_id"],
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "supabase":
                    print_verbose("reaches supabase for logging!")
                    print_verbose(f"supabaseClient: {supabaseClient}")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = args[1] if len(args) > 1 else kwargs["messages"]
                    result = {
                        "model": model,
                        "created": time.time(),
                        "error": traceback_exception,
                        "usage": {
                            "prompt_tokens": prompt_token_calculator(
                                model, messages=messages
                            ),
                            "completion_tokens": 0,
                        },
                    }
                    supabaseClient.log_event(
                        model=model,
                        messages=messages,
                        end_user=litellm._thread_context.user,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        litellm_call_id=kwargs["litellm_call_id"],
                        print_verbose=print_verbose,
                    )
            except:
                print_verbose(
                    f"Error Occurred while logging failure: {traceback.format_exc()}"
                )
                pass

        if failure_handler and callable(failure_handler):
            call_details = {
                "exception": exception,
                "additional_details": additional_details,
            }
            failure_handler(call_details)
        pass
    except Exception as e:
        # LOGGING
        exception_logging(logger_fn=user_logger_fn, exception=e)
        pass


def handle_success(args, kwargs, result, start_time, end_time):
    global heliconeLogger, aispendLogger, supabaseClient, liteDebuggerClient, llmonitorLogger
    try:
        model = args[0] if len(args) > 0 else kwargs["model"]
        input = (
            args[1]
            if len(args) > 1
            else kwargs.get("messages", kwargs.get("input", None))
        )
        success_handler = additional_details.pop("success_handler", None)
        failure_handler = additional_details.pop("failure_handler", None)
        additional_details["Event_Name"] = additional_details.pop(
            "successful_event_name", "litellm.succes_query"
        )
        for callback in litellm.success_callback:
            try:
                if callback == "posthog":
                    ph_obj = {}
                    for detail in additional_details:
                        ph_obj[detail] = additional_details[detail]
                    event_name = additional_details["Event_Name"]
                    if "user_id" in additional_details:
                        posthog.capture(
                            additional_details["user_id"], event_name, ph_obj
                        )
                    else:  # PostHog calls require a unique id to identify a user - https://posthog.com/docs/libraries/python
                        unique_id = str(uuid.uuid4())
                        posthog.capture(unique_id, event_name, ph_obj)
                    pass
                elif callback == "slack":
                    slack_msg = ""
                    for detail in additional_details:
                        slack_msg += f"{detail}: {additional_details[detail]}\n"
                    slack_app.client.chat_postMessage(
                        channel=alerts_channel, text=slack_msg
                    )
                elif callback == "helicone":
                    print_verbose("reaches helicone for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = args[1] if len(args) > 1 else kwargs["messages"]
                    heliconeLogger.log_success(
                        model=model,
                        messages=messages,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "llmonitor":
                    print_verbose("reaches llmonitor for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]

                    input = (
                        args[1]
                        if len(args) > 1
                        else kwargs.get("messages", kwargs.get("input", None))
                    )

                    # if contains input, it's 'embedding', otherwise 'llm'
                    type = "embed" if "input" in kwargs else "llm"

                    llmonitorLogger.log_event(
                        type=type,
                        event="end",
                        model=model,
                        input=input,
                        user_id=litellm._thread_context.user,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        run_id=kwargs["litellm_call_id"],
                        print_verbose=print_verbose,
                    )
                elif callback == "promptlayer":
                    print_verbose("reaches promptlayer for logging!")
                    promptLayerLogger.log_event(
                        kwargs=kwargs,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "langfuse":
                    print_verbose("reaches langfuse for logging!")
                    langFuseLogger.log_event(
                        kwargs=kwargs,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )

                elif callback == "aispend":
                    print_verbose("reaches aispend for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    aispendLogger.log_event(
                        model=model,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "supabase":
                    print_verbose("reaches supabase for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = (
                        args[1]
                        if len(args) > 1
                        else kwargs.get("messages", {"role": "user", "content": ""})
                    )
                    print(f"supabaseClient: {supabaseClient}")
                    supabaseClient.log_event(
                        model=model,
                        messages=messages,
                        end_user=litellm._thread_context.user,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        litellm_call_id=kwargs["litellm_call_id"],
                        print_verbose=print_verbose,
                    )
                elif callable(callback): # custom logger functions
                    customLogger.log_event(
                        kwargs=kwargs,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
            except Exception as e:
                # LOGGING
                exception_logging(logger_fn=user_logger_fn, exception=e)
                print_verbose(
                    f"[Non-Blocking] Success Callback Error - {traceback.format_exc()}"
                )
                pass

        if success_handler and callable(success_handler):
            success_handler(args, kwargs)
        pass
    except Exception as e:
        # LOGGING
        exception_logging(logger_fn=user_logger_fn, exception=e)
        print_verbose(
            f"[Non-Blocking] Success Callback Error - {traceback.format_exc()}"
        )
        pass


def acreate(*args, **kwargs):  ## Thin client to handle the acreate langchain call
    return litellm.acompletion(*args, **kwargs)


def prompt_token_calculator(model, messages):
    # use tiktoken or anthropic's tokenizer depending on the model
    text = " ".join(message["content"] for message in messages)
    num_tokens = 0
    if "claude" in model:
        try:
            import anthropic
        except:
            Exception("Anthropic import failed please run `pip install anthropic`")
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

        anthropic = Anthropic()
        num_tokens = anthropic.count_tokens(text)
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
            openai.Model.retrieve(model)
        else:
            messages = [{"role": "user", "content": "Hello World"}]
            litellm.completion(model=model, messages=messages)
    except:
        raise InvalidRequestError(message="", model=model, llm_provider="")

# check valid api key 
def check_valid_key(model: str, api_key: str):
    # returns True if key is valid for the model
    # returns False if key is invalid for the model
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        litellm.completion(model=model, messages=messages, api_key=api_key, max_tokens=10)
        return True
    except AuthenticationError as e:
        return False
    except Exception as e:
        return False

# integration helper function
def modify_integration(integration_name, integration_params):
    global supabaseClient
    if integration_name == "supabase":
        if "table_name" in integration_params:
            Supabase.supabase_table_name = integration_params["table_name"]


# custom prompt helper function
def register_prompt_template(model: str, roles: dict, initial_prompt_value: str = "", final_prompt_value: str = ""):
    """
    Example usage:
    ```
    import litellm 
    litellm.register_prompt_template(
	    model="llama-2",
	    roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n",
                "post_message": "\n<</SYS>>\n [/INST]\n"
            },
            "user": { # follow this format https://github.com/facebookresearch/llama/blob/77062717054710e352a99add63d160274ce670c6/llama/generation.py#L348
                "pre_message": "[INST] ",
                "post_message": " [/INST]\n"
            }, 
            "assistant": {
                "post_message": "\n" # follows this - https://replicate.com/blog/how-to-prompt-llama
            }
        }
    )
    ```
    """
    litellm.custom_prompt_dict[model] = {
        "roles": roles,
        "initial_prompt_value": initial_prompt_value,
        "final_prompt_value": final_prompt_value
    }
    return litellm.custom_prompt_dict

####### [BETA] HOSTED PRODUCT ################ - https://docs.litellm.ai/docs/debugging/hosted_debugging


def get_all_keys(llm_provider=None):
    try:
        global last_fetched_at_keys
        # if user is using hosted product -> instantiate their env with their hosted api keys - refresh every 5 minutes
        print_verbose(f"Reaches get all keys, llm_provider: {llm_provider}")
        user_email = (
            os.getenv("LITELLM_EMAIL")
            or litellm.email
            or litellm.token
            or os.getenv("LITELLM_TOKEN")
        )
        if user_email:
            time_delta = 0
            if last_fetched_at_keys != None:
                current_time = time.time()
                time_delta = current_time - last_fetched_at_keys
            if (
                time_delta > 300 or last_fetched_at_keys == None or llm_provider
            ):  # if the llm provider is passed in , assume this happening due to an AuthError for that provider
                # make the api call
                last_fetched_at = time.time()
                print_verbose(f"last_fetched_at: {last_fetched_at}")
                response = requests.post(
                    url="http://api.litellm.ai/get_all_keys",
                    headers={"content-type": "application/json"},
                    data=json.dumps({"user_email": user_email}),
                )
                print_verbose(f"get model key response: {response.text}")
                data = response.json()
                # update model list
                for key, value in data[
                    "model_keys"
                ].items():  # follows the LITELLM API KEY format - <UPPERCASE_PROVIDER_NAME>_API_KEY - e.g. HUGGINGFACE_API_KEY
                    os.environ[key] = value
                # set model alias map
                for model_alias, value in data["model_alias_map"].items():
                    litellm.model_alias_map[model_alias] = value
                return "it worked!"
            return None
        return None
    except:
        print_verbose(
            f"[Non-Blocking Error] get_all_keys error - {traceback.format_exc()}"
        )
        pass


def get_model_list():
    global last_fetched_at
    try:
        # if user is using hosted product -> get their updated model list
        user_email = (
            os.getenv("LITELLM_EMAIL")
            or litellm.email
            or litellm.token
            or os.getenv("LITELLM_TOKEN")
        )
        if user_email:
            # make the api call
            last_fetched_at = time.time()
            print(f"last_fetched_at: {last_fetched_at}")
            response = requests.post(
                url="http://api.litellm.ai/get_model_list",
                headers={"content-type": "application/json"},
                data=json.dumps({"user_email": user_email}),
            )
            print_verbose(f"get_model_list response: {response.text}")
            data = response.json()
            # update model list
            model_list = data["model_list"]
            # # check if all model providers are in environment
            # model_providers = data["model_providers"]
            # missing_llm_provider = None
            # for item in model_providers:
            #     if f"{item.upper()}_API_KEY" not in os.environ:
            #         missing_llm_provider = item
            #         break
            # # update environment - if required
            # threading.Thread(target=get_all_keys, args=(missing_llm_provider)).start()
            return model_list
        return []  # return empty list by default
    except:
        print_verbose(
            f"[Non-Blocking Error] get_model_list error - {traceback.format_exc()}"
        )

####### EXCEPTION MAPPING ################
def exception_type(model, original_exception, custom_llm_provider):
    global user_logger_fn, liteDebuggerClient
    exception_mapping_worked = False
    try:
        if isinstance(original_exception, OriginalError):
            # Handle the OpenAIError
            exception_mapping_worked = True
            if model in litellm.openrouter_models:
                if original_exception.http_status == 413:
                    raise InvalidRequestError(
                        message=str(original_exception),
                        model=model,
                        llm_provider="openrouter"
                    )
                original_exception.llm_provider = "openrouter"
            elif custom_llm_provider == "azure":
                original_exception.llm_provider = "azure"
            else:
                original_exception.llm_provider = "openai"
            if "This model's maximum context length is" in original_exception._message:
                raise ContextWindowExceededError(
                    message=str(original_exception),
                    model=model,
                    llm_provider=original_exception.llm_provider
                )
            raise original_exception
        elif model:
            error_str = str(original_exception)
            if isinstance(original_exception, BaseException):
                exception_type = type(original_exception).__name__
            else:
                exception_type = ""
            if "claude" in model:  # one of the anthropics
                if hasattr(original_exception, "message"):
                    if "prompt is too long" in original_exception.message:
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=original_exception.message, 
                            model=model,
                            llm_provider="anthropic"
                        )
                if hasattr(original_exception, "status_code"):
                    print_verbose(f"status_code: {original_exception.status_code}")
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AnthropicException - {original_exception.message}",
                            llm_provider="anthropic",
                            model=model
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"AnthropicException - {original_exception.message}",
                            model=model,
                            llm_provider="anthropic",
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AnthropicException - {original_exception.message}",
                            model=model,
                            llm_provider="anthropic"
                        )
                    elif original_exception.status_code == 413:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"AnthropicException - {original_exception.message}",
                            model=model,
                            llm_provider="anthropic",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AnthropicException - {original_exception.message}",
                            llm_provider="anthropic",
                            model=model
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"AnthropicException - {original_exception.message}",
                            llm_provider="anthropic",
                            model=model
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"AnthropicException - {original_exception.message}",
                            llm_provider="anthropic",
                            model=model
                        )
            elif "replicate" in model:
                if "Incorrect authentication token" in error_str:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"ReplicateException - {error_str}",
                        llm_provider="replicate",
                        model=model
                    )
                elif "input is too long" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"ReplicateException - {error_str}",
                        model=model,
                        llm_provider="replicate",
                    )
                elif exception_type == "ModelError":
                    exception_mapping_worked = True
                    raise InvalidRequestError(
                        message=f"ReplicateException - {error_str}",
                        model=model,
                        llm_provider="replicate",
                    )
                elif "Request was throttled" in error_str:
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"ReplicateException - {error_str}",
                        llm_provider="replicate",
                        model=model
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate",
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate"
                        )
                    elif original_exception.status_code == 413:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model
                        )
                exception_mapping_worked = True
                raise APIError(
                    status_code=original_exception.status_code, 
                    message=f"ReplicateException - {original_exception.message}",
                    llm_provider="replicate",
                    model=model
                )
            elif model in litellm.cohere_models or custom_llm_provider == "cohere":  # Cohere
                if (
                    "invalid api token" in error_str
                    or "No API key provided." in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model
                    )
                elif "too many tokens" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"CohereException - {original_exception.message}",
                        model=model,
                        llm_provider="cohere",
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 400 or original_exception.status_code == 498:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model
                        )
                elif (
                    "CohereConnectionError" in exception_type
                ):  # cohere seems to fire these errors when we load test it (1k+ messages / min)
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model
                    )
                elif "invalid type:" in error_str:
                    exception_mapping_worked = True
                    raise InvalidRequestError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model
                    )
                elif "Unexpected server error" in error_str:
                    exception_mapping_worked = True
                    raise ServiceUnavailableError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model
                    )
                else:
                    if hasattr(original_exception, "status_code"):
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code, 
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model
                        )
                    raise original_exception
            elif custom_llm_provider == "huggingface":
                if "length limit exceeded" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=error_str,
                        model=model,
                        llm_provider="huggingface"
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            model=model,
                            llm_provider="huggingface",
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"HuggingfaceException - {original_exception.message}",
                            model=model,
                            llm_provider="huggingface"
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code, 
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model
                        )
            elif custom_llm_provider == "ai21":
                if hasattr(original_exception, "message"):
                    if "Prompt has too many tokens" in original_exception.message:
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21"
                        )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21"
                        )
                    if original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code, 
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model
                        )
            elif model in litellm.nlp_cloud_models or custom_llm_provider == "nlp_cloud":
                if "detail" in error_str:
                    if "Input text length should not exceed" in error_str:
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud"
                        )
                    elif "value is not a valid" in error_str:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud"
                        )
                    else: 
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=500,
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud"
                        )
                if hasattr(original_exception, "status_code"): # https://docs.nlpcloud.com/?shell#errors
                    if original_exception.status_code == 400 or original_exception.status_code == 406 or original_exception.status_code == 413 or original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model
                        )
                    elif original_exception.status_code == 401 or original_exception.status_code == 403:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model
                        )
                    elif original_exception.status_code == 522 or original_exception.status_code == 524:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"NLPCloudException - {original_exception.message}",
                            model=model,
                            llm_provider="nlp_cloud"
                        )
                    elif original_exception.status_code == 429 or original_exception.status_code == 402:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                        )
                    elif original_exception.status_code == 500 or original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code, 
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model
                        )
                    elif original_exception.status_code == 504 or original_exception.status_code == 520:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"NLPCloudException - {original_exception.message}",
                            model=model,
                            llm_provider="nlp_cloud"
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code, 
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model
                        )
            elif custom_llm_provider == "together_ai":
                error_response = json.loads(error_str)
                if "error" in error_response and "`inputs` tokens + `max_new_tokens` must be <=" in error_response["error"]:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai"
                    )
                elif "error" in error_response and "invalid private key" in error_response["error"]:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"TogetherAIException - {error_response['error']}",
                        llm_provider="together_ai",
                        model=model
                    )
                elif "error" in error_response and "INVALID_ARGUMENT" in error_response["error"]:
                    exception_mapping_worked = True
                    raise InvalidRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai"
                    )
                elif "error_type" in error_response and error_response["error_type"] == "validation":
                    exception_mapping_worked = True
                    raise InvalidRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai"
                    )
                elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"TogetherAIException - {original_exception.message}",
                            model=model,
                            llm_provider="together_ai"
                        )
                elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"TogetherAIException - {original_exception.message}",
                            llm_provider="together_ai",
                            model=model
                        )
                else: 
                    exception_mapping_worked = True
                    raise APIError(
                        status_code=original_exception.status_code, 
                        message=f"TogetherAIException - {original_exception.message}",
                        llm_provider="together_ai",
                        model=model
                    )
            elif model in litellm.aleph_alpha_models:
                if "This is longer than the model's maximum context length" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"AlephAlphaException - {original_exception.message}",
                        llm_provider="aleph_alpha", 
                        model=model
                    )
                elif hasattr(original_exception, "status_code"):
                    print(f"status code: {original_exception.status_code}")
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise InvalidRequestError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model
                        )
                    raise original_exception
                raise original_exception
            elif custom_llm_provider == "vllm":
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 0:
                        exception_mapping_worked = True
                        raise APIConnectionError(
                            message=f"VLLMException - {original_exception.message}",
                            llm_provider="vllm",
                            model=model
                        )
        raise original_exception
    except Exception as e:
        # LOGGING
        exception_logging(
            logger_fn=user_logger_fn,
            additional_args={
                "exception_mapping_worked": exception_mapping_worked,
                "original_exception": original_exception,
            },
            exception=e,
        )
        ## AUTH ERROR
        if isinstance(e, AuthenticationError) and (
            litellm.email or "LITELLM_EMAIL" in os.environ
        ):
            threading.Thread(target=get_all_keys, args=(e.llm_provider,)).start()
        # don't let an error with mapping interrupt the user from receiving an error from the llm api calls
        if exception_mapping_worked:
            raise e
        else:
            raise original_exception


####### CRASH REPORTING ################
def safe_crash_reporting(model=None, exception=None, custom_llm_provider=None):
    data = {
        "model": model,
        "exception": str(exception),
        "custom_llm_provider": custom_llm_provider,
    }
    threading.Thread(target=litellm_telemetry, args=(data,)).start()

def get_or_generate_uuid():
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp")
    uuid_file =  os.path.join(temp_dir, "litellm_uuid.txt")
    try:
        # Try to open the file and load the UUID
        with open(uuid_file, "r") as file:
            uuid_value = file.read()
            if uuid_value:
                uuid_value = uuid_value.strip()
            else:
                raise FileNotFoundError

    except FileNotFoundError:
        # Generate a new UUID if the file doesn't exist or is empty
        try: 
            new_uuid = uuid.uuid4()
            uuid_value = str(new_uuid)
            with open(uuid_file, "w") as file:
                file.write(uuid_value)
        except: # if writing to tmp/litellm_uuid.txt then retry writing to litellm_uuid.txt
            try:
                new_uuid = uuid.uuid4()
                uuid_value = str(new_uuid)
                with open("litellm_uuid.txt", "w") as file:
                    file.write(uuid_value)
            except: # if this 3rd attempt fails just pass
                # Good first issue for someone to improve this function :) 
                return
    except:
        # [Non-Blocking Error]
        return
    return uuid_value


def litellm_telemetry(data):
    # Load or generate the UUID
    uuid_value = ""
    try:
        uuid_value = get_or_generate_uuid()
    except:
        uuid_value = str(uuid.uuid4())
    try:
        # Prepare the data to send to litellm logging api
        payload = {
            "uuid": uuid_value,
            "data": data,
            "version:": importlib.metadata.version("litellm"),
        }
        # Make the POST request to litellm logging api
        response = requests.post(
            "https://litellm.berri.ai/logging",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
    except:
        # [Non-Blocking Error]
        return


######### Secret Manager ############################
# checks if user has passed in a secret manager client
# if passed in then checks the secret there
def get_secret(secret_name):
    if litellm.secret_manager_client != None:
        # TODO: check which secret manager is being used
        # currently only supports Infisical
        try:
            secret = litellm.secret_manager_client.get_secret(secret_name).secret_value
        except:
            secret = None
        return secret
    else:
        return os.environ.get(secret_name)


######## Streaming Class ############################
# wraps the completion stream to return the correct format for the model
# replicate/anthropic/cohere
class CustomStreamWrapper:
    def __init__(self, completion_stream, model, custom_llm_provider=None, logging_obj=None):
        self.model = model
        self.custom_llm_provider = custom_llm_provider
        self.logging_obj = logging_obj
        self.completion_stream = completion_stream
        self.sent_first_chunk = False
        if self.logging_obj:
                # Log the type of the received item
                self.logging_obj.post_call(str(type(completion_stream)))

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def logging(self, text):
        if self.logging_obj: 
            self.logging_obj.post_call(text)

    def handle_anthropic_chunk(self, chunk):
        str_line = chunk.decode("utf-8")  # Convert bytes to string
        if str_line.startswith("data:"):
            data_json = json.loads(str_line[5:])
            return data_json.get("completion", "")
        return ""

    def handle_together_ai_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        text_index = chunk.find('"text":"')  # this checks if text: exists
        text_start = text_index + len('"text":"')
        text_end = chunk.find('"}', text_start)
        if text_index != -1 and text_end != -1:
            extracted_text = chunk[text_start:text_end]
            return extracted_text
        else:
            return ""

    def handle_huggingface_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        if chunk.startswith("data:"):
            data_json = json.loads(chunk[5:])
            if "token" in data_json and "text" in data_json["token"]:
                return data_json["token"]["text"]
            else:
                return ""
        return ""
    
    def handle_ai21_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            return data_json["completions"][0]["data"]["text"]
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")
    
    def handle_nlp_cloud_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            print(f"data json: {data_json}")
            return data_json["generated_text"]
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")
    
    def handle_aleph_alpha_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            return data_json["completions"][0]["completion"]
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")
    
    def handle_cohere_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            return data_json["text"]
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")
    
    def handle_openai_text_completion_chunk(self, chunk):
        try:
            print(f"chunk: {chunk}")
            return chunk["choices"][0]["text"]
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_openai_chat_completion_chunk(self, chunk):
        try:
            return chunk["choices"][0]["delta"]["content"]
        except:
            return ""

    def handle_baseten_chunk(self, chunk):
        try:
            chunk = chunk.decode("utf-8")
            if len(chunk) > 0:
                if chunk.startswith("data:"):
                    data_json = json.loads(chunk[5:])
                    if "token" in data_json and "text" in data_json["token"]:
                        return data_json["token"]["text"]
                    else:
                        return ""
                data_json = json.loads(chunk)
                if "model_output" in data_json:
                    if isinstance(data_json["model_output"], dict) and "data" in data_json["model_output"] and isinstance(data_json["model_output"]["data"], list):
                        return data_json["model_output"]["data"][0]
                    elif isinstance(data_json["model_output"], str):
                        return data_json["model_output"]
                    elif "completion" in data_json and isinstance(data_json["completion"], str):
                        return data_json["completion"]
                    else:
                        raise ValueError(f"Unable to parse response. Original response: {chunk}")
                else:
                    return ""
            else:
                return ""
        except:
            traceback.print_exc()
            return ""

    def handle_bedrock_stream(self):
        if self.completion_stream:
            event = next(self.completion_stream)
            chunk = event.get('chunk')
            if chunk:
                chunk_data = json.loads(chunk.get('bytes').decode())
                return chunk_data['outputText']
        return ""

    def __next__(self):
        model_response = ModelResponse(stream=True, model=self.model)
        try:
            # return this for all models
            if self.sent_first_chunk == False:
                model_response.choices[0].delta.role = "assistant"
                self.sent_first_chunk = True
            completion_obj = {"content": ""} # default to role being assistant
            if self.model in litellm.anthropic_models:
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_anthropic_chunk(chunk)
            elif self.model == "replicate" or self.custom_llm_provider == "replicate":
                chunk = next(self.completion_stream)
                completion_obj["content"] = chunk
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "together_ai"):
                chunk = next(self.completion_stream)
                text_data = self.handle_together_ai_chunk(chunk)
                if text_data == "":
                    return self.__next__()
                completion_obj["content"] = text_data
            elif self.custom_llm_provider and self.custom_llm_provider == "huggingface":
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_huggingface_chunk(chunk)
            elif self.custom_llm_provider and self.custom_llm_provider == "baseten": # baseten doesn't provide streaming
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_baseten_chunk(chunk)
            elif self.custom_llm_provider and self.custom_llm_provider == "ai21": #ai21 doesn't provide streaming
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_ai21_chunk(chunk)
            elif self.custom_llm_provider and self.custom_llm_provider == "vllm":
                chunk = next(self.completion_stream)
                completion_obj["content"] = chunk[0].outputs[0].text
            elif self.model in litellm.aleph_alpha_models: #aleph alpha doesn't provide streaming
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_aleph_alpha_chunk(chunk)
            elif self.model in litellm.open_ai_text_completion_models:
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_openai_text_completion_chunk(chunk)
            elif self.model in litellm.nlp_cloud_models or self.custom_llm_provider == "nlp_cloud":
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_nlp_cloud_chunk(chunk)
            elif self.model in (litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models):
                chunk = next(self.completion_stream)
                completion_obj["content"] = str(chunk)
            elif self.model in litellm.cohere_models or self.custom_llm_provider == "cohere":
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_cohere_chunk(chunk)
            elif self.custom_llm_provider == "bedrock":
                completion_obj["content"] = self.handle_bedrock_stream()
            else: # openai chat/azure models
                chunk = next(self.completion_stream)
                model_response = chunk
                # LOGGING
                threading.Thread(target=self.logging_obj.success_handler, args=(completion_obj,)).start()
                return model_response

            # LOGGING
            threading.Thread(target=self.logging_obj.success_handler, args=(completion_obj,)).start()
            model_response.model = self.model
            model_response.choices[0].delta["content"] = completion_obj["content"]
            return model_response
        except StopIteration:
            raise StopIteration
        except Exception as e:
            traceback.print_exc()
            model_response.choices[0].finish_reason = "stop"
            return model_response
    
    async def __anext__(self):
        try:
            return next(self)
        except StopIteration:
            raise StopAsyncIteration


def mock_completion_streaming_obj(model_response, mock_response, model):
    for i in range(0, len(mock_response), 3):
        completion_obj = {"role": "assistant", "content": mock_response[i: i+3]}
        model_response.choices[0].delta = completion_obj
        yield model_response

########## Reading Config File ############################
def read_config_args(config_path):
    try:
        import os

        current_path = os.getcwd()
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        # read keys/ values from config file and return them
        return config
    except Exception as e:
        print("An error occurred while reading config:", str(e))
        raise e

########## experimental completion variants ############################

def get_model_split_test(models, completion_call_id):
    global last_fetched_at
    try:
        # make the api call
        last_fetched_at = time.time()
        print(f"last_fetched_at: {last_fetched_at}")
        response = requests.post(
            #http://api.litellm.ai
            url="http://api.litellm.ai/get_model_split_test", # get the updated dict from table or update the table with the dict
            headers={"content-type": "application/json"},
            data=json.dumps({"completion_call_id": completion_call_id, "models": models}),
        )
        print_verbose(f"get_model_list response: {response.text}")
        data = response.json()
        # update model list
        split_test_models = data["split_test_models"]
        model_configs = data.get("model_configs", {})
        # update environment - if required
        threading.Thread(target=get_all_keys, args=()).start()
        return split_test_models, model_configs
    except:
        print_verbose(
            f"[Non-Blocking Error] get_all_keys error - {traceback.format_exc()}"
        )


def completion_with_split_tests(models={}, messages=[], use_client=False, override_client=False, **kwargs):
    """
    Example Usage: 

    models =  {
	    "gpt-4": 0.7, 
	    "huggingface/wizard-coder": 0.3
    }
    messages = [{ "content": "Hello, how are you?","role": "user"}]
    completion_with_split_tests(models=models, messages=messages)
    """
    import random
    model_configs = {}
    if use_client and not override_client:
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = str(uuid.uuid4())
            #raise ValueError("Please tag this completion call, if you'd like to update it's split test values through the UI. - eg. `completion_with_split_tests(.., id=1234)`.")
        # get the most recent model split list from server 
        models, model_configs = get_model_split_test(models=models, completion_call_id=kwargs["id"])

    try:
        selected_llm = random.choices(list(models.keys()), weights=list(models.values()))[0]
    except:
        traceback.print_exc()
        raise ValueError("""models does not follow the required format - {'model_name': 'split_percentage'}, e.g. {'gpt-4': 0.7, 'huggingface/wizard-coder': 0.3}""")
    
    # use dynamic model configs if users set 
    if model_configs!={}:
        selected_model_configs = model_configs.get(selected_llm, {})
        if "prompt" in selected_model_configs: # special case, add this to messages as system prompt
            messages.append({"role": "system", "content": selected_model_configs["prompt"]})
            selected_model_configs.pop("prompt")
        for param_name in selected_model_configs:
            if param_name == "temperature":
                kwargs[param_name] = float(selected_model_configs[param_name])
            elif param_name == "max_tokens":
                kwargs[param_name] = int(selected_model_configs[param_name])
            else:
                kwargs[param_name] = selected_model_configs[param_name]

    return litellm.completion(model=selected_llm, messages=messages, use_client=use_client, **kwargs)

def completion_with_fallbacks(**kwargs):
    response = None
    rate_limited_models = set()
    model_expiration_times = {}
    start_time = time.time()
    fallbacks = [kwargs["model"]] + kwargs["fallbacks"]
    del kwargs["fallbacks"]  # remove fallbacks so it's not recursive

    while response == None and time.time() - start_time < 45:
        for model in fallbacks:
            # loop thru all models
            try:
                if (
                    model in rate_limited_models
                ):  # check if model is currently cooling down
                    if (
                        model_expiration_times.get(model)
                        and time.time() >= model_expiration_times[model]
                    ):
                        rate_limited_models.remove(
                            model
                        )  # check if it's been 60s of cool down and remove model
                    else:
                        continue  # skip model

                # delete model from kwargs if it exists
                if kwargs.get("model"):
                    del kwargs["model"]

                print("making completion call", model)
                response = litellm.completion(**kwargs, model=model)

                if response != None:
                    return response

            except Exception as e:
                print(f"got exception {e} for model {model}")
                rate_limited_models.add(model)
                model_expiration_times[model] = (
                    time.time() + 60
                )  # cool down this selected model
                # print(f"rate_limited_models {rate_limited_models}")
                pass
    return response

def process_system_message(system_message, max_tokens, model):
    system_message_event = {"role": "system", "content": system_message}
    system_message_tokens = get_token_count(system_message_event, model)

    if system_message_tokens > max_tokens:
        print_verbose("`tokentrimmer`: Warning, system message exceeds token limit. Trimming...")
        # shorten system message to fit within max_tokens
        new_system_message = shorten_message_to_fit_limit(system_message_event, max_tokens, model)
        system_message_tokens = get_token_count(new_system_message, model)
        
    return system_message_event, max_tokens - system_message_tokens

def process_messages(messages, max_tokens, model):
    # Process messages from older to more recent
    messages = messages[::-1]
    final_messages = []

    for message in messages:
        final_messages = attempt_message_addition(final_messages, message, max_tokens, model)

    return final_messages

def attempt_message_addition(final_messages, message, max_tokens, model):
    temp_messages = [message] + final_messages
    temp_message_tokens = get_token_count(messages=temp_messages, model=model)

    if temp_message_tokens <= max_tokens:
        return temp_messages
    
    # if temp_message_tokens > max_tokens, try shortening temp_messages
    elif "function_call" not in message:
        # fit updated_message to be within temp_message_tokens - max_tokens (aka the amount temp_message_tokens is greate than max_tokens)
        updated_message = shorten_message_to_fit_limit(message, temp_message_tokens - max_tokens, model)
        if can_add_message(updated_message, final_messages, max_tokens, model):
            return [updated_message] + final_messages

    return final_messages

def can_add_message(message, messages, max_tokens, model):
    if get_token_count(messages + [message], model) <= max_tokens:
        return True
    return False

def get_token_count(messages, model):
    return token_counter(model=model, messages=messages)


def shorten_message_to_fit_limit(
        message,
        tokens_needed,
        model):
    """
    Shorten a message to fit within a token limit by removing characters from the middle.
    """
    content = message["content"]

    while True:
        total_tokens = get_token_count([message], model)

        if total_tokens <= tokens_needed:
            break

        ratio = (tokens_needed) / total_tokens
        
        new_length = int(len(content) * ratio)
        print_verbose(new_length)

        half_length = new_length // 2
        left_half = content[:half_length]
        right_half = content[-half_length:]

        trimmed_content = left_half + '..' + right_half
        message["content"] = trimmed_content
        content = trimmed_content

    return message

# LiteLLM token trimmer 
# this code is borrowed from https://github.com/KillianLucas/tokentrim/blob/main/tokentrim/tokentrim.py
# Credits for this code go to Killian Lucas
def trim_messages(
    messages,
    model = None,
    system_message = None, # str of user system message
    trim_ratio: float = 0.75,
    return_response_tokens: bool = False,
    max_tokens = None
    ):
    """
    Trim a list of messages to fit within a model's token limit.

    Args:
        messages: Input messages to be trimmed. Each message is a dictionary with 'role' and 'content'.
        model: The LiteLLM model being used (determines the token limit).
        system_message: Optional system message to preserve at the start of the conversation.
        trim_ratio: Target ratio of tokens to use after trimming. Default is 0.75, meaning it will trim messages so they use about 75% of the model's token limit.
        return_response_tokens: If True, also return the number of tokens left available for the response after trimming.
        max_tokens: Instead of specifying a model or trim_ratio, you can specify this directly.

    Returns:
        Trimmed messages and optionally the number of tokens available for response.
    """
    # Initialize max_tokens
    # if users pass in max tokens, trim to this amount
    try:
        if max_tokens == None:
            # Check if model is valid
            if model in litellm.model_cost:
                max_tokens_for_model  = litellm.model_cost[model]['max_tokens'] 
                max_tokens = int(max_tokens_for_model * trim_ratio)
            else:
                # if user did not specify max tokens 
                # or passed an llm litellm does not know
                # do nothing, just return messages
                return 
        
        current_tokens = token_counter(model=model, messages=messages)

        # Do nothing if current tokens under messages
        if current_tokens < max_tokens:
            return messages 
        
        #### Trimming messages if current_tokens > max_tokens
        print_verbose(f"Need to trim input messages: {messages}, current_tokens{current_tokens}, max_tokens: {max_tokens}")
        if system_message:
            system_message_event, max_tokens = process_system_message(system_message=system_message, max_tokens=max_tokens, model=model)
            messages = messages + [system_message_event]

        final_messages = process_messages(messages=messages, max_tokens=max_tokens, model=model)

        if return_response_tokens: # if user wants token count with new trimmed messages
            response_tokens = max_tokens - get_token_count(final_messages, model)
            return final_messages, response_tokens

        return final_messages
    except: # [NON-Blocking, if error occurs just return final_messages
        return messages

# this helper reads the .env and returns a list of supported llms for user
def get_valid_models():
    try:
        # get keys set in .env
        environ_keys = os.environ.keys()
        valid_providers = []
        # for all valid providers, make a list of supported llms
        valid_models = []

        for provider in litellm.provider_list:
            # edge case litellm has together_ai as a provider, it should be togetherai
            provider = provider.replace("_", "")

            # litellm standardizes expected provider keys to 
            # PROVIDER_API_KEY. Example: OPENAI_API_KEY, COHERE_API_KEY
            expected_provider_key = f"{provider.upper()}_API_KEY"
            if expected_provider_key in environ_keys:            
                # key is set 
                valid_providers.append(provider)
        
        for provider in valid_providers:
            if provider == "azure":
                valid_models.append("Azure-LLM")
            else:
                models_for_provider = litellm.models_by_provider.get(provider, [])
                valid_models.extend(models_for_provider)
        return valid_models
    except:
        return [] # NON-Blocking
