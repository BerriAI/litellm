import sys
import dotenv, json, traceback, threading
import subprocess, os
import litellm, openai
import random, uuid, requests
import datetime, time
import tiktoken
import uuid

encoding = tiktoken.get_encoding("cl100k_base")
import importlib.metadata
from .integrations.traceloop import TraceloopLogger
from .integrations.helicone import HeliconeLogger
from .integrations.aispend import AISpendLogger
from .integrations.berrispend import BerriSpendLogger
from .integrations.supabase import Supabase
from .integrations.llmonitor import LLMonitorLogger
from .integrations.prompt_layer import PromptLayerLogger
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
    ContextWindowExceededError
)
from typing import List, Dict, Union, Optional
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


class Message(OpenAIObject):
    def __init__(self, content="default", role="assistant", logprobs=None, **params):
        super(Message, self).__init__(**params)
        self.content = content
        self.role = role
        self.logprobs = logprobs


class Choices(OpenAIObject):
    def __init__(self, finish_reason="stop", index=0, message=Message(), **params):
        super(Choices, self).__init__(**params)
        self.finish_reason = finish_reason
        self.index = index
        self.message = message


class ModelResponse(OpenAIObject):
    def __init__(self, choices=None, created=None, model=None, usage=None, **params):
        super(ModelResponse, self).__init__(**params)
        self.choices = self.choices = choices if choices else [Choices(message=Message())]
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

    def to_dict_recursive(self):
        d = super().to_dict_recursive()
        d["choices"] = [choice.to_dict_recursive() for choice in self.choices]
        return d


############################################################
def print_verbose(print_statement):
    if litellm.set_verbose:
        print(f"LiteLLM: {print_statement}")
        if random.random() <= 0.3:
            print("Get help - https://discord.com/invite/wuPM9dRgDw")


####### Package Import Handler ###################


def install_and_import(package: str):
    if package in globals().keys():
        print_verbose(f"{package} has already been imported.")
        return
    try:
        # Import the module
        module = importlib.import_module(package)
    except ImportError:
        print_verbose(f"{package} is not installed. Installing...")
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = importlib.import_module(package)
    # except VersionConflict as vc:
    #     print_verbose(f"Detected version conflict for {package}. Upgrading...")
    #     subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", package])
    #     globals()[package] = importlib.import_module(package)
    finally:
        if package not in globals().keys():
            globals()[package] = importlib.import_module(package)


##################################################


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
    
    def update_environment_variables(self, model, optional_params, litellm_params):
        self.optional_params = optional_params
        self.model = model
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
        print_verbose(f"Logging Details Pre-API Call")
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
                        print_verbose("reaches litedebugger for logging!")
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
            if "use_client" in kwargs and kwargs["use_client"] == True: 
                print_verbose(f"litedebugger initialized")
                litellm.input_callback.append("lite_debugger")
                litellm.success_callback.append("lite_debugger")
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
        except:  # DO NOT BLOCK running the function because of this
            print_verbose(f"[Non-Blocking] {traceback.format_exc()}; args - {args}; kwargs - {kwargs}")
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
        logging_obj = function_setup(start_time, *args, **kwargs)
        kwargs["litellm_logging_obj"] = logging_obj
        try:
            # [OPTIONAL] CHECK CACHE
            # remove this after deprecating litellm.caching
            if (litellm.caching or litellm.caching_with_models) and litellm.cache is None:
                litellm.cache = Cache() 

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
            threading.Thread(target=logging_obj.success_handler, args=(result, start_time, end_time)).start()
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


def token_counter(model, text):
    # use tiktoken or anthropic's tokenizer depending on the model
    num_tokens = 0
    if "claude" in model:
        install_and_import("anthropic")
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


def completion_cost(model="gpt-3.5-turbo", prompt="", completion=""):
    prompt_tokens = token_counter(model=model, text=prompt)
    completion_tokens = token_counter(model=model, text=completion)
    prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar


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
    user="",
    deployment_id=None,
    model=None,
    custom_llm_provider="",
    top_k=40,
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
        # any replicate models
        # TODO: handle translating remaining replicate params
        if stream:
            optional_params["stream"] = stream
            return optional_params
    elif custom_llm_provider == "together_ai" or ("togethercomputer" in model):
        if stream:
            optional_params["stream_tokens"] = stream
        if temperature != 1:
            optional_params["temperature"] = temperature
        if top_p != 1:
            optional_params["top_p"] = top_p
        if max_tokens != float("inf"):
            optional_params["max_tokens"] = max_tokens
        if frequency_penalty != 0:
            optional_params["frequency_penalty"] = frequency_penalty
    elif (
        model == "chat-bison"
    ):  # chat-bison has diff args from chat-bison@001 ty Google
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
    elif custom_llm_provider == "baseten":
        optional_params["temperature"] = temperature
        optional_params["stream"] = stream
        if top_p != 1:
            optional_params["top_p"] = top_p
        optional_params["top_k"] = top_k
        optional_params["num_beams"] = num_beams
        if max_tokens != float("inf"):
            optional_params["max_new_tokens"] = max_tokens
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
        optional_params["details"] = True
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


def get_max_tokens(model: str):
    try:
        return litellm.model_cost[model]
    except:
        raise Exception("This model isn't mapped yet. Add it here - https://raw.githubusercontent.com/BerriAI/litellm/main/cookbook/community-resources/max_tokens.json")
    

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


def set_callbacks(callback_list, function_id=None):
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, traceloopLogger, heliconeLogger, aispendLogger, berrispendLogger, supabaseClient, liteDebuggerClient, llmonitorLogger, promptLayerLogger, langFuseLogger
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
        install_and_import("anthropic")
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


# integration helper function
def modify_integration(integration_name, integration_params):
    global supabaseClient
    if integration_name == "supabase":
        if "table_name" in integration_params:
            Supabase.supabase_table_name = integration_params["table_name"]


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
            # check if all model providers are in environment
            model_providers = data["model_providers"]
            missing_llm_provider = None
            for item in model_providers:
                if f"{item.upper()}_API_KEY" not in os.environ:
                    missing_llm_provider = item
                    break
            # update environment - if required
            threading.Thread(target=get_all_keys, args=(missing_llm_provider)).start()
            return model_list
        return []  # return empty list by default
    except:
        print_verbose(
            f"[Non-Blocking Error] get_all_keys error - {traceback.format_exc()}"
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
                elif (
                    "Could not resolve authentication method. Expected either api_key or auth_token to be set."
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
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
                elif (
                    exception_type == "ReplicateError"
                ):  # ReplicateError implies an error on Replicate server side, not user side
                    raise ServiceUnavailableError(
                        message=f"ReplicateException - {error_str}",
                        llm_provider="replicate",
                        model=model
                    )
            elif model in litellm.cohere_models:  # Cohere
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
                elif (
                    "CohereConnectionError" in exception_type
                ):  # cohere seems to fire these errors when we load test it (1k+ messages / min)
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model
                    )
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
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
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
                elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"TogetherAIException - {original_exception.message}",
                            llm_provider="together_ai",
                            model=model
                        )
            raise original_exception  # base case - return the original exception
        else:
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
        if exception_mapping_worked:
            raise e
        else:  # don't let an error with mapping interrupt the user from receiving an error from the llm api calls
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
        new_uuid = uuid.uuid4()
        uuid_value = str(new_uuid)
        with open(uuid_file, "w") as file:
            file.write(uuid_value)
    except:
        # [Non-Blocking Error]
        return
    return uuid_value


def litellm_telemetry(data):
    # Load or generate the UUID
    uuid_value = get_or_generate_uuid()
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
    elif litellm.api_key != None:  # if users use litellm default key
        return litellm.api_key
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
        if self.logging_obj:
                # Log the type of the received item
                self.logging_obj.post_call(str(type(completion_stream)))
        if model in litellm.cohere_models:
            # cohere does not return an iterator, so we need to wrap it in one
            self.completion_stream = iter(completion_stream)
        else:
            self.completion_stream = completion_stream

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
    
    def handle_aleph_alpha_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            return data_json["completions"][0]["completion"]
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")
    
    def handle_openai_text_completion_chunk(self, chunk):
        try:
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

    def __next__(self):
        try:
            completion_obj = {"role": "assistant", "content": ""}
            if self.model in litellm.anthropic_models:
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_anthropic_chunk(chunk)
            elif self.model == "replicate":
                chunk = next(self.completion_stream)
                completion_obj["content"] = chunk
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "together_ai"
            ) or ("togethercomputer" in self.model):
                chunk = next(self.completion_stream)
                text_data = self.handle_together_ai_chunk(chunk)
                if text_data == "":
                    return self.__next__()
                completion_obj["content"] = text_data
            elif self.model in litellm.cohere_models:
                chunk = next(self.completion_stream)
                completion_obj["content"] = chunk.text
            elif self.custom_llm_provider and self.custom_llm_provider == "huggingface":
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_huggingface_chunk(chunk)
            elif self.custom_llm_provider and self.custom_llm_provider == "baseten": # baseten doesn't provide streaming
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_baseten_chunk(chunk)
            elif self.custom_llm_provider and self.custom_llm_provider == "ai21": #ai21 doesn't provide streaming
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_ai21_chunk(chunk)
            elif self.model in litellm.aleph_alpha_models: #ai21 doesn't provide streaming
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_aleph_alpha_chunk(chunk)
            elif self.model in litellm.open_ai_text_completion_models:
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_openai_text_completion_chunk(chunk)
            else: # openai chat/azure models
                chunk = next(self.completion_stream)
                completion_obj["content"] = self.handle_openai_chat_completion_chunk(chunk)
            
            # LOGGING
            threading.Thread(target=self.logging_obj.success_handler, args=(completion_obj,)).start()
            # return this for all models
            return {"choices": [{"delta": completion_obj}]}
        except Exception as e:
            raise StopIteration
    
    async def __anext__(self):
        try:
            return next(self)
        except StopIteration:
            raise StopAsyncIteration


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


########## ollama implementation ############################


async def get_ollama_response_stream(
    api_base="http://localhost:11434", model="llama2", prompt="Why is the sky blue?"
):
    session = aiohttp.ClientSession()
    url = f"{api_base}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
    }
    try:
        async with session.post(url, json=data) as resp:
            async for line in resp.content.iter_any():
                if line:
                    try:
                        json_chunk = line.decode("utf-8")
                        chunks = json_chunk.split("\n")
                        for chunk in chunks:
                            if chunk.strip() != "":
                                j = json.loads(chunk)
                                if "response" in j:
                                    completion_obj = {
                                        "role": "assistant",
                                        "content": "",
                                    }
                                    completion_obj["content"] = j["response"]
                                    yield {"choices": [{"delta": completion_obj}]}
                                    # self.responses.append(j["response"])
                                    # yield "blank"
                    except Exception as e:
                        print(f"Error decoding JSON: {e}")
    finally:
        await session.close()


async def stream_to_string(generator):
    response = ""
    async for chunk in generator:
        response += chunk["content"]
    return response


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