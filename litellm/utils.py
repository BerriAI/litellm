import dotenv
import json
import traceback
import threading
import traceback
import subprocess
import uuid
import litellm
import os 
import openai 
import random
from openai.error import AuthenticationError, InvalidRequestError, RateLimitError, ServiceUnavailableError, OpenAIError
####### ENVIRONMENT VARIABLES ###################
dotenv.load_dotenv() # Loading env variables using dotenv
sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
posthog = None
slack_app = None
alerts_channel = None
callback_list = []
user_logger_fn = None
additional_details = {}

def print_verbose(print_statement):
  if litellm.set_verbose:
    print(f"LiteLLM: {print_statement}")
    if random.random() <= 0.3:
      print("Get help - https://discord.com/invite/wuPM9dRgDw")

####### LOGGING ###################
#Logging function -> log the exact model details + what's being sent | Non-Blocking
def logging(model, input, azure=False, additional_args={}, logger_fn=None, exception=None):
  try:
    model_call_details = {}
    model_call_details["model"] = model
    model_call_details["input"] = input
    model_call_details["azure"] = azure
    # log exception details
    if exception:
      model_call_details["original_exception"] = exception
    # log additional call details -> api key, etc. 
    if azure == True or model in litellm.open_ai_chat_completion_models or model in litellm.open_ai_chat_completion_models or model in litellm.open_ai_embedding_models:
      model_call_details["api_type"] = openai.api_type
      model_call_details["api_base"] = openai.api_base
      model_call_details["api_version"] = openai.api_version
      model_call_details["api_key"] = openai.api_key
    elif "replicate" in model:
      model_call_details["api_key"] = os.environ.get("REPLICATE_API_TOKEN")
    elif model in litellm.anthropic_models:
      model_call_details["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
    elif model in litellm.cohere_models:
      model_call_details["api_key"] = os.environ.get("COHERE_API_KEY")
    model_call_details["additional_args"] = additional_args
    ## User Logging -> if you pass in a custom logging function or want to use sentry breadcrumbs
    print_verbose(f"Basic model call details: {model_call_details}")
    if logger_fn and callable(logger_fn):
      try:
        logger_fn(model_call_details) # Expectation: any logger function passed in by the user should accept a dict object
      except:
        print_verbose(f"[Non-Blocking] Exception occurred while logging {traceback.format_exc()}")
  except:
    traceback.print_exc()
    pass

####### CLIENT ################### 
# make it easy to log if completion/embedding runs succeeded or failed + see what happened | Non-Blocking
def client(original_function):
    def function_setup(*args, **kwargs): #just run once to check if user wants to send their data anywhere - PostHog/Sentry/Slack/etc.
      try: 
        global callback_list, add_breadcrumb
        if (len(litellm.success_callback) > 0 or len(litellm.failure_callback) > 0) and len(callback_list) == 0: 
          callback_list = list(set(litellm.success_callback + litellm.failure_callback))
          set_callbacks(callback_list=callback_list)
        if add_breadcrumb:
          add_breadcrumb(
                category="litellm.llm_call",
                message=f"Positional Args: {args}, Keyword Args: {kwargs}",
                level="info",
            )
      except: # DO NOT BLOCK running the function because of this
        print_verbose(f"[Non-Blocking] {traceback.format_exc()}")
      pass

    def wrapper(*args, **kwargs):
        try:
          function_setup(args, kwargs)
          ## MODEL CALL
          result = original_function(*args, **kwargs)
          ## LOG SUCCESS 
          my_thread = threading.Thread(target=handle_success, args=(args, kwargs)) # don't interrupt execution of main thread
          my_thread.start()
          return result
        except Exception as e:
          traceback_exception = traceback.format_exc()
          my_thread = threading.Thread(target=handle_failure, args=(e, traceback_exception, args, kwargs)) # don't interrupt execution of main thread
          my_thread.start()
          raise e
    return wrapper

####### HELPER FUNCTIONS ################
def set_callbacks(callback_list):
  global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel
  for callback in callback_list:
    if callback == "sentry":
      try:
          import sentry_sdk
      except ImportError:
          print_verbose("Package 'sentry_sdk' is missing. Installing it...")
          subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentry_sdk'])
          import sentry_sdk
      sentry_sdk_instance = sentry_sdk
      sentry_sdk_instance.init(dsn=os.environ.get("SENTRY_API_URL"), traces_sample_rate=float(os.environ.get("SENTRY_API_TRACE_RATE")))
      capture_exception = sentry_sdk_instance.capture_exception
      add_breadcrumb = sentry_sdk_instance.add_breadcrumb 
    elif callback == "posthog":
      try:
          from posthog import Posthog
      except ImportError:
          print_verbose("Package 'posthog' is missing. Installing it...")
          subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'posthog'])
          from posthog import Posthog
      posthog = Posthog(
        project_api_key=os.environ.get("POSTHOG_API_KEY"),
        host=os.environ.get("POSTHOG_API_URL"))
    elif callback == "slack":
      try:
          from slack_bolt import App
      except ImportError:
          print_verbose("Package 'slack_bolt' is missing. Installing it...")
          subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'slack_bolt'])
          from slack_bolt import App
      slack_app = App(
        token=os.environ.get("SLACK_API_TOKEN"),
        signing_secret=os.environ.get("SLACK_API_SECRET")
      )
      alerts_channel = os.environ["SLACK_API_CHANNEL"]
      print_verbose(f"Initialized Slack App: {slack_app}")


def handle_failure(exception, traceback_exception, args, kwargs):
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel
    print_verbose(f"handle_failure args: {args}")
    print_verbose(f"handle_failure kwargs: {kwargs}")
    
    success_handler = additional_details.pop("success_handler", None)
    failure_handler = additional_details.pop("failure_handler", None)
    
    additional_details["Event_Name"] = additional_details.pop("failed_event_name", "litellm.failed_query")
    print_verbose(f"self.failure_callback: {litellm.failure_callback}")

    print_verbose(f"additional_details: {additional_details}")
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
          slack_app.client.chat_postMessage(channel=alerts_channel, text=slack_msg)
        elif callback == "sentry":
          capture_exception(exception)
        elif callback == "posthog": 
          print_verbose(f"inside posthog, additional_details: {len(additional_details.keys())}")
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
            posthog.capture(additional_details["user_id"], event_name, ph_obj)
          else: # PostHog calls require a unique id to identify a user - https://posthog.com/docs/libraries/python
            unique_id = str(uuid.uuid4())
            posthog.capture(unique_id, event_name)
            print_verbose(f"successfully logged to PostHog!")
      except:
        print_verbose(f"Error Occurred while logging failure: {traceback.format_exc()}")
        pass
    
    if failure_handler and callable(failure_handler):
      call_details = {
        "exception": exception,
        "additional_details": additional_details
      }
      failure_handler(call_details)
    pass

def handle_success(*args, **kwargs):
  success_handler = additional_details.pop("success_handler", None)
  failure_handler = additional_details.pop("failure_handler", None)
  additional_details["Event_Name"] = additional_details.pop("successful_event_name", "litellm.succes_query")
  for callback in litellm.success_callback:
    try:
      if callback == "posthog":
        ph_obj = {}
        for detail in additional_details:
          ph_obj[detail] = additional_details[detail]
        event_name = additional_details["Event_Name"]
        if "user_id" in additional_details:
          posthog.capture(additional_details["user_id"], event_name, ph_obj)
        else: # PostHog calls require a unique id to identify a user - https://posthog.com/docs/libraries/python
          unique_id = str(uuid.uuid4())
          posthog.capture(unique_id, event_name, ph_obj)
        pass
      elif callback == "slack":
        slack_msg = "" 
        for detail in additional_details: 
          slack_msg += f"{detail}: {additional_details[detail]}\n"
        slack_app.client.chat_postMessage(channel=alerts_channel, text=slack_msg)
    except:
      pass
  
  if success_handler and callable(success_handler):
    success_handler(args, kwargs)
  pass


def exception_type(model, original_exception):
    if isinstance(original_exception, OpenAIError):
        # Handle the OpenAIError
        raise original_exception
    elif model:
      error_str = str(original_exception)
      if isinstance(original_exception, BaseException):
        exception_type = type(original_exception).__name__
      else:
        exception_type = ""
      if "claude" in model: #one of the anthropics
        print_verbose(f"status_code: {original_exception.status_code}")
        if original_exception.status_code == 401:
          raise AuthenticationError(f"AnthropicException - {original_exception.message}")
        elif original_exception.status_code == 400:
          raise InvalidRequestError(f"AnthropicException - {original_exception.message}", f"{model}")
        elif original_exception.status_code == 429:
          raise RateLimitError(f"AnthropicException - {original_exception.message}")
      elif "replicate" in model:
        if "Incorrect authentication token" in error_str:
          raise AuthenticationError(f"ReplicateException - {error_str}")
        elif exception_type == "ModelError":
          raise InvalidRequestError(f"ReplicateException - {error_str}", f"{model}")
        elif "Request was throttled" in error_str:
          raise RateLimitError(f"ReplicateException - {error_str}")
        elif exception_type == "ReplicateError": ## ReplicateError implies an error on Replicate server side, not user side
          raise ServiceUnavailableError(f"ReplicateException - {error_str}")
      elif model == "command-nightly": #Cohere
        if "invalid api token" in error_str or "No API key provided." in error_str:
          raise AuthenticationError(f"CohereException - {error_str}")
        elif "too many tokens" in error_str:
          raise InvalidRequestError(f"CohereException - {error_str}", f"{model}")
        elif "CohereConnectionError" in exception_type: # cohere seems to fire these errors when we load test it (1k+ messages / min)
          raise RateLimitError(f"CohereException - {original_exception.message}")
      raise original_exception # base case - return the original exception
    else:
      raise original_exception
                        