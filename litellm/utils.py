import sys
import dotenv, json, traceback, threading
import subprocess, os 
import litellm, openai 
import random, uuid, requests
import datetime, time
import tiktoken
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
encoding = tiktoken.get_encoding("cl100k_base")
from .integrations.helicone import HeliconeLogger
from .integrations.aispend import AISpendLogger
from .integrations.berrispend import BerriSpendLogger
from .integrations.supabase import Supabase
from openai.error import AuthenticationError, InvalidRequestError, RateLimitError, ServiceUnavailableError, OpenAIError
####### ENVIRONMENT VARIABLES ###################
dotenv.load_dotenv() # Loading env variables using dotenv
sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
posthog = None
slack_app = None
alerts_channel = None
heliconeLogger = None
aispendLogger = None
berrispendLogger = None
supabaseClient = None
callback_list = []
user_logger_fn = None
additional_details = {}

def print_verbose(print_statement):
  if litellm.set_verbose:
    print(f"LiteLLM: {print_statement}")
    if random.random() <= 0.3:
      print("Get help - https://discord.com/invite/wuPM9dRgDw")

####### Package Import Handler ###################
import importlib
import subprocess
def install_and_import(package: str):
    if package in globals().keys():
      print_verbose(f"{package} has already been imported.")
      return
    try:
        # Import the module 
        module = importlib.import_module(package)
    except (ModuleNotFoundError, ImportError):
        print_verbose(f"{package} is not installed. Installing...")
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = importlib.import_module(package)
    except (DistributionNotFound, ImportError):
        print_verbose(f"{package} is not installed. Installing...")
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = importlib.import_module(package)
    except VersionConflict as vc:
        print_verbose(f"Detected version conflict for {package}. Upgrading...")
        subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", package])
        globals()[package] = importlib.import_module(package)
    finally:
        if package not in globals().keys():
            globals()[package] = importlib.import_module(package)
##################################################

####### LOGGING ###################
#Logging function -> log the exact model details + what's being sent | Non-Blocking
def logging(model=None, input=None, custom_llm_provider=None, azure=False, additional_args={}, logger_fn=None, exception=None):
  try:
    model_call_details = {}
    if model:
      model_call_details["model"] = model
    if azure:
      model_call_details["azure"] = azure
    if custom_llm_provider:
       model_call_details["custom_llm_provider"] = custom_llm_provider
    if exception:
      model_call_details["exception"] = exception
    if input:
      model_call_details["input"] = input
    
    if len(additional_args):
       model_call_details["additional_args"] = additional_args
    # log additional call details -> api key, etc. 
    if model:
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
    ## User Logging -> if you pass in a custom logging function or want to use sentry breadcrumbs
    print_verbose(f"Logging Details: logger_fn - {logger_fn} | callable(logger_fn) - {callable(logger_fn)}")
    if logger_fn and callable(logger_fn):
      try:
        logger_fn(model_call_details) # Expectation: any logger function passed in by the user should accept a dict object
      except Exception as e:
        print(f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}")
  except Exception as e:
    print(f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}")
    pass

####### CLIENT ################### 
# make it easy to log if completion/embedding runs succeeded or failed + see what happened | Non-Blocking
def client(original_function):
    def function_setup(*args, **kwargs): #just run once to check if user wants to send their data anywhere - PostHog/Sentry/Slack/etc.
      try: 
        global callback_list, add_breadcrumb, user_logger_fn
        if (len(litellm.success_callback) > 0 or len(litellm.failure_callback) > 0) and len(callback_list) == 0: 
          callback_list = list(set(litellm.success_callback + litellm.failure_callback))
          set_callbacks(callback_list=callback_list,)
        if add_breadcrumb:
          add_breadcrumb(
                category="litellm.llm_call",
                message=f"Positional Args: {args}, Keyword Args: {kwargs}",
                level="info",
            )
        if "logger_fn" in kwargs:
           user_logger_fn = kwargs["logger_fn"]
      except: # DO NOT BLOCK running the function because of this
        print_verbose(f"[Non-Blocking] {traceback.format_exc()}")
      pass

    def crash_reporting(*args, **kwargs):
      if litellm.telemetry:
        try:
          model = args[0] if len(args) > 0 else kwargs["model"]
          exception = kwargs["exception"] if "exception" in kwargs else None
          custom_llm_provider = kwargs["custom_llm_provider"] if "custom_llm_provider" in kwargs else None
          safe_crash_reporting(model=model, exception=exception, custom_llm_provider=custom_llm_provider) # log usage-crash details. Do not log any user details. If you want to turn this off, set `litellm.telemetry=False`.
        except:
           #[Non-Blocking Error]
           pass

    def wrapper(*args, **kwargs):
        start_time = None
        try:
          function_setup(*args, **kwargs)
          ## MODEL CALL
          start_time = datetime.datetime.now()
          result = original_function(*args, **kwargs)
          end_time = datetime.datetime.now()
          ## LOG SUCCESS 
          crash_reporting(*args, **kwargs)
          my_thread = threading.Thread(target=handle_success, args=(args, kwargs, result, start_time, end_time)) # don't interrupt execution of main thread
          my_thread.start()
          return result
        except Exception as e:
          traceback_exception = traceback.format_exc()
          crash_reporting(*args, **kwargs, exception=traceback_exception)
          end_time = datetime.datetime.now()
          my_thread = threading.Thread(target=handle_failure, args=(e, traceback_exception, start_time, end_time, args, kwargs)) # don't interrupt execution of main thread
          my_thread.start()
          raise e
    return wrapper

####### USAGE CALCULATOR ################

def token_counter(model, text):
  # use tiktoken or anthropic's tokenizer depending on the model
  num_tokens = 0
  if "claude" in model:
    install_and_import('anthropic')
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    anthropic = Anthropic()
    num_tokens = anthropic.count_tokens(text)
  else:
    num_tokens = len(encoding.encode(text))
  return num_tokens


def cost_per_token(model="gpt-3.5-turbo", prompt_tokens = 0, completion_tokens = 0):
   ## given 
  prompt_tokens_cost_usd_dollar = 0
  completion_tokens_cost_usd_dollar = 0
  model_cost_ref = litellm.model_cost
  if model in model_cost_ref:
    prompt_tokens_cost_usd_dollar = model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
    completion_tokens_cost_usd_dollar = model_cost_ref[model]["output_cost_per_token"] * completion_tokens
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
   prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(model=model, prompt_tokens = prompt_tokens, completion_tokens = completion_tokens)
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
    custom_api_base=None
): 
    litellm_params = {
        "return_async": return_async,
        "api_key": api_key,
        "force_timeout": force_timeout,
        "logger_fn": logger_fn,
        "verbose": verbose,
        "custom_llm_provider": custom_llm_provider,
        "custom_api_base": custom_api_base
    }
    
    return litellm_params


def get_optional_params(
    # 12 optional params
    functions = [],
    function_call = "",
    temperature = 1,
    top_p = 1,
    n = 1,
    stream = False,
    stop = None,
    max_tokens = float('inf'),
    presence_penalty = 0,
    frequency_penalty = 0,
    logit_bias = {},
    user = "",
    deployment_id = None,
    model = None,
    custom_llm_provider = "",
    top_k = 40,
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
    if max_tokens != float('inf'):
        optional_params["max_tokens"] = max_tokens
    return optional_params
  elif custom_llm_provider == "replicate":
    # any replicate models
    # TODO: handle translating remaining replicate params
    if stream:
      optional_params["stream"] = stream
      return optional_params
  elif custom_llm_provider == "together_ai":
      if stream:
        optional_params["stream_tokens"] = stream
      if temperature != 1:
          optional_params["temperature"] = temperature
      if top_p != 1:
        optional_params["top_p"] = top_p
      if max_tokens != float('inf'):
        optional_params["max_tokens"] = max_tokens
      if frequency_penalty != 0:
        optional_params["frequency_penalty"] = frequency_penalty
  elif model == "chat-bison": # chat-bison has diff args from chat-bison@001 ty Google
     if temperature != 1:
        optional_params["temperature"] = temperature
     if top_p != 1:
        optional_params["top_p"] = top_p
     if max_tokens != float('inf'):
        optional_params["max_output_tokens"] = max_tokens
  elif model in litellm.vertex_text_models:
      # required params for all text vertex calls
      # temperature=0.2, top_p=0.1, top_k=20
      # always set temperature, top_p, top_k else, text bison fails
      optional_params["temperature"] = temperature
      optional_params["top_p"] = top_p
      optional_params["top_k"] = top_k

  else:# assume passing in params for openai/azure openai
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
    if max_tokens != float('inf'):
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

def load_test_model(model: str, custom_llm_provider: str = None, custom_api_base: str = None, prompt: str = None, num_calls: int = None, force_timeout: int = None):
  test_prompt = "Hey, how's it going"
  test_calls = 100
  if prompt:
     test_prompt = prompt
  if num_calls:
     test_calls = num_calls
  messages = [[{"role": "user", "content": test_prompt}] for _ in range(test_calls)]
  start_time = time.time()
  try:
    litellm.batch_completion(model=model, messages=messages, custom_llm_provider=custom_llm_provider, custom_api_base = custom_api_base, force_timeout=force_timeout)
    end_time = time.time() 
    response_time = end_time - start_time
    return {"total_response_time": response_time, "calls_made": 100, "status": "success", "exception": None}
  except Exception as e:
    end_time = time.time() 
    response_time = end_time - start_time
    return {"total_response_time": response_time, "calls_made": 100, "status": "failed", "exception": e}

def set_callbacks(callback_list):
  global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, heliconeLogger, aispendLogger, berrispendLogger, supabaseClient
  try:
    for callback in callback_list:
      if callback == "sentry":
        try:
            import sentry_sdk
        except ImportError:
            print_verbose("Package 'sentry_sdk' is missing. Installing it...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentry_sdk'])
            import sentry_sdk
        sentry_sdk_instance = sentry_sdk
        sentry_trace_rate = os.environ.get("SENTRY_API_TRACE_RATE") if "SENTRY_API_TRACE_RATE" in os.environ else "1.0"
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
      elif callback == "helicone":
        heliconeLogger = HeliconeLogger()
      elif callback == "aispend":
        aispendLogger = AISpendLogger()
      elif callback == "berrispend": 
        berrispendLogger = BerriSpendLogger()
      elif callback == "supabase":
         supabaseClient = Supabase()
  except Exception as e:
    raise e


def handle_failure(exception, traceback_exception, start_time, end_time, args, kwargs):
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, aispendLogger, berrispendLogger
    try:
      # print_verbose(f"handle_failure args: {args}")
      # print_verbose(f"handle_failure kwargs: {kwargs}")
      
      success_handler = additional_details.pop("success_handler", None)
      failure_handler = additional_details.pop("failure_handler", None)
      
      additional_details["Event_Name"] = additional_details.pop("failed_event_name", "litellm.failed_query")
      print_verbose(f"self.failure_callback: {litellm.failure_callback}")


      # print_verbose(f"additional_details: {additional_details}")
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
          elif callback == "berrispend": 
              print_verbose("reaches berrispend for logging!")
              model = args[0] if len(args) > 0 else kwargs["model"]
              messages = args[1] if len(args) > 1 else kwargs["messages"]
              result = {
                 "model": model,
                 "created": time.time(),
                 "error": traceback_exception,
                 "usage": {
                    "prompt_tokens": prompt_token_calculator(model, messages=messages),
                    "completion_tokens": 0
                 }
              }
              berrispendLogger.log_event(model=model, messages=messages, response_obj=result, start_time=start_time, end_time=end_time, print_verbose=print_verbose)
          elif callback == "aispend":
              print_verbose("reaches aispend for logging!")
              model = args[0] if len(args) > 0 else kwargs["model"]
              messages = args[1] if len(args) > 1 else kwargs["messages"]
              result = {
                 "model": model,
                 "created": time.time(),
                 "usage": {
                    "prompt_tokens": prompt_token_calculator(model, messages=messages),
                    "completion_tokens": 0
                 }
              }
              aispendLogger.log_event(model=model, response_obj=result, start_time=start_time, end_time=end_time, print_verbose=print_verbose)
          elif callback == "supabase":
              print_verbose("reaches supabase for logging!")
              model = args[0] if len(args) > 0 else kwargs["model"]
              messages = args[1] if len(args) > 1 else kwargs["messages"]
              result = {
                 "model": model,
                 "created": time.time(),
                 "error": traceback_exception,
                 "usage": {
                    "prompt_tokens": prompt_token_calculator(model, messages=messages),
                    "completion_tokens": 0
                 }
              }
              print(f"litellm._thread_context: {litellm._thread_context}")
              supabaseClient.log_event(model=model, messages=messages, end_user=litellm._thread_context.user, response_obj=result, start_time=start_time, end_time=end_time, print_verbose=print_verbose)

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
    except Exception as e:
      ## LOGGING
      logging(logger_fn=user_logger_fn, exception=e)
      pass

def handle_success(args, kwargs, result, start_time, end_time):
  global heliconeLogger, aispendLogger
  try:
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
        elif callback == "helicone":
          print_verbose("reaches helicone for logging!")
          model = args[0] if len(args) > 0 else kwargs["model"]
          messages = args[1] if len(args) > 1 else kwargs["messages"]
          heliconeLogger.log_success(model=model, messages=messages, response_obj=result, start_time=start_time, end_time=end_time, print_verbose=print_verbose)
        elif callback == "aispend":
          print_verbose("reaches aispend for logging!")
          model = args[0] if len(args) > 0 else kwargs["model"]
          aispendLogger.log_event(model=model, response_obj=result, start_time=start_time, end_time=end_time, print_verbose=print_verbose)
        elif callback == "berrispend":
          print_verbose("reaches berrispend for logging!")
          model = args[0] if len(args) > 0 else kwargs["model"]
          messages = args[1] if len(args) > 1 else kwargs["messages"]
          berrispendLogger.log_event(model=model, messages=messages, response_obj=result, start_time=start_time, end_time=end_time, print_verbose=print_verbose)
        elif callback == "supabase":
          print_verbose("reaches supabase for logging!")
          model = args[0] if len(args) > 0 else kwargs["model"]
          messages = args[1] if len(args) > 1 else kwargs["messages"]
          print(f"litellm._thread_context: {litellm._thread_context}")
          supabaseClient.log_event(model=model, messages=messages, end_user=litellm._thread_context.user, response_obj=result, start_time=start_time, end_time=end_time, print_verbose=print_verbose)
      except Exception as e:
        ## LOGGING
        logging(logger_fn=user_logger_fn, exception=e)
        print_verbose(f"[Non-Blocking] Success Callback Error - {traceback.format_exc()}")
        pass

    if success_handler and callable(success_handler):
      success_handler(args, kwargs)
    pass
  except Exception as e:
    ## LOGGING
    logging(logger_fn=user_logger_fn, exception=e)
    print_verbose(f"[Non-Blocking] Success Callback Error - {traceback.format_exc()}")
    pass

def prompt_token_calculator(model, messages):
  # use tiktoken or anthropic's tokenizer depending on the model
  text = " ".join(message["content"] for message in messages)
  num_tokens = 0
  if "claude" in model:
    install_and_import('anthropic')
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    anthropic = Anthropic()
    num_tokens = anthropic.count_tokens(text)
  else:
    num_tokens = len(encoding.encode(text))
  return num_tokens

# integration helper function 
def modify_integration(integration_name, integration_params):
   global supabaseClient
   if integration_name == "supabase":
      if "table_name" in integration_params:
         Supabase.supabase_table_name = integration_params["table_name"]

def exception_type(model, original_exception):
    global user_logger_fn
    exception_mapping_worked = False
    try:
      if isinstance(original_exception, OpenAIError):
          # Handle the OpenAIError
          raise original_exception
      elif model:
        error_str = str(original_exception)
        if isinstance(original_exception, BaseException):
          exception_type = type(original_exception).__name__
        else:
          exception_type = ""
        logging(model=model, additional_args={"error_str": error_str, "exception_type": exception_type, "original_exception": original_exception}, logger_fn=user_logger_fn)
        if "claude" in model: #one of the anthropics
          if hasattr(original_exception, "status_code"):
            print_verbose(f"status_code: {original_exception.status_code}")
            if original_exception.status_code == 401:
              exception_mapping_worked = True
              raise AuthenticationError(f"AnthropicException - {original_exception.message}")
            elif original_exception.status_code == 400:
              exception_mapping_worked = True
              raise InvalidRequestError(f"AnthropicException - {original_exception.message}", f"{model}")
            elif original_exception.status_code == 429:
              exception_mapping_worked = True
              raise RateLimitError(f"AnthropicException - {original_exception.message}")
          elif "Could not resolve authentication method. Expected either api_key or auth_token to be set." in error_str:
            exception_mapping_worked = True
            raise AuthenticationError(f"AnthropicException - {error_str}")
        elif "replicate" in model:
          if "Incorrect authentication token" in error_str:
            exception_mapping_worked = True
            raise AuthenticationError(f"ReplicateException - {error_str}")
          elif exception_type == "ModelError":
            exception_mapping_worked = True
            raise InvalidRequestError(f"ReplicateException - {error_str}", f"{model}")
          elif "Request was throttled" in error_str:
            exception_mapping_worked = True
            raise RateLimitError(f"ReplicateException - {error_str}")
          elif exception_type == "ReplicateError": ## ReplicateError implies an error on Replicate server side, not user side
            raise ServiceUnavailableError(f"ReplicateException - {error_str}")
        elif model == "command-nightly": #Cohere
          if "invalid api token" in error_str or "No API key provided." in error_str:
            exception_mapping_worked = True
            raise AuthenticationError(f"CohereException - {error_str}")
          elif "too many tokens" in error_str:
            exception_mapping_worked = True
            raise InvalidRequestError(f"CohereException - {error_str}", f"{model}")
          elif "CohereConnectionError" in exception_type: # cohere seems to fire these errors when we load test it (1k+ messages / min)
            exception_mapping_worked = True
            raise RateLimitError(f"CohereException - {original_exception.message}")
        raise original_exception # base case - return the original exception
      else:
        raise original_exception
    except Exception as e:
      ## LOGGING
      logging(logger_fn=user_logger_fn, additional_args={"exception_mapping_worked": exception_mapping_worked, "original_exception": original_exception}, exception=e) 
      if exception_mapping_worked:
        raise e
      else: # don't let an error with mapping interrupt the user from receiving an error from the llm api calls 
         raise original_exception

def safe_crash_reporting(model=None, exception=None, custom_llm_provider=None):
    data = {
      "model": model,
      "exception": str(exception),
      "custom_llm_provider": custom_llm_provider
    }
    threading.Thread(target=litellm_telemetry, args=(data,)).start()

def litellm_telemetry(data):
    # Load or generate the UUID
    uuid_file = 'litellm_uuid.txt'
    try:
        # Try to open the file and load the UUID
        with open(uuid_file, 'r') as file:
            uuid_value = file.read()
            if uuid_value:
                uuid_value = uuid_value.strip()
            else:
                raise FileNotFoundError
    except FileNotFoundError:
        # Generate a new UUID if the file doesn't exist or is empty
        new_uuid = uuid.uuid4()
        uuid_value = str(new_uuid)
        with open(uuid_file, 'w') as file:
            file.write(uuid_value)
    except:
      # [Non-Blocking Error]
      return
    
    try:
      # Prepare the data to send to litellm logging api
      payload = {
          'uuid': uuid_value,
          'data': data,
          'version': pkg_resources.get_distribution("litellm").version
      }
      # Make the POST request to litellm logging api
      response = requests.post('https://litellm.berri.ai/logging', headers={"Content-Type": "application/json"}, json=payload)
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
     secret = litellm.secret_manager_client.get_secret(secret_name).secret_value
     if secret != None:
        # if secret manager fails default to using .env variables
        os.environ[secret_name] = secret # set to env to be safe
        return secret
     else:
      return os.environ.get(secret_name)
  else:
    return os.environ.get(secret_name)

######## Streaming Class ############################
# wraps the completion stream to return the correct format for the model
# replicate/anthropic/cohere
class CustomStreamWrapper:
    def __init__(self, completion_stream, model):
        self.model = model
        if model in litellm.cohere_models:
           # cohere does not return an iterator, so we need to wrap it in one
           self.completion_stream = iter(completion_stream)
        elif model == "together_ai":
            self.completion_stream = iter(completion_stream)
        else: 
          self.completion_stream = completion_stream

    def __iter__(self):
        return self

    def handle_anthropic_chunk(self, chunk):
      str_line = chunk.decode('utf-8')  # Convert bytes to string
      if str_line.startswith('data:'):
          data_json = json.loads(str_line[5:])
          return data_json.get("completion", "")
      return ""

    def handle_together_ai_chunk(self, chunk): 
      chunk = chunk.decode("utf-8")
      text_index = chunk.find('"text":"') # this checks if text: exists
      text_start = text_index + len('"text":"')
      text_end = chunk.find('"}', text_start)
      if text_index != -1 and text_end != -1: 
          extracted_text = chunk[text_start:text_end]
          return extracted_text
      else:
          return ""

    def __next__(self):
        completion_obj ={ "role": "assistant", "content": ""}
        if self.model in litellm.anthropic_models:
          chunk = next(self.completion_stream)
          completion_obj["content"] = self.handle_anthropic_chunk(chunk)
        elif self.model == "replicate":
           chunk = next(self.completion_stream)
           completion_obj["content"] = chunk
        elif self.model == "together_ai":
          chunk = next(self.completion_stream)
          text_data =  self.handle_together_ai_chunk(chunk)
          if text_data == "":
             return self.__next__()
          completion_obj["content"] = text_data
        elif self.model in litellm.cohere_models:
          chunk = next(self.completion_stream)
          completion_obj["content"] = chunk.text
        # return this for all models
        return {"choices": [{"delta": completion_obj}]}



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
import aiohttp
async def get_ollama_response_stream(api_base="http://localhost:11434", model="llama2", prompt="Why is the sky blue?"):
    session = aiohttp.ClientSession()
    url = f'{api_base}/api/generate'
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
                                    completion_obj ={ "role": "assistant", "content": ""}
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

   
