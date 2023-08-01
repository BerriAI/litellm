import os, openai, cohere, replicate, sys
from typing import Any
from func_timeout import func_set_timeout, FunctionTimedOut
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json
import traceback
import threading
import dotenv
import traceback
import subprocess
import uuid

####### ENVIRONMENT VARIABLES ###################
dotenv.load_dotenv() # Loading env variables using dotenv
set_verbose = False
sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
posthog = None
slack_app = None
alerts_channel = None
success_callback = []
failure_callback = []
callback_list = []
user_logger_fn = None
additional_details = {}

## Set verbose to true -> ```litellm.verbose = True```    
def print_verbose(print_statement):
  if set_verbose:
    print(f"LiteLLM: {print_statement}")
    print("Get help - https://discord.com/invite/wuPM9dRgDw")

####### COMPLETION MODELS ###################
open_ai_chat_completion_models = [
  'gpt-3.5-turbo', 
  'gpt-4'
]
open_ai_text_completion_models = [
    'text-davinci-003'
]

cohere_models = [
    'command-nightly',
]

anthropic_models = [
  "claude-2", 
  "claude-instant-1"
]

####### EMBEDDING MODELS ###################
open_ai_embedding_models = [
    'text-embedding-ada-002'
]

####### CLIENT ################### make it easy to log completion/embedding runs
def client(original_function):
    def function_setup(): #just run once to check if user wants to send their data anywhere
      try: 
        if len(success_callback) > 0 or len(failure_callback) > 0 and len(callback_list) == 0: 
          callback_list = list(set(success_callback + failure_callback))
          set_callbacks(callback_list=callback_list)
      except: # DO NOT BLOCK running the function because of this
        print_verbose(f"[Non-Blocking] {traceback.format_exc()}")
      pass

    def wrapper(*args, **kwargs):
        # Code to be executed before the embedding function
        try:
          function_setup()
          ## EMBEDDING CALL
          result = original_function(*args, **kwargs)
          ## LOG SUCCESS 
          my_thread = threading.Thread(target=handle_success, args=(args, kwargs)) # don't interrupt execution of main thread
          my_thread.start()
          return result
        except Exception as e:
          traceback_exception = traceback.format_exc()
          my_thread = threading.Thread(target=handle_failure, args=(e, traceback.format_exc(), args, kwargs)) # don't interrupt execution of main thread
          my_thread.start()
          raise e
    return wrapper


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
):
  optional_params = {}
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
  return optional_params

####### COMPLETION ENDPOINTS ################
#############################################
@client
@func_set_timeout(120, allowOverride=True) ## https://pypi.org/project/func-timeout/ - timeouts, in case calls hang (e.g. Azure)
def completion(
    model, messages, # required params
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    functions=[], function_call="", # optional params
    temperature=1, top_p=1, n=1, stream=False, stop=None, max_tokens=float('inf'),
    presence_penalty=0, frequency_penalty=0, logit_bias={}, user="",
    # Optional liteLLM function params
    *, forceTimeout=60, azure=False, logger_fn=None, verbose=False
  ):
  try:
    # check if user passed in any of the OpenAI optional params
    optional_params = get_optional_params(
      functions=functions, function_call=function_call, 
      temperature=temperature, top_p=top_p, n=n, stream=stream, stop=stop, max_tokens=max_tokens,
      presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, logit_bias=logit_bias, user=user
    )
    if azure == True:
      # azure configs
      openai.api_type = "azure"
      openai.api_base = os.environ.get("AZURE_API_BASE")
      openai.api_version = os.environ.get("AZURE_API_VERSION")
      openai.api_key = os.environ.get("AZURE_API_KEY")
      ## LOGGING
      logging(model=model, input=messages, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      response = openai.ChatCompletion.create(
        engine=model,
        messages = messages,
        **optional_params
      )
    elif model in open_ai_chat_completion_models:
      openai.api_type = "openai"
      openai.api_base = "https://api.openai.com/v1"
      openai.api_version = None
      openai.api_key = os.environ.get("OPENAI_API_KEY")
      ## LOGGING
      logging(model=model, input=messages, azure=azure, logger_fn=logger_fn)

      ## COMPLETION CALL
      response = openai.ChatCompletion.create(
        model=model,
        messages = messages,
        **optional_params
      )
    elif model in open_ai_text_completion_models:
      openai.api_type = "openai"
      openai.api_base = "https://api.openai.com/v1"
      openai.api_version = None
      openai.api_key = os.environ.get("OPENAI_API_KEY")
      prompt = " ".join([message["content"] for message in messages])
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      response = openai.Completion.create(
          model=model,
          prompt = prompt
      )
    elif "replicate" in model:
      # replicate defaults to os.environ.get("REPLICATE_API_TOKEN")
      # checking in case user set it to REPLICATE_API_KEY instead 
      if not os.environ.get("REPLICATE_API_TOKEN") and  os.environ.get("REPLICATE_API_KEY"):
        replicate_api_token = os.environ.get("REPLICATE_API_KEY")
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
      prompt = " ".join([message["content"] for message in messages])
      input = {"prompt": prompt}
      if max_tokens != float('inf'):
        input["max_length"] = max_tokens # for t5 models 
        input["max_new_tokens"] = max_tokens # for llama2 models 
      ## LOGGING
      logging(model=model, input=input, azure=azure, additional_args={"max_tokens": max_tokens}, logger_fn=logger_fn)
      ## COMPLETION CALL
      output = replicate.run(
        model,
        input=input)
      response = ""
      for item in output: 
        response += item
      new_response = {
        "choices": [
          {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": response,
                "role": "assistant"
            }
          }
        ]
      }
      response = new_response
    elif model in anthropic_models:
      #anthropic defaults to os.environ.get("ANTHROPIC_API_KEY")
      prompt = f"{HUMAN_PROMPT}" 
      for message in messages:
        if "role" in message:
          if message["role"] == "user":
            prompt += f"{HUMAN_PROMPT}{message['content']}"
          else:
            prompt += f"{AI_PROMPT}{message['content']}"
        else:
          prompt += f"{HUMAN_PROMPT}{message['content']}"
      prompt += f"{AI_PROMPT}"
      anthropic = Anthropic()
      # check if user passed in max_tokens != float('inf')
      if max_tokens != float('inf'):
        max_tokens_to_sample = max_tokens
      else:
        max_tokens_to_sample = 300 # default in Anthropic docs https://docs.anthropic.com/claude/reference/client-libraries
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, additional_args={"max_tokens": max_tokens}, logger_fn=logger_fn)
      ## COMPLETION CALL
      completion = anthropic.completions.create(
          model=model,
          prompt=prompt,
          max_tokens_to_sample=max_tokens_to_sample
      )
      new_response = {
        "choices": [
          {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": completion.completion,
                "role": "assistant"
            }
          }
        ]
      }
      print_verbose(f"new response: {new_response}")
      response = new_response
    elif model in cohere_models:
      cohere_key = os.environ.get("COHERE_API_KEY")
      co = cohere.Client(cohere_key)
      prompt = " ".join([message["content"] for message in messages])
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      response = co.generate(  
        model=model,
        prompt = prompt
      )
      new_response = {
          "choices": [
              {
                  "finish_reason": "stop",
                  "index": 0,
                  "message": {
                      "content": response[0],
                      "role": "assistant"
                  }
              }
          ],
      }
      response = new_response
    else: 
      logging(model=model, input=messages, azure=azure, logger_fn=logger_fn)
    return response
  except Exception as e:
    logging(model=model, input=messages, azure=azure, additional_args={"max_tokens": max_tokens}, logger_fn=logger_fn)
    raise e


### EMBEDDING ENDPOINTS ####################
@client
@func_set_timeout(60, allowOverride=True) ## https://pypi.org/project/func-timeout/
def embedding(model, input=[], azure=False, forceTimeout=60, logger_fn=None):
  response = None
  if azure == True:
    # azure configs
    openai.api_type = "azure"
    openai.api_base = os.environ.get("AZURE_API_BASE")
    openai.api_version = os.environ.get("AZURE_API_VERSION")
    openai.api_key = os.environ.get("AZURE_API_KEY")
    ## LOGGING
    logging(model=model, input=input, azure=azure, logger_fn=logger_fn)
    ## EMBEDDING CALL
    response = openai.Embedding.create(input=input, engine=model)
    print_verbose(f"response_value: {str(response)[:50]}")
  elif model in open_ai_embedding_models:
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    ## LOGGING
    logging(model=model, input=input, azure=azure, logger_fn=logger_fn)
    ## EMBEDDING CALL
    response = openai.Embedding.create(input=input, model=model)
    print_verbose(f"response_value: {str(response)[:50]}")
  else: 
    logging(model=model, input=input, azure=azure, logger_fn=logger_fn)
    args = locals()
    raise ValueError(f"No valid embedding model args passed in - {args}")
  
  return response


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
    print_verbose(f"handle_failure args: {args}")
    print_verbose(f"handle_failure kwargs: {kwargs}")
    
    success_handler = additional_details.pop("success_handler", None)
    failure_handler = additional_details.pop("failure_handler", None)
    
    additional_details["Event_Name"] = additional_details.pop("failed_event_name", "litellm.failed_query")
    print_verbose(f"self.failure_callback: {failure_callback}")

    print_verbose(f"additional_details: {additional_details}")
    for callback in failure_callback:
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
          print_verbose(f"This is the slack message: {slack_msg}")
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
          print_verbose(f"ph_obj: {ph_obj}")
          for detail in additional_details:
            ph_obj[detail] = additional_details[detail]
          event_name = additional_details["Event_Name"]
          print_verbose(f"PostHog Event Name: {event_name}")
          if "user_id" in additional_details:
            posthog.capture(additional_details["user_id"], event_name, ph_obj)
          else: # PostHog calls require a unique id to identify a user - https://posthog.com/docs/libraries/python
            print(f"ph_obj: {ph_obj})")
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


def handle_input(model_call_details={}):
      if len(model_call_details.keys()) > 0:
        model = model_call_details["model"] if "model" in model_call_details else None
        if model:
          for callback in callback_list:
            if callback == "sentry": # add a sentry breadcrumb if user passed in sentry integration
              add_breadcrumb(
                category=f'{model}',
                message='Trying request model {} input {}'.format(model, json.dumps(model_call_details)),
                level='info',
              )
          if user_logger_fn and callable(user_logger_fn):
            user_logger_fn(model_call_details)
      pass

def handle_success(*args, **kwargs):
  success_handler = additional_details.pop("success_handler", None)
  failure_handler = additional_details.pop("failure_handler", None)
  additional_details["Event_Name"] = additional_details.pop("successful_event_name", "litellm.succes_query")
  for callback in success_callback:
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

#Logging function -> log the exact model details + what's being sent | Non-Blocking
def logging(model, input, azure=False, additional_args={}, logger_fn=None):
  try:
    model_call_details = {}
    model_call_details["model"] = model
    model_call_details["input"] = input
    model_call_details["azure"] = azure
    # log additional call details -> api key, etc. 
    if azure == True or model in open_ai_chat_completion_models or model in open_ai_chat_completion_models or model in open_ai_embedding_models:
      model_call_details["api_type"] = openai.api_type
      model_call_details["api_base"] = openai.api_base
      model_call_details["api_version"] = openai.api_version
      model_call_details["api_key"] = openai.api_key
    elif "replicate" in model:
      model_call_details["api_key"] = os.environ.get("REPLICATE_API_TOKEN")
    elif model in anthropic_models:
      model_call_details["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
    elif model in cohere_models:
      model_call_details["api_key"] = os.environ.get("COHERE_API_KEY")
    model_call_details["additional_args"] = additional_args
    ## Logging
    print_verbose(f"Basic model call details: {model_call_details}")
    if logger_fn and callable(logger_fn):
      try:
        logger_fn(model_call_details) # Expectation: any logger function passed in by the user should accept a dict object
      except:
        print_verbose(f"[Non-Blocking] Exception occurred while logging {traceback.format_exc()}")
        pass
  except:
    pass
