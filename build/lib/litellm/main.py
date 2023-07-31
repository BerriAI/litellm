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
####### ENVIRONMENT VARIABLES ###################
# Loading env variables using dotenv
dotenv.load_dotenv()
set_verbose = False

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

#############################################


####### COMPLETION ENDPOINTS ################
#############################################
@func_set_timeout(10, allowOverride=True) ## https://pypi.org/project/func-timeout/ - timeouts, in case calls hang (e.g. Azure)
def completion(model, messages, max_tokens=None, forceTimeout=10, azure=False, logger_fn=None):
  try:
    if azure == True:
      # azure configs
      openai.api_type = "azure"
      openai.api_base = os.environ.get("AZURE_API_BASE")
      openai.api_version = os.environ.get("AZURE_API_VERSION")
      openai.api_key = os.environ.get("AZURE_API_KEY")
      ## LOGGING
      logging(model=model, input=input, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      response = openai.ChatCompletion.create(
        engine=model,
        messages = messages
      )
    elif "replicate" in model: 
      # replicate defaults to os.environ.get("REPLICATE_API_TOKEN")
      # checking in case user set it to REPLICATE_API_KEY instead 
      if not os.environ.get("REPLICATE_API_TOKEN") and  os.environ.get("REPLICATE_API_KEY"):
        replicate_api_token = os.environ.get("REPLICATE_API_KEY")
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
      prompt = " ".join([message["content"] for message in messages])
      input = [{"prompt": prompt}]
      if max_tokens:
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
      if max_tokens:
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
      print(f"new response: {new_response}")
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
          messages = messages
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
    else: 
      logging(model=model, input=messages, azure=azure, logger_fn=logger_fn)
    return response
  except Exception as e:
    logging(model=model, input=messages, azure=azure, additional_args={"max_tokens": max_tokens}, logger_fn=logger_fn)
    raise e


### EMBEDDING ENDPOINTS ####################
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
  
  return response


### CLIENT CLASS #################### make it easy to push completion/embedding runs to different sources -> sentry/posthog/slack, etc.
class litellm_client:
  def __init__(self, success_callback=[], failure_callback=[], verbose=False):  # Constructor
      set_verbose = verbose
      self.success_callback = success_callback
      self.failure_callback = failure_callback
      self.logger_fn = None # if user passes in their own logging function
      self.callback_list = list(set(self.success_callback + self.failure_callback))
      self.set_callbacks()
  
  ## COMPLETION CALL 
  def completion(self, model, messages, max_tokens=None, forceTimeout=10, azure=False, logger_fn=None, additional_details={}) -> Any:
    try:
      self.logger_fn = logger_fn
      response = completion(model=model, messages=messages, max_tokens=max_tokens, forceTimeout=forceTimeout, azure=azure, logger_fn=self.handle_input)
      my_thread = threading.Thread(target=self.handle_success, args=(model, messages, additional_details)) # don't interrupt execution of main thread
      my_thread.start()
      return response
    except Exception as e: 
      args = locals() # get all the param values
      self.handle_failure(e, args)
      raise e

  ## EMBEDDING CALL 
  def embedding(self, model, input=[], azure=False, logger_fn=None, forceTimeout=60, additional_details={}) -> Any:
    try:
      self.logger_fn = logger_fn
      response = embedding(model, input, azure=azure, logger_fn=self.handle_input)
      my_thread = threading.Thread(target=self.handle_success, args=(model, input, additional_details)) # don't interrupt execution of main thread
      my_thread.start()
      return response
    except Exception as e:
      args = locals() # get all the param values 
      self.handle_failure(e, args)
      raise e


  def set_callbacks(self):  #instantiate any external packages
    for callback in self.callback_list: # only install what's required
      if callback == "sentry":
        try:
          import sentry_sdk
        except ImportError:
          print_verbose("Package 'sentry_sdk' is missing. Installing it...")
          subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentry_sdk'])
          import sentry_sdk
        self.sentry_sdk = sentry_sdk
        self.sentry_sdk.init(dsn=os.environ.get("SENTRY_API_URL"), traces_sample_rate=float(os.environ.get("SENTRY_API_TRACE_RATE")))
        self.capture_exception = self.sentry_sdk.capture_exception
        self.add_breadcrumb = self.sentry_sdk.add_breadcrumb
      elif callback == "posthog":
        try:
          from posthog import Posthog
        except:
          print_verbose("Package 'posthog' is missing. Installing it...")
          subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'posthog'])
          from posthog import Posthog
        self.posthog = Posthog(
            project_api_key=os.environ.get("POSTHOG_API_KEY"),
            host=os.environ.get("POSTHOG_API_URL"))
      elif callback == "slack":
        try:
          from slack_bolt import App
        except ImportError:
          print_verbose("Package 'slack_bolt' is missing. Installing it...")
          subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'slack_bolt'])
          from slack_bolt import App
        self.slack_app = App(
          token=os.environ.get("SLACK_API_TOKEN"),
          signing_secret=os.environ.get("SLACK_API_SECRET")
        )
        self.alerts_channel = os.environ["SLACK_API_CHANNEL"]

  def handle_input(self, model_call_details={}):
      if len(model_call_details.keys()) > 0:
        model = model_call_details["model"] if "model" in model_call_details else None
        if model:
          for callback in self.callback_list:
            if callback == "sentry": # add a sentry breadcrumb if user passed in sentry integration
              self.add_breadcrumb(
                category=f'{model}',
                message='Trying request model {} input {}'.format(model, json.dumps(model_call_details)),
                level='info',
              )
          if self.logger_fn and callable(self.logger_fn):
            self.logger_fn(model_call_details)
      pass

  def handle_success(self, model, messages, additional_details):
    success_handler = additional_details.pop("success_handler", None)
    failure_handler = additional_details.pop("failure_handler", None)
    additional_details["litellm_model"] = str(model)
    additional_details["litellm_messages"] = str(messages)
    for callback in self.success_callback:
      try:
        if callback == "posthog":
          ph_obj = {}
          for detail in additional_details:
            ph_obj[detail] = additional_details[detail]
          event_name = additional_details["successful_event"] if "successful_event" in additional_details else "litellm.succes_query"
          if "user_id" in additional_details:
            self.posthog.capture(additional_details["user_id"], event_name, ph_obj)
          else: 
            self.posthog.capture(event_name, ph_obj)
          pass
        elif callback == "slack":
          slack_msg = "" 
          if len(additional_details.keys()) > 0:
            for detail in additional_details: 
              slack_msg += f"{detail}: {additional_details[detail]}\n"
          slack_msg += f"Successful call"
          self.slack_app.client.chat_postMessage(channel=self.alerts_channel, text=slack_msg)
      except:
        pass
    
    if success_handler and callable(success_handler):
      call_details = {
        "model": model,
        "messages": messages,
        "additional_details": additional_details
      }
      success_handler(call_details)
    pass

  def handle_failure(self, exception, args):
    args.pop("self")
    additional_details = args.pop("additional_details", {})

    success_handler = additional_details.pop("success_handler", None)
    failure_handler = additional_details.pop("failure_handler", None)

    for callback in self.failure_callback:
      try:
        if callback == "slack":
          slack_msg = "" 
          for param in args: 
            slack_msg += f"{param}: {args[param]}\n"
          if len(additional_details.keys()) > 0:
            for detail in additional_details: 
              slack_msg += f"{detail}: {additional_details[detail]}\n"
          slack_msg += f"Traceback: {traceback.format_exc()}"
          self.slack_app.client.chat_postMessage(channel=self.alerts_channel, text=slack_msg)
        elif callback == "sentry":
          self.capture_exception(exception)
        elif callback == "posthog":
          if len(additional_details.keys()) > 0:
            ph_obj = {}
            for param in args: 
              ph_obj[param] += args[param]
            for detail in additional_details:
              ph_obj[detail] = additional_details[detail]
            event_name = additional_details["failed_event"] if "failed_event" in additional_details else "litellm.failed_query"
            if "user_id" in additional_details:
              self.posthog.capture(additional_details["user_id"], event_name, ph_obj)
            else: 
              self.posthog.capture(event_name, ph_obj)
          else: 
            pass
      except:
        print(f"got an error calling {callback} - {traceback.format_exc()}")
    
    if failure_handler and callable(failure_handler):
      call_details = {
        "exception": exception,
        "additional_details": additional_details
      }
      failure_handler(call_details)
    pass
####### HELPER FUNCTIONS ################

#Logging function -> log the exact model details + what's being sent | Non-Blocking
def logging(model, input, azure=False, additional_args={}, logger_fn=None):
  try:
    model_call_details = {}
    model_call_details["model"] = model
    model_call_details["input"] = input
    model_call_details["azure"] = azure
    model_call_details["additional_args"] = additional_args
    if logger_fn and callable(logger_fn):
      try:
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
        
        logger_fn(model_call_details) # Expectation: any logger function passed in by the user should accept a dict object
      except:
        print_verbose(f"Basic model call details: {model_call_details}")
        print_verbose(f"[Non-Blocking] Exception occurred while logging {traceback.format_exc()}")
        pass
    else:
      print_verbose(f"Basic model call details: {model_call_details}")
      pass
  except:
    pass

## Set verbose to true -> ```litellm.verbose = True```    
def print_verbose(print_statement):
  if set_verbose:
    print(f"LiteLLM: {print_statement}")
    print("Get help - https://discord.com/invite/wuPM9dRgDw")