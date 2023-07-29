import os, openai, cohere, replicate
from typing import Any
from func_timeout import func_set_timeout, FunctionTimedOut
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import sentry_sdk
from sentry_sdk import capture_exception, add_breadcrumb
from posthog import Posthog
from slack_bolt import App
import json
import traceback
import threading
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
@func_set_timeout(10, allowOverride=True)
def completion(model, messages, max_tokens=300, forceTimeout=10, azure=False):
  if azure == True:
    # azure configs
    openai.api_type = "azure"
    openai.api_base = os.environ.get("AZURE_API_BASE")
    openai.api_version = os.environ.get("AZURE_API_VERSION")
    openai.api_key =   os.environ.get("AZURE_API_KEY")
    response = openai.ChatCompletion.create(
      engine=model,
      messages = messages
    )
  elif "replicate" in model: 
    prompt = " ".join([message["content"] for message in messages])
    output = replicate.run(
      model,
      input={
        "prompt": prompt,
      })
    print(f"output: {output}")
    response = ""
    for item in output: 
      print(f"item: {item}")
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
    print(f"new response: {new_response}")
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
    completion = anthropic.completions.create(
        model=model,
        prompt=prompt,
        max_tokens_to_sample=max_tokens
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
    response = openai.Completion.create(
        model=model,
        prompt = prompt
    )
  return response



### EMBEDDING ENDPOINTS ####################
def embedding(model, input=[], azure=False, logger_fn=None):
  if azure == True:
    # azure configs
    openai.api_type = "azure"
    openai.api_base = os.environ.get("AZURE_API_BASE")
    openai.api_version = os.environ.get("AZURE_API_VERSION")
    openai.api_key = os.environ.get("AZURE_API_KEY")
    print(f"openai api_key: {openai.api_key}")
    ## expose a logging function -> log the exact model details + what's being sent 
    if logger_fn and callable(logger_fn):
        model_call_details = {}
        model_call_details["model"] = model
        model_call_details["input"] = input
        model_call_details["azure_value"] = azure
        model_call_details["api_type"] = openai.api_type
        model_call_details["api_base"] = openai.api_base
        model_call_details["api_version"] = openai.api_version
        model_call_details["api_key"] = openai.api_key
        logger_fn(model_call_details)
    response = openai.Embedding.create(input=input, engine=model)
  elif model in open_ai_embedding_models:
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    ## LOGGING
    if logger_fn and callable(logger_fn):
        model_call_details = {}
        model_call_details["model"] = model
        model_call_details["input"] = input
        model_call_details["azure_value"] = azure
        model_call_details["api_type"] = openai.api_type
        model_call_details["api_base"] = openai.api_base
        model_call_details["api_version"] = openai.api_version
        model_call_details["api_key"] = openai.api_key
        logger_fn(model_call_details)
    response = openai.Embedding.create(input=input, model=model)

  return response


#############################################


####### COMPLETION CLIENT ################
#############################################

class berri_client:
    def __init__(self, success_callback=[], failure_callback=[]):  # Constructor
        self.success_callback = success_callback
        self.failure_callback = failure_callback
        self.callback_list = list(set(self.success_callback + self.failure_callback))
        self.set_callbacks()

    def set_callbacks(self):  # Method
        # Method code here
        for callback in self.callback_list:
          if callback == "sentry":
            sentry_sdk.init(dsn=os.environ.get("SENTRY_API_URL"), traces_sample_rate=float(os.environ.get("SENTRY_API_TRACE_RATE")))
          elif callback == "posthog":
            self.posthog = Posthog(
              project_api_key=os.environ.get("POSTHOG_API_KEY"),
              host=os.environ.get("POSTHOG_API_URL"))
          elif callback == "slack":
            self.slack_app = App(
              token=os.environ.get("SLACK_API_TOKEN"),
              signing_secret=os.environ.get("SLACK_API_SECRET")
            )
            self.alerts_channel = os.environ["SLACK_API_CHANNEL"]
    
    def handle_input(self, model_call_details={}):
      # add a sentry breadcrumb
      if len(model_call_details.keys()) > 0:
        model = model_call_details["model"] if "model" in model_call_details else None
        if model:
          for callback in self.callback_list:
            if callback == "sentry":
              add_breadcrumb(
                category=f'{model}',
                message='Trying request model {} input {}'.format(model, json.dumps(model_call_details)),
                level='info',
              )
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
            event_name = additional_details["success_event"] if "success_event" in additional_details else "litellm.succes_query"
            self.posthog.capture(additional_details["user_email"], event_name, ph_obj)
            pass
        except:
          pass
      
      if success_handler:
        if callable(success_handler):
          success_handler(additional_details)
      pass

    def handle_failure(self, exception, additional_details):
      success_handler = additional_details.pop("success_handler", None)
      failure_handler = additional_details.pop("failure_handler", None)

      for callback in self.failure_callback:
        try:
          if callback == "slack":
            slack_msg = "" 
            if len(additional_details.keys()) > 0:
              for detail in additional_details: 
                slack_msg += f"{detail}: {additional_details[detail]}\n"
            slack_msg += f"Traceback: {traceback.format_exc()}"
            self.slack_app.client.chat_postMessage(channel=self.alerts_channel, text=slack_msg)
          elif callback == "sentry":
            capture_exception(exception)
          elif callback == "posthog":
            if len(additional_details.keys()) > 0:
              ph_obj = {}
              for detail in additional_details:
                ph_obj[detail] = additional_details[detail]
              event_name = additional_details["failed_event"] if "failed_event" in additional_details else "litellm.failed_query"
              self.posthog.capture(additional_details["user_email"], event_name, ph_obj)
            else: 
              pass
        except:
          print(f"got an error calling {callback} - {traceback.format_exc()}")
      
      if failure_handler:
        if callable(failure_handler):
          failure_handler(additional_details)
      pass

    def completion(self, model, messages, max_tokens=300, forceTimeout=10, azure=False, additional_details={}) -> Any:
      try:
        model_call_details = {}
        model_call_details["model"] = model
        model_call_details["input"] = messages
        model_call_details["azure_value"] = azure
        self.handle_input(model_call_details)
        response = completion(model=model, messages=messages, max_tokens=max_tokens, forceTimeout=forceTimeout, azure=azure)
        my_thread = threading.Thread(target=self.handle_success, args=(model, messages, additional_details)) # don't interrupt execution of main thread
        my_thread.start()
        return response
      except Exception as e: 
        self.handle_failure(e, additional_details)
        raise e
      
    def embedding(self, model, input=[], azure=False, additional_details={}) -> Any:
      try:
        response = embedding(model, input, azure=azure, logger_fn=self.handle_input)
        my_thread = threading.Thread(target=self.handle_success, args=(model, input, additional_details)) # don't interrupt execution of main thread
        my_thread.start()
        return response
      except Exception as e: 
        self.handle_failure(e, additional_details)
        raise e
