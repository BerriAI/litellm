import os, openai, sys
from typing import Any
from functools import partial
import dotenv, traceback, random, asyncio, time
from copy import deepcopy
import litellm
from litellm import client, logging, exception_type, timeout, get_optional_params
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
from litellm.utils import get_secret, install_and_import
####### ENVIRONMENT VARIABLES ###################
dotenv.load_dotenv() # Loading env variables using dotenv

new_response = {
        "choices": [
          {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "role": "assistant"
            }
          }
        ]
      }
# TODO add translations
####### COMPLETION ENDPOINTS ################
#############################################
async def acompletion(*args, **kwargs):
  loop = asyncio.get_event_loop()
  
  # Use a partial function to pass your keyword arguments
  func = partial(completion, *args, **kwargs)

  # Call the synchronous function using run_in_executor
  return await loop.run_in_executor(None, func)

@client
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2), reraise=True, retry_error_callback=lambda retry_state: setattr(retry_state.outcome, 'retry_variable', litellm.retry)) # retry call, turn this off by setting `litellm.retry = False`
@timeout(60) ## set timeouts, in case calls hang (e.g. Azure) - default is 60s, override with `force_timeout`
def completion(
    model, messages, # required params
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    functions=[], function_call="", # optional params
    temperature=1, top_p=1, n=1, stream=False, stop=None, max_tokens=float('inf'),
    presence_penalty=0, frequency_penalty=0, logit_bias={}, user="", deployment_id=None,
    # Optional liteLLM function params
    *, return_async=False, api_key=None, force_timeout=60, azure=False, logger_fn=None, verbose=False,
    hugging_face = False
  ):
  try:
    global new_response
    model_response = deepcopy(new_response) # deep copy the default response format so we can mutate it and it's thread-safe. 
    # check if user passed in any of the OpenAI optional params
    optional_params = get_optional_params(
      functions=functions, function_call=function_call, 
      temperature=temperature, top_p=top_p, n=n, stream=stream, stop=stop, max_tokens=max_tokens,
      presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, logit_bias=logit_bias, user=user, deployment_id=deployment_id
    )
    if azure == True:
      # azure configs
      openai.api_type = "azure"
      openai.api_base = litellm.api_base if litellm.api_base is not None else get_secret("AZURE_API_BASE")
      openai.api_version = litellm.api_version if litellm.api_version is not None else get_secret("AZURE_API_VERSION")
      # set key
      if api_key:
          openai.api_key = api_key
      elif litellm.azure_key:
          openai.api_key = litellm.azure_key
      else:
          openai.api_key = get_secret("AZURE_API_KEY")
      ## LOGGING
      logging(model=model, input=messages, additional_args=optional_params, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      if litellm.headers:
         response = openai.ChatCompletion.create(
            engine=model,
            messages = messages,
            headers = litellm.headers,
            **optional_params,
          )
      else:
        response = openai.ChatCompletion.create(
          model=model,
          messages = messages,
          **optional_params
        )
    elif model in litellm.open_ai_chat_completion_models:
      openai.api_type = "openai"
      # note: if a user sets a custom base - we should ensure this works
      openai.api_base = litellm.api_base if litellm.api_base is not None else "https://api.openai.com/v1"
      openai.api_version = None
      if litellm.organization:
        openai.organization = litellm.organization
      if api_key:
          openai.api_key = api_key
      elif litellm.openai_key:
          openai.api_key = litellm.openai_key
      else:
          openai.api_key = get_secret("OPENAI_API_KEY")
      ## LOGGING
      logging(model=model, input=messages, additional_args=optional_params, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      if litellm.headers:
         response = openai.ChatCompletion.create(
          model=model,
          messages = messages,
          headers = litellm.headers,
          **optional_params
        )
      else:
        response = openai.ChatCompletion.create(
          model=model,
          messages = messages,
          **optional_params
        )
    elif model in litellm.open_ai_text_completion_models:
      openai.api_type = "openai"
      openai.api_base = litellm.api_base if litellm.api_base is not None else "https://api.openai.com/v1"
      openai.api_version = None
      if api_key:
          openai.api_key = api_key
      elif litellm.openai_key:
          openai.api_key = litellm.openai_key
      else:
          openai.api_key = get_secret("OPENAI_API_KEY")
      if litellm.organization:
        openai.organization = litellm.organization
      prompt = " ".join([message["content"] for message in messages])
      ## LOGGING
      logging(model=model, input=prompt, additional_args=optional_params, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      if litellm.headers:
        response = openai.Completion.create(
          model=model,
          prompt = prompt,
          headers = litellm.headers,
        )
      else:
        response = openai.Completion.create(
            model=model,
            prompt = prompt
        )
      completion_response = response["choices"]["text"]
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, additional_args={"max_tokens": max_tokens, "original_response": completion_response}, logger_fn=logger_fn)
      ## RESPONSE OBJECT
      model_response["choices"][0]["message"]["content"] = completion_response
      model_response["created"] = response["created"]
      model_response["model"] = model
      model_response["usage"] = response["usage"]
      response = model_response
    elif "replicate" in model:
      # import replicate/if it fails then pip install replicate
      install_and_import("replicate")
      import replicate
      # replicate defaults to os.environ.get("REPLICATE_API_TOKEN")
      # checking in case user set it to REPLICATE_API_KEY instead 
      if not get_secret("REPLICATE_API_TOKEN") and get_secret("REPLICATE_API_KEY"):
        replicate_api_token = get_secret("REPLICATE_API_KEY")
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
      elif api_key:
         os.environ["REPLICATE_API_TOKEN"] = api_key
      elif litellm.replicate_key:
         os.environ["REPLICATE_API_TOKEN"] = litellm.replicate_key
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
      completion_response = response
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, additional_args={"max_tokens": max_tokens, "original_response": completion_response}, logger_fn=logger_fn)
      prompt_tokens = len(encoding.encode(prompt))
      completion_tokens = len(encoding.encode(completion_response))
      ## RESPONSE OBJECT
      model_response["choices"][0]["message"]["content"] = completion_response
      model_response["created"] = time.time()
      model_response["model"] = model
      model_response["usage"] = {
          "prompt_tokens": prompt_tokens,
          "completion_tokens": completion_tokens,
          "total_tokens": prompt_tokens + completion_tokens
        }
      response = model_response
    elif model in litellm.anthropic_models:
      # import anthropic/if it fails then pip install anthropic
      install_and_import("anthropic")
      from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

      #anthropic defaults to os.environ.get("ANTHROPIC_API_KEY")
      if api_key:
         os.environ["ANTHROPIC_API_KEY"] = api_key
      elif litellm.anthropic_key:
         os.environ["ANTHROPIC_API_KEY"] = litellm.anthropic_key
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
      if max_tokens != float('inf'):
        max_tokens_to_sample = max_tokens
      else:
        max_tokens_to_sample = litellm.max_tokens # default in Anthropic docs https://docs.anthropic.com/claude/reference/client-libraries
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, additional_args={"max_tokens": max_tokens}, logger_fn=logger_fn)
      ## COMPLETION CALL
      completion = anthropic.completions.create(
            model=model,
            prompt=prompt,
            max_tokens_to_sample=max_tokens_to_sample
        )
      completion_response = completion.completion
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, additional_args={"max_tokens": max_tokens, "original_response": completion_response}, logger_fn=logger_fn)
      prompt_tokens = anthropic.count_tokens(prompt)
      completion_tokens = anthropic.count_tokens(completion_response)
      ## RESPONSE OBJECT
      print_verbose(f"raw model_response: {model_response}")
      model_response["choices"][0]["message"]["content"] = completion_response
      model_response["created"] = time.time()
      model_response["model"] = model
      model_response["usage"] = {
          "prompt_tokens": prompt_tokens,
          "completion_tokens": completion_tokens,
          "total_tokens": prompt_tokens + completion_tokens
        }
      response = model_response
    elif model in litellm.cohere_models:
      # import cohere/if it fails then pip install cohere
      install_and_import("cohere")
      import cohere
      if api_key:
        cohere_key = api_key
      elif litellm.cohere_key:
        cohere_key = litellm.cohere_key
      else:
        cohere_key = get_secret("COHERE_API_KEY")
      co = cohere.Client(cohere_key)
      prompt = " ".join([message["content"] for message in messages])
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, logger_fn=logger_fn)
      ## COMPLETION CALL
      response = co.generate(  
        model=model,
        prompt = prompt
      )
      completion_response = response[0].text
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, additional_args={"max_tokens": max_tokens, "original_response": completion_response}, logger_fn=logger_fn)
      prompt_tokens = len(encoding.encode(prompt))
      completion_tokens = len(encoding.encode(completion_response))
      ## RESPONSE OBJECT
      model_response["choices"][0]["message"]["content"] = completion_response
      model_response["created"] = time.time()
      model_response["model"] = model
      model_response["usage"] = {
          "prompt_tokens": prompt_tokens,
          "completion_tokens": completion_tokens,
          "total_tokens": prompt_tokens + completion_tokens
        }
      response = model_response
    elif hugging_face == True:
      import requests
      API_URL = f"https://api-inference.huggingface.co/models/{model}"
      HF_TOKEN = get_secret("HF_TOKEN")
      headers = {"Authorization": f"Bearer {HF_TOKEN}"}

      prompt = " ".join([message["content"] for message in messages])
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, logger_fn=logger_fn)
      input_payload = {"inputs": prompt}
      response = requests.post(API_URL, headers=headers, json=input_payload)
  
      completion_response = response.json()[0]['generated_text']
      ## LOGGING
      logging(model=model, input=prompt, azure=azure, additional_args={"max_tokens": max_tokens, "original_response": completion_response}, logger_fn=logger_fn)
      prompt_tokens = len(encoding.encode(prompt))
      completion_tokens = len(encoding.encode(completion_response))
      ## RESPONSE OBJECT
      model_response["choices"][0]["message"]["content"] = completion_response
      model_response["created"] = time.time()
      model_response["model"] = model
      model_response["usage"] = {
          "prompt_tokens": prompt_tokens,
          "completion_tokens": completion_tokens,
          "total_tokens": prompt_tokens + completion_tokens
        }
    else: 
      ## LOGGING
      logging(model=model, input=messages, azure=azure, logger_fn=logger_fn)
      args = locals()
      raise ValueError(f"No valid completion model args passed in - {args}")
    return response
  except Exception as e:
    ## LOGGING
    logging(model=model, input=messages, azure=azure, additional_args={"max_tokens": max_tokens}, logger_fn=logger_fn, exception=e)
    ## Map to OpenAI Exception
    raise exception_type(model=model, original_exception=e)

### EMBEDDING ENDPOINTS ####################
@client
@timeout(60) ## set timeouts, in case calls hang (e.g. Azure) - default is 60s, override with `force_timeout`
def embedding(model, input=[], azure=False, force_timeout=60, logger_fn=None):
  try:
    response = None
    if azure == True:
      # azure configs
      openai.api_type = "azure"
      openai.api_base = get_secret("AZURE_API_BASE")
      openai.api_version = get_secret("AZURE_API_VERSION")
      openai.api_key = get_secret("AZURE_API_KEY")
      ## LOGGING
      logging(model=model, input=input, azure=azure, logger_fn=logger_fn)
      ## EMBEDDING CALL
      response = openai.Embedding.create(input=input, engine=model)
      print_verbose(f"response_value: {str(response)[:50]}")
    elif model in litellm.open_ai_embedding_models:
      openai.api_type = "openai"
      openai.api_base = "https://api.openai.com/v1"
      openai.api_version = None
      openai.api_key = get_secret("OPENAI_API_KEY")
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
  except Exception as e:
    # log the original exception
    logging(model=model, input=input, azure=azure, logger_fn=logger_fn, exception=e)
    ## Map to OpenAI Exception
    raise exception_type(model=model, original_exception=e)
    raise e
####### HELPER FUNCTIONS ################
## Set verbose to true -> ```litellm.set_verbose = True```    
def print_verbose(print_statement):
  if litellm.set_verbose:
    print(f"LiteLLM: {print_statement}")
    if random.random() <= 0.3:
      print("Get help - https://discord.com/invite/wuPM9dRgDw")

