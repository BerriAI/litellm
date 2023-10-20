from typing import Dict, Optional
from collections import defaultdict
import threading
import os, subprocess, traceback, json
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import backoff
import openai.error

import litellm
from litellm.utils import trim_messages
from litellm.exceptions import ServiceUnavailableError, InvalidRequestError

cost_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
cost_dict_lock = threading.Lock()

debug = False
##### HELPER FUNCTIONS #####
def print_verbose(print_statement):
    global debug 
    if debug: 
         print(print_statement)

# for streaming
def data_generator(response):
    print_verbose("inside generator")
    for chunk in response:
        print_verbose(f"returned chunk: {chunk}")
        yield f"data: {json.dumps(chunk)}\n\n"

def run_ollama_serve():
    command = ['ollama', 'serve']
    
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(command, stdout=devnull, stderr=devnull)

##### ERROR HANDLING #####
class RetryConstantError(Exception):
    pass


class RetryExpoError(Exception):
    pass


class UnknownLLMError(Exception):
    pass


def handle_llm_exception(e: Exception, user_api_base: Optional[str]=None):
    print(f"\033[1;31mLiteLLM.Exception: {str(e)}\033[0m")
    if isinstance(e, ServiceUnavailableError) and e.llm_provider == "ollama": # type: ignore
        run_ollama_serve()
    if isinstance(e, InvalidRequestError) and e.llm_provider == "ollama": # type: ignore
        completion_call_details = {}
        completion_call_details["model"] = e.model # type: ignore
        if user_api_base: 
            completion_call_details["api_base"] = user_api_base
        else: 
            completion_call_details["api_base"] = None
        print(f"\033[1;31mLiteLLM.Exception: Invalid API Call. Call details: Model: \033[1;37m{e.model}\033[1;31m; LLM Provider: \033[1;37m{e.llm_provider}\033[1;31m; Custom API Base - \033[1;37m{completion_call_details['api_base']}\033[1;31m\033[0m") # type: ignore
        if completion_call_details["api_base"] == "http://localhost:11434": 
            print()
            print("Trying to call ollama? Try `litellm --model ollama/llama2 --api_base http://localhost:11434`")
            print()
    if isinstance(
        e,
        (
            openai.error.APIError,
            openai.error.TryAgain,
            openai.error.Timeout,
            openai.error.ServiceUnavailableError,
        ),
    ):
        raise RetryConstantError from e
    elif isinstance(e, openai.error.RateLimitError):
        raise RetryExpoError from e
    elif isinstance(
        e,
        (
            openai.error.APIConnectionError,
            openai.error.InvalidRequestError,
            openai.error.AuthenticationError,
            openai.error.PermissionError,
            openai.error.InvalidAPIType,
            openai.error.SignatureVerificationError,
        ),
    ):
        raise e
    else:
        raise UnknownLLMError from e


@backoff.on_exception(
    wait_gen=backoff.constant,
    exception=RetryConstantError,
    max_tries=3,
    interval=3,
)
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=RetryExpoError,
    jitter=backoff.full_jitter,
    max_value=100,
    factor=1.5,
)

def litellm_completion(data: Dict,
                type: str, 
                user_model: Optional[str], 
                user_temperature: Optional[str], 
                user_max_tokens: Optional[int], 
                user_request_timeout: Optional[int],
                user_api_base: Optional[str], 
                user_headers: Optional[dict], 
                user_debug: bool,
                model_router: Optional[litellm.Router]):
    try:  
        global debug
        debug = user_debug
        if user_model:
            data["model"] = user_model
        # override with user settings
        if user_temperature: 
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens: 
            data["max_tokens"] = user_max_tokens
        if user_api_base: 
            data["api_base"] = user_api_base
        if user_headers: 
            data["headers"] = user_headers
        if type == "completion": 
            if model_router and data["model"] in model_router.get_model_names(): 
                model_router.text_completion(**data)
            else:
                response = litellm.text_completion(**data)
        elif type == "chat_completion": 
            if model_router and data["model"] in model_router.get_model_names(): 
                model_router.completion(**data)
            else:
                response = litellm.completion(**data)
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(data_generator(response), media_type='text/event-stream')
        print_verbose(f"response: {response}")
        return response
    except Exception as e: 
        print(e)
        handle_llm_exception(e=e, user_api_base=user_api_base)
        return {"message": "An error occurred"}, 500
