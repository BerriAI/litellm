import os
import json
from enum import Enum
import time
from typing import Callable
from litellm.utils import ModelResponse, get_secret
import sys

class PalmError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    api_key,
    encoding,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    try:
        import google.generativeai as palm
    except:
        raise Exception("Importing google.generativeai failed, please run 'pip install -q google-generativeai")
    palm.configure(api_key=api_key)

    model = model
    prompt = ""
    for message in messages:
        if "role" in message:
            if message["role"] == "user":
                prompt += (
                    f"{message['content']}"
                )
            else:
                prompt += (
                    f"{message['content']}"
                )
        else:
            prompt += f"{message['content']}"
    
    ## LOGGING
    logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={"complete_input_dict": {}},
        )
    ## COMPLETION CALL
    response = palm.chat(messages=prompt)

    ## LOGGING
    logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=response,
            additional_args={"complete_input_dict": {}},
        )
    print_verbose(f"raw model_response: {response}")
    ## RESPONSE OBJECT
    completion_response = response.last

    if "error" in completion_response:
        raise PalmError(
            message=completion_response["error"],
            status_code=response.status_code,
        )
    else:
        try:
            model_response["choices"][0]["message"]["content"] = completion_response
        except:
            raise PalmError(message=json.dumps(completion_response), status_code=response.status_code)

    ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here. 
    prompt_tokens = len(
        encoding.encode(prompt)
    ) 
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"]["content"])
    )

    model_response["created"] = time.time()
    model_response["model"] = "palm/" + model
    model_response["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return model_response

def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
