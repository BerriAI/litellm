import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse

class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman:"
    AI_PROMPT = "\n\nAssistant:"

class AnthropicError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

# makes headers for API call
def validate_environment(api_key):
    if api_key is None:
        raise ValueError(
            "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
        )
    headers = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    return headers

def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    prompt = f"{AnthropicConstants.HUMAN_PROMPT.value}"
    for message in messages:
        if "role" in message:
            if message["role"] == "user":
                prompt += (
                    f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
                )
            else:
                prompt += (
                    f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
                )
        else:
            prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
    prompt += f"{AnthropicConstants.AI_PROMPT.value}"
    if "max_tokens" in optional_params and optional_params["max_tokens"] != float("inf"):
        max_tokens = optional_params["max_tokens"]
    else:
        max_tokens = 256 # required anthropic param, default to 256 if user does not provide an input
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )
    
    ## COMPLETION CALL
    if "stream" in optional_params and optional_params["stream"] == True:
        response = requests.post(
            "https://api.anthropic.com/v1/complete",
            headers=headers,
            data=json.dumps(data),
            stream=optional_params["stream"],
        )
        return response.iter_lines()
    else:
        response = requests.post(
            "https://api.anthropic.com/v1/complete", headers=headers, data=json.dumps(data)
        )
        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise AnthropicError(
                message=response.text, status_code=response.status_code
            )
        if "error" in completion_response:
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
            )
        else:
            model_response["choices"][0]["message"]["content"] = completion_response[
                "completion"
            ]
            model_response.choices[0].finish_reason = completion_response["stop_reason"]

        ## CALCULATING USAGE
        prompt_tokens = len(
            encoding.encode(prompt)
        )  ##[TODO] use the anthropic tokenizer here
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"]["content"])
        )  ##[TODO] use the anthropic tokenizer here

        model_response["created"] = time.time()
        model_response["model"] = model
        model_response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return model_response

def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
