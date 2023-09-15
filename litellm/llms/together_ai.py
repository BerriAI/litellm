import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse
from .prompt_templates.factory import prompt_factory, custom_prompt

class TogetherAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

def validate_environment(api_key):
    if api_key is None:
        raise ValueError(
            "Missing TogetherAI API Key - A call is being made to together_ai but no key is set either in the environment variables or via params"
        )
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer " + api_key,
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
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details = custom_prompt_dict[model]
        prompt = custom_prompt(
            role_dict=model_prompt_details["roles"], 
            initial_prompt_value=model_prompt_details["initial_prompt_value"],  
            final_prompt_value=model_prompt_details["final_prompt_value"], 
            messages=messages
        )
    else:
        prompt = prompt_factory(model=model, messages=messages)
    data = {
        "model": model,
        "prompt": prompt,
        "request_type": "language-model-inference",
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
        )
    ## COMPLETION CALL
    if (
            "stream_tokens" in optional_params
            and optional_params["stream_tokens"] == True
        ):
        response = requests.post(
            "https://api.together.xyz/inference",
            headers=headers,
            data=json.dumps(data),
            stream=optional_params["stream_tokens"],
        )
        return response.iter_lines()
    else:
        response = requests.post(
            "https://api.together.xyz/inference",
            headers=headers,
            data=json.dumps(data)
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
        completion_response = response.json()

        if "error" in completion_response:
            raise TogetherAIError(
                message=json.dumps(completion_response),
                status_code=response.status_code,
            )
        elif "error" in completion_response["output"]:
            raise TogetherAIError(
                message=json.dumps(completion_response["output"]), status_code=response.status_code
            )

        completion_text = completion_response["output"]["choices"][0]["text"]

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(completion_text)
        )
        model_response["choices"][0]["message"]["content"] = completion_text
        if "finish_reason" in completion_response["output"]["choices"][0]:
            model_response.choices[0].finish_reason = completion_response["output"]["choices"][0]["finish_reason"]
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
