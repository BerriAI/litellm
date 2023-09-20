import os
import json
from enum import Enum
import requests
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse
from .prompt_templates.factory import prompt_factory, custom_prompt

class OobaboogaError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

def validate_environment(api_key):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    return headers

def completion(
    model: str,
    messages: list,
    api_base: Optional[str],
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    default_max_tokens_to_sample=None,
):
    headers = validate_environment(api_key)
    if "https" in model:
        completion_url = model
    elif api_base:
        completion_url = api_base
    else: 
        raise OobaboogaError(status_code=404, message="API Base not set. Set one via completion(..,api_base='your-api-url')")
    model = model
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
    
    completion_url = completion_url + "/api/v1/generate"
    data = {
        "prompt": prompt,
        **optional_params,
    }
    ## LOGGING
    logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
        )
    ## COMPLETION CALL
    response = requests.post(
        completion_url, headers=headers, data=json.dumps(data), stream=optional_params["stream"] if "stream" in optional_params else False
    )
    if "stream" in optional_params and optional_params["stream"] == True:
        return response.iter_lines()
    else:
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
            raise OobaboogaError(message=response.text, status_code=response.status_code)
        if "error" in completion_response:
            raise OobaboogaError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            try:
                model_response["choices"][0]["message"]["content"] = completion_response['results'][0]['text']
            except:
                raise OobaboogaError(message=json.dumps(completion_response), status_code=response.status_code)

        ## CALCULATING USAGE
        prompt_tokens = len(
            encoding.encode(prompt)
        ) 
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"]["content"])
        )

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
