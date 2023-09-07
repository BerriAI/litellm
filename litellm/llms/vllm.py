import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse
from .prompt_templates.factory import prompt_factory, custom_prompt

class VLLMError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

# check if vllm is installed
def validate_environment():
    try: 
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except:
        raise VLLMError(status_code=0, message="The vllm package is not installed in your environment. Run - `pip install vllm` before proceeding.")

def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    LLM, SamplingParams = validate_environment()
    try:
        llm = LLM(model=model)
    except Exception as e:
        raise VLLMError(status_code=0, message=str(e))
    sampling_params = SamplingParams(**optional_params)
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


    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key="",
        additional_args={"complete_input_dict": sampling_params},
    )

    outputs = llm.generate(prompt, sampling_params)

    
    ## COMPLETION CALL
    if "stream" in optional_params and optional_params["stream"] == True:
        return iter(outputs)
    else:
        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=outputs,
            additional_args={"complete_input_dict": sampling_params},
        )
        print_verbose(f"raw model_response: {outputs}")
        ## RESPONSE OBJECT
        model_response["choices"][0]["message"]["content"] = outputs[0].outputs[0].text

        ## CALCULATING USAGE
        prompt_tokens = len(outputs[0].prompt_token_ids)  
        completion_tokens = len(outputs[0].outputs[0].token_ids)  

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
