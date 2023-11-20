import os, types
import json
from enum import Enum
import requests
import time
from typing import Callable, Optional
import litellm
import httpx
from litellm.utils import ModelResponse, Usage
from .prompt_templates.factory import prompt_factory, custom_prompt

class TogetherAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.together.xyz/inference")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class TogetherAIConfig():
    """
    Reference: https://docs.together.ai/reference/inference

    The class `TogetherAIConfig` provides configuration for the TogetherAI's API interface. Here are the parameters:

    - `max_tokens` (int32, required): The maximum number of tokens to generate.

    - `stop` (string, optional): A string sequence that will truncate (stop) the inference text output. For example, "\n\n" will stop generation as soon as the model generates two newlines.

    - `temperature` (float, optional): A decimal number that determines the degree of randomness in the response. A value of 1 will always yield the same output. A temperature less than 1 favors more correctness and is appropriate for question answering or summarization. A value greater than 1 introduces more randomness in the output.

    - `top_p` (float, optional): The `top_p` (nucleus) parameter is used to dynamically adjust the number of choices for each predicted token based on the cumulative probabilities. It specifies a probability threshold, below which all less likely tokens are filtered out. This technique helps to maintain diversity and generate more fluent and natural-sounding text.

    - `top_k` (int32, optional): The `top_k` parameter is used to limit the number of choices for the next predicted word or token. It specifies the maximum number of tokens to consider at each step, based on their probability of occurrence. This technique helps to speed up the generation process and can improve the quality of the generated text by focusing on the most likely options.

    - `repetition_penalty` (float, optional): A number that controls the diversity of generated text by reducing the likelihood of repeated sequences. Higher values decrease repetition.

    - `logprobs` (int32, optional): This parameter is not described in the prompt. 
    """
    max_tokens: Optional[int]=None
    stop: Optional[str]=None
    temperature:Optional[int]=None
    top_p: Optional[float]=None
    top_k: Optional[int]=None
    repetition_penalty: Optional[float]=None
    logprobs: Optional[int]=None
    
    def __init__(self,
                 max_tokens: Optional[int]=None,
                 stop: Optional[str]=None,
                 temperature:Optional[int]=None,
                 top_p: Optional[float]=None,
                 top_k: Optional[int]=None,
                 repetition_penalty: Optional[float]=None,
                 logprobs: Optional[int]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}


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
    api_base: str, 
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

    ## Load Config
    config = litellm.TogetherAIConfig.get_config() 
    for k, v in config.items(): 
        if k not in optional_params: # completion(top_k=3) > togetherai_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    print_verbose(f"CUSTOM PROMPT DICT: {custom_prompt_dict}; model: {model}")
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details = custom_prompt_dict[model]
        prompt = custom_prompt(
                role_dict=model_prompt_details.get("roles", {}), 
                initial_prompt_value=model_prompt_details.get("initial_prompt_value", ""),  
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""), 
                bos_token=model_prompt_details.get("bos_token", ""),
                eos_token=model_prompt_details.get("eos_token", ""),
                messages=messages,
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
        additional_args={"complete_input_dict": data, "headers": headers, "api_base": api_base},
    )
    ## COMPLETION CALL
    if (
            "stream_tokens" in optional_params
            and optional_params["stream_tokens"] == True
        ):
        response = requests.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=optional_params["stream_tokens"],
        )
        return response.iter_lines()
    else:
        response = requests.post(
            api_base,
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
        if response.status_code != 200:
            raise TogetherAIError(
                status_code=response.status_code, message=response.text
            )
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
    
        if len(completion_response["output"]["choices"][0]["text"]) > 0:
            model_response["choices"][0]["message"]["content"] = completion_response["output"]["choices"][0]["text"]

        ## CALCULATING USAGE
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )
        if "finish_reason" in completion_response["output"]["choices"][0]:
            model_response.choices[0].finish_reason = completion_response["output"]["choices"][0]["finish_reason"]
        model_response["created"] = time.time()
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        model_response.usage = usage
        return model_response

def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
