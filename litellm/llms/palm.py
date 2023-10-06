import os, types
import json
from enum import Enum
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse, get_secret
import litellm
import sys

class PalmError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class PalmConfig(): 
    """
    Reference: https://developers.generativeai.google/api/python/google/generativeai/chat

    The class `PalmConfig` provides configuration for the Palm's API interface. Here are the parameters:

    - `context` (string): Text that should be provided to the model first, to ground the response. This could be a prompt to guide the model's responses.

    - `examples` (list): Examples of what the model should generate. They are treated identically to conversation messages except that they take precedence over the history in messages if the total input size exceeds the model's input_token_limit.

    - `temperature` (float): Controls the randomness of the output. Must be positive. Higher values produce a more random and varied response. A temperature of zero will be deterministic.

    - `candidate_count` (int): Maximum number of generated response messages to return. This value must be between [1, 8], inclusive. Only unique candidates are returned.

    - `top_k` (int): The API uses combined nucleus and top-k sampling. `top_k` sets the maximum number of tokens to sample from on each step.

    - `top_p` (float): The API uses combined nucleus and top-k sampling. `top_p` configures the nucleus sampling. It sets the maximum cumulative probability of tokens to sample from.

    - `maxOutputTokens` (int): Sets the maximum number of tokens to be returned in the output
    """
    context: Optional[str]=None
    examples: Optional[list]=None
    temperature: Optional[float]=None
    candidate_count: Optional[int]=None
    top_k: Optional[int]=None
    top_p: Optional[float]=None
    maxOutputTokens: Optional[int]=None

    def __init__(self,
                 context: Optional[str]=None,
                 examples: Optional[list]=None,
                 temperature: Optional[float]=None,
                 candidate_count: Optional[int]=None,
                 top_k: Optional[int]=None,
                 top_p: Optional[float]=None,
                 maxOutputTokens: Optional[int]=None) -> None:
        
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
    
    ## Load Config
    config = litellm.PalmConfig.get_config() 
    for k, v in config.items(): 
        if k not in optional_params: # completion(top_k=3) > palm_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

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
