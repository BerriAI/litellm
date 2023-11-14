import os, types, traceback
import json
from enum import Enum
import requests
import time, httpx
from typing import Callable, Optional
from litellm.utils import ModelResponse, Choices, Message
import litellm

class AI21Error(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.ai21.com/studio/v1/")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class AI21Config():
    """
    Reference: https://docs.ai21.com/reference/j2-complete-ref

    The class `AI21Config` provides configuration for the AI21's API interface. Below are the parameters:

    - `numResults` (int32): Number of completions to sample and return. Optional, default is 1. If the temperature is greater than 0 (non-greedy decoding), a value greater than 1 can be meaningful.
        
    - `maxTokens` (int32): The maximum number of tokens to generate per result. Optional, default is 16. If no `stopSequences` are given, generation stops after producing `maxTokens`.
        
    - `minTokens` (int32): The minimum number of tokens to generate per result. Optional, default is 0. If `stopSequences` are given, they are ignored until `minTokens` are generated.
        
    - `temperature` (float): Modifies the distribution from which tokens are sampled. Optional, default is 0.7. A value of 0 essentially disables sampling and results in greedy decoding.
        
    - `topP` (float): Used for sampling tokens from the corresponding top percentile of probability mass. Optional, default is 1. For instance, a value of 0.9 considers only tokens comprising the top 90% probability mass.
        
    - `stopSequences` (array of strings): Stops decoding if any of the input strings is generated. Optional.
        
    - `topKReturn` (int32): Range between 0 to 10, including both. Optional, default is 0. Specifies the top-K alternative tokens to return. A non-zero value includes the string representations and log-probabilities for each of the top-K alternatives at each position.
        
    - `frequencyPenalty` (object): Placeholder for frequency penalty object.
        
    - `presencePenalty` (object): Placeholder for presence penalty object.
        
    - `countPenalty` (object): Placeholder for count penalty object.
    """
    numResults: Optional[int]=None
    maxTokens: Optional[int]=None
    minTokens: Optional[int]=None
    temperature: Optional[float]=None
    topP: Optional[float]=None
    stopSequences: Optional[list]=None
    topKReturn: Optional[int]=None
    frequencePenalty: Optional[dict]=None
    presencePenalty: Optional[dict]=None
    countPenalty: Optional[dict]=None

    def __init__(self, 
                 numResults: Optional[int]=None,
                 maxTokens: Optional[int]=None,
                 minTokens: Optional[int]=None,
                 temperature: Optional[float]=None,
                 topP: Optional[float]=None,
                 stopSequences: Optional[list]=None,
                 topKReturn: Optional[int]=None,
                 frequencePenalty: Optional[dict]=None,
                 presencePenalty: Optional[dict]=None,
                 countPenalty: Optional[dict]=None) -> None:
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
            "Missing AI21 API Key - A call is being made to ai21 but no key is set either in the environment variables or via params"
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
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
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
    
    ## Load Config
    config = litellm.AI21Config.get_config() 
    for k, v in config.items(): 
        if k not in optional_params: # completion(top_k=3) > ai21_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    data = {
        "prompt": prompt,
        # "instruction": prompt, # some baseten models require the prompt to be passed in via the 'instruction' kwarg
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
        api_base + model + "/complete", headers=headers, data=json.dumps(data)
    )
    if response.status_code != 200:
        raise AI21Error(
            status_code=response.status_code,
            message=response.text
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
        ## RESPONSE OBJECT
        completion_response = response.json()
        try:
            choices_list = []
            for idx, item in enumerate(completion_response["completions"]):
                if len(item["data"]["text"]) > 0:
                    message_obj = Message(content=item["data"]["text"])
                else: 
                    message_obj = Message(content=None)
                choice_obj = Choices(finish_reason=item["finishReason"]["reason"], index=idx+1, message=message_obj)
                choices_list.append(choice_obj)
            model_response["choices"] = choices_list
        except Exception as e:
            raise AI21Error(message=traceback.format_exc(), status_code=response.status_code)

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here. 
        prompt_tokens = len(
            encoding.encode(prompt)
        ) 
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content"))
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
