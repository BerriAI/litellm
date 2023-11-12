import os, types
import json
from enum import Enum
import requests
import time
from typing import Callable, Optional
import litellm
from litellm.utils import ModelResponse, Usage

class NLPCloudError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class NLPCloudConfig():
    """
    Reference: https://docs.nlpcloud.com/#generation

    - `max_length` (int): Optional. The maximum number of tokens that the generated text should contain.

    - `length_no_input` (boolean): Optional. Whether `min_length` and `max_length` should not include the length of the input text.

    - `end_sequence` (string): Optional. A specific token that should be the end of the generated sequence.

    - `remove_end_sequence` (boolean): Optional. Whether to remove the `end_sequence` string from the result.

    - `remove_input` (boolean): Optional. Whether to remove the input text from the result.

    - `bad_words` (list of strings): Optional. List of tokens that are not allowed to be generated.

    - `temperature` (float): Optional. Temperature sampling. It modulates the next token probabilities.

    - `top_p` (float): Optional. Top P sampling. Below 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.

    - `top_k` (int): Optional. Top K sampling. The number of highest probability vocabulary tokens to keep for top k filtering.

    - `repetition_penalty` (float): Optional. Prevents the same word from being repeated too many times.

    - `num_beams` (int): Optional. Number of beams for beam search.

    - `num_return_sequences` (int): Optional. The number of independently computed returned sequences.
    """
    max_length: Optional[int]=None
    length_no_input: Optional[bool]=None
    end_sequence: Optional[str]=None
    remove_end_sequence: Optional[bool]=None
    remove_input: Optional[bool]=None
    bad_words: Optional[list]=None
    temperature: Optional[float]=None
    top_p: Optional[float]=None
    top_k: Optional[int]=None
    repetition_penalty: Optional[float]=None
    num_beams: Optional[int]=None
    num_return_sequences: Optional[int]=None


    def __init__(self,
                 max_length: Optional[int]=None,
                 length_no_input: Optional[bool]=None,
                 end_sequence: Optional[str]=None,
                 remove_end_sequence: Optional[bool]=None,
                 remove_input: Optional[bool]=None,
                 bad_words: Optional[list]=None,
                 temperature: Optional[float]=None,
                 top_p: Optional[float]=None,
                 top_k: Optional[int]=None,
                 repetition_penalty: Optional[float]=None,
                 num_beams: Optional[int]=None,
                 num_return_sequences: Optional[int]=None) -> None:
        
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
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    default_max_tokens_to_sample=None,
):
    headers = validate_environment(api_key)

    ## Load Config
    config = litellm.NLPCloudConfig.get_config() 
    for k, v in config.items(): 
        if k not in optional_params: # completion(top_k=3) > togetherai_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    completion_url_fragment_1 = api_base
    completion_url_fragment_2 = "/generation"
    model = model
    text = " ".join(message["content"] for message in messages)

    data = {
        "text": text,
        **optional_params,
    }

    completion_url = completion_url_fragment_1 + model + completion_url_fragment_2

    ## LOGGING
    logging_obj.pre_call(
            input=text,
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
                input=text,
                api_key=api_key,
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise NLPCloudError(message=response.text, status_code=response.status_code)
        if "error" in completion_response:
            raise NLPCloudError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            try:
                if len(completion_response["generated_text"]) > 0:
                    model_response["choices"][0]["message"]["content"] = completion_response["generated_text"]
            except:
                raise NLPCloudError(message=json.dumps(completion_response), status_code=response.status_code)

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here. 
        prompt_tokens = completion_response["nb_input_tokens"]
        completion_tokens = completion_response["nb_generated_tokens"]

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
