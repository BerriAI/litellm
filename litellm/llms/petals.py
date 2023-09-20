import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse

class PetalsError(Exception):
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
    encoding,
    logging_obj,
    optional_params=None,
    stream=False,
    litellm_params=None,
    logger_fn=None,
):
    try:
        import torch
        from transformers import AutoTokenizer
        from petals import AutoDistributedModelForCausalLM
    except:
        raise Exception(
            "Importing torch, transformers, petals failed\nTry pip installing petals \npip install git+https://github.com/bigscience-workshop/petals"
        )

    model = model

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, add_bos_token=False)
    model_obj = AutoDistributedModelForCausalLM.from_pretrained(model)

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
            additional_args={"complete_input_dict": optional_params},
        )
    
    ## COMPLETION CALL
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    # optional params: max_new_tokens=1,temperature=0.9, top_p=0.6
    outputs = model_obj.generate(inputs, **optional_params)

    ## LOGGING
    logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=outputs,
            additional_args={"complete_input_dict": optional_params},
        )
    ## RESPONSE OBJECT
    output_text = tokenizer.decode(outputs[0])
    model_response["choices"][0]["message"]["content"] = output_text

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
