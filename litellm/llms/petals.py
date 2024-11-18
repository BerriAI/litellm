import json
import os
import time
import types
from enum import Enum
from typing import Callable, Optional

import requests  # type: ignore

import litellm
from litellm.utils import ModelResponse, Usage

from .prompt_templates.factory import custom_prompt, prompt_factory


class PetalsError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class PetalsConfig:
    """
    Reference: https://github.com/petals-infra/chat.petals.dev#post-apiv1generate
    The `PetalsConfig` class encapsulates the configuration for the Petals API. The properties of this class are described below:

    - `max_length` (integer): This represents the maximum length of the generated text (including the prefix) in tokens.

    - `max_new_tokens` (integer): This represents the maximum number of newly generated tokens (excluding the prefix).

    The generation parameters are compatible with `.generate()` from Hugging Face's Transformers library:

    - `do_sample` (boolean, optional): If set to 0 (default), the API runs greedy generation. If set to 1, the API performs sampling using the parameters below:

    - `temperature` (float, optional): This value sets the temperature for sampling.

    - `top_k` (integer, optional): This value sets the limit for top-k sampling.

    - `top_p` (float, optional): This value sets the limit for top-p (nucleus) sampling.

    - `repetition_penalty` (float, optional): This helps apply the repetition penalty during text generation, as discussed in this paper.
    """

    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = (
        litellm.max_tokens
    )  # petals requires max tokens to be set
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    def __init__(
        self,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[
            int
        ] = litellm.max_tokens,  # petals requires max tokens to be set
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }


def completion(
    model: str,
    messages: list,
    api_base: Optional[str],
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    optional_params: dict,
    stream=False,
    litellm_params=None,
    logger_fn=None,
):
    ## Load Config
    config = litellm.PetalsConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > petals_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    if model in litellm.custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details = litellm.custom_prompt_dict[model]
        prompt = custom_prompt(
            role_dict=model_prompt_details["roles"],
            initial_prompt_value=model_prompt_details["initial_prompt_value"],
            final_prompt_value=model_prompt_details["final_prompt_value"],
            messages=messages,
        )
    else:
        prompt = prompt_factory(model=model, messages=messages)

    output_text: Optional[str] = None
    if api_base:
        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
            },
        )
        data = {"model": model, "inputs": prompt, **optional_params}

        ## COMPLETION CALL
        response = requests.post(api_base, data=data)

        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=response.text,
            additional_args={"complete_input_dict": optional_params},
        )

        ## RESPONSE OBJECT
        try:
            output_text = response.json()["outputs"]
        except Exception as e:
            PetalsError(status_code=response.status_code, message=str(e))

    else:
        try:
            import torch
            from petals import AutoDistributedModelForCausalLM  # type: ignore
            from transformers import AutoTokenizer
        except Exception:
            raise Exception(
                "Importing torch, transformers, petals failed\nTry pip installing petals \npip install git+https://github.com/bigscience-workshop/petals"
            )

        model = model

        tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=False, add_bos_token=False
        )
        model_obj = AutoDistributedModelForCausalLM.from_pretrained(model)

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

    if output_text is not None and len(output_text) > 0:
        model_response.choices[0].message.content = output_text  # type: ignore

    prompt_tokens = len(encoding.encode(prompt))
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"].get("content"))
    )

    model_response.created = int(time.time())
    model_response.model = model
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    setattr(model_response, "usage", usage)
    return model_response


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
