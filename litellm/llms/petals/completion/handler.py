import time
from typing import Callable, Optional, Union

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.utils import ModelResponse, Usage

from ..common_utils import PetalsError


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
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
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
        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client()
        response = client.post(api_base, data=data)

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
            PetalsError(
                status_code=response.status_code,
                message=str(e),
                headers=response.headers,
            )

    else:
        try:
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
