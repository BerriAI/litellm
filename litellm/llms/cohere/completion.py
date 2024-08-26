##### Calls /generate endpoint #######

import json
import os
import time
import traceback
import types
from enum import Enum
from typing import Any, Callable, Optional, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import Choices, Message, ModelResponse, Usage


class CohereError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.cohere.ai/v1/generate"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


def construct_cohere_tool(tools=None):
    if tools is None:
        tools = []
    return {"tools": tools}


class CohereConfig:
    """
    Reference: https://docs.cohere.com/reference/generate

    The class `CohereConfig` provides configuration for the Cohere's API interface. Below are the parameters:

    - `num_generations` (integer): Maximum number of generations returned. Default is 1, with a minimum value of 1 and a maximum value of 5.

    - `max_tokens` (integer): Maximum number of tokens the model will generate as part of the response. Default value is 20.

    - `truncate` (string): Specifies how the API handles inputs longer than maximum token length. Options include NONE, START, END. Default is END.

    - `temperature` (number): A non-negative float controlling the randomness in generation. Lower temperatures result in less random generations. Default is 0.75.

    - `preset` (string): Identifier of a custom preset, a combination of parameters such as prompt, temperature etc.

    - `end_sequences` (array of strings): The generated text gets cut at the beginning of the earliest occurrence of an end sequence, which will be excluded from the text.

    - `stop_sequences` (array of strings): The generated text gets cut at the end of the earliest occurrence of a stop sequence, which will be included in the text.

    - `k` (integer): Limits generation at each step to top `k` most likely tokens. Default is 0.

    - `p` (number): Limits generation at each step to most likely tokens with total probability mass of `p`. Default is 0.

    - `frequency_penalty` (number): Reduces repetitiveness of generated tokens. Higher values apply stronger penalties to previously occurred tokens.

    - `presence_penalty` (number): Reduces repetitiveness of generated tokens. Similar to frequency_penalty, but this penalty applies equally to all tokens that have already appeared.

    - `return_likelihoods` (string): Specifies how and if token likelihoods are returned with the response. Options include GENERATION, ALL and NONE.

    - `logit_bias` (object): Used to prevent the model from generating unwanted tokens or to incentivize it to include desired tokens. e.g. {"hello_world": 1233}
    """

    num_generations: Optional[int] = None
    max_tokens: Optional[int] = None
    truncate: Optional[str] = None
    temperature: Optional[int] = None
    preset: Optional[str] = None
    end_sequences: Optional[list] = None
    stop_sequences: Optional[list] = None
    k: Optional[int] = None
    p: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    return_likelihoods: Optional[str] = None
    logit_bias: Optional[dict] = None

    def __init__(
        self,
        num_generations: Optional[int] = None,
        max_tokens: Optional[int] = None,
        truncate: Optional[str] = None,
        temperature: Optional[int] = None,
        preset: Optional[str] = None,
        end_sequences: Optional[list] = None,
        stop_sequences: Optional[list] = None,
        k: Optional[int] = None,
        p: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        return_likelihoods: Optional[str] = None,
        logit_bias: Optional[dict] = None,
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


def validate_environment(api_key, headers: dict):
    headers.update(
        {
            "Request-Source": "unspecified:litellm",
            "accept": "application/json",
            "content-type": "application/json",
        }
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
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
    headers: dict,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key, headers=headers)
    completion_url = api_base
    model = model
    prompt = " ".join(message["content"] for message in messages)

    ## Load Config
    config = litellm.CohereConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    ## Handle Tool Calling
    if "tools" in optional_params:
        _is_function_call = True
        tool_calling_system_prompt = construct_cohere_tool(
            tools=optional_params["tools"]
        )
        optional_params["tools"] = tool_calling_system_prompt

    data = {
        "model": model,
        "prompt": prompt,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": completion_url,
        },
    )
    ## COMPLETION CALL
    response = requests.post(
        completion_url,
        headers=headers,
        data=json.dumps(data),
        stream=optional_params["stream"] if "stream" in optional_params else False,
    )
    ## error handling for cohere calls
    if response.status_code != 200:
        raise CohereError(message=response.text, status_code=response.status_code)

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
        completion_response = response.json()
        if "error" in completion_response:
            raise CohereError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            try:
                choices_list = []
                for idx, item in enumerate(completion_response["generations"]):
                    if len(item["text"]) > 0:
                        message_obj = Message(content=item["text"])
                    else:
                        message_obj = Message(content=None)
                    choice_obj = Choices(
                        finish_reason=item["finish_reason"],
                        index=idx + 1,
                        message=message_obj,
                    )
                    choices_list.append(choice_obj)
                model_response.choices = choices_list  # type: ignore
            except Exception as e:
                raise CohereError(
                    message=response.text, status_code=response.status_code
                )

        ## CALCULATING USAGE
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
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
