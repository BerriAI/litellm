import json
import os
import time
import traceback
import types
from enum import Enum
from typing import Callable, List, Optional

import requests  # type: ignore

import litellm
from litellm.utils import Choices, Message, ModelResponse, Usage


class MaritalkError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class MaritTalkConfig:
    """
    The class `MaritTalkConfig` provides configuration for the MaritTalk's API interface. Here are the parameters:

    - `max_tokens` (integer): Maximum number of tokens the model will generate as part of the response. Default is 1.

    - `model` (string): The model used for conversation. Default is 'maritalk'.

    - `do_sample` (boolean): If set to True, the API will generate a response using sampling. Default is True.

    - `temperature` (number): A non-negative float controlling the randomness in generation. Lower temperatures result in less random generations. Default is 0.7.

    - `top_p` (number): Selection threshold for token inclusion based on cumulative probability. Default is 0.95.

    - `repetition_penalty` (number): Penalty for repetition in the generated conversation. Default is 1.

    - `stopping_tokens` (list of string): List of tokens where the conversation can be stopped/stopped.
    """

    max_tokens: Optional[int] = None
    model: Optional[str] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stopping_tokens: Optional[List[str]] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stopping_tokens: Optional[List[str]] = None,
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


def validate_environment(api_key):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Key {api_key}"
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
    completion_url = api_base
    model = model

    ## Load Config
    config = litellm.MaritTalkConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > maritalk_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    data = {
        "messages": messages,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )
    ## COMPLETION CALL
    response = requests.post(
        completion_url,
        headers=headers,
        data=json.dumps(data),
        stream=optional_params["stream"] if "stream" in optional_params else False,
    )
    if "stream" in optional_params and optional_params["stream"] == True:
        return response.iter_lines()
    else:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        completion_response = response.json()
        if "error" in completion_response:
            raise MaritalkError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            try:
                if len(completion_response["answer"]) > 0:
                    model_response.choices[0].message.content = completion_response[  # type: ignore
                        "answer"
                    ]
            except Exception as e:
                raise MaritalkError(
                    message=response.text, status_code=response.status_code
                )

        ## CALCULATING USAGE
        prompt = "".join(m["content"] for m in messages)
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


def embedding(
    model: str,
    input: list,
    api_key: Optional[str] = None,
    logging_obj=None,
    model_response=None,
    encoding=None,
):
    pass
