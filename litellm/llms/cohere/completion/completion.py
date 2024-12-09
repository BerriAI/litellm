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

from ..common_utils import CohereError


def construct_cohere_tool(tools=None):
    if tools is None:
        tools = []
    return {"tools": tools}


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
    optional_params: dict,
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

    if "stream" in optional_params and optional_params["stream"] is True:
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
            except Exception:
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
