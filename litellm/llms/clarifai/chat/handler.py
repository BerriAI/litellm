import json
import os
import time
import traceback
import types
from typing import Callable, List, Optional

import httpx
import requests

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import Choices, CustomStreamWrapper, Message, ModelResponse, Usage

from ...prompt_templates.factory import custom_prompt, prompt_factory
from ..common_utils import ClarifaiError


async def async_completion(
    model: str,
    messages: List[AllMessageValues],
    model_response: ModelResponse,
    encoding,
    api_key,
    api_base: str,
    logging_obj,
    data: dict,
    optional_params: dict,
    litellm_params=None,
    logger_fn=None,
    headers={},
):

    async_handler = get_async_httpx_client(
        llm_provider=litellm.LlmProviders.CLARIFAI,
        params={"timeout": 600.0},
    )
    response = await async_handler.post(
        url=api_base, headers=headers, data=json.dumps(data)
    )

    return litellm.ClarifaiConfig().transform_response(
        model=model,
        raw_response=response,
        model_response=model_response,
        logging_obj=logging_obj,
        api_key=api_key,
        request_data=data,
        messages=messages,
        optional_params=optional_params,
        encoding=encoding,
    )


def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params: dict,
    litellm_params: dict,
    custom_prompt_dict={},
    acompletion=False,
    logger_fn=None,
    headers={},
):
    headers = litellm.ClarifaiConfig().validate_environment(
        api_key=api_key,
        headers=headers,
        model=model,
        messages=messages,
        optional_params=optional_params,
    )
    data = litellm.ClarifaiConfig().transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    ## LOGGING
    logging_obj.pre_call(
        input=data,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": model,
        },
    )
    if acompletion is True:
        return async_completion(
            model=model,
            messages=messages,
            api_base=api_base,
            model_response=model_response,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            data=data,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
        )
    else:
        ## COMPLETION CALL
        httpx_client = _get_httpx_client(
            params={"timeout": 600.0},
        )
        response = httpx_client.post(
            url=api_base,
            headers=headers,
            data=json.dumps(data),
        )

    if response.status_code != 200:
        raise ClarifaiError(status_code=response.status_code, message=response.text)

    if "stream" in optional_params and optional_params["stream"] is True:
        completion_stream = response.iter_lines()
        stream_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="clarifai",
            logging_obj=logging_obj,
        )
        return stream_response

    else:
        return litellm.ClarifaiConfig().transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
        )


class ModelResponseIterator:
    def __init__(self, model_response):
        self.model_response = model_response
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.model_response
