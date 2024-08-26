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


def _process_embedding_response(
    embeddings: list,
    model_response: litellm.EmbeddingResponse,
    model: str,
    encoding: Any,
    input: list,
) -> litellm.EmbeddingResponse:
    output_data = []
    for idx, embedding in enumerate(embeddings):
        output_data.append(
            {"object": "embedding", "index": idx, "embedding": embedding}
        )
    model_response.object = "list"
    model_response.data = output_data
    model_response.model = model
    input_tokens = 0
    for text in input:
        input_tokens += len(encoding.encode(text))

    setattr(
        model_response,
        "usage",
        Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        ),
    )

    return model_response


async def async_embedding(
    model: str,
    data: dict,
    input: list,
    model_response: litellm.utils.EmbeddingResponse,
    timeout: Union[float, httpx.Timeout],
    logging_obj: LiteLLMLoggingObj,
    optional_params: dict,
    api_base: str,
    api_key: Optional[str],
    headers: dict,
    encoding: Callable,
    client: Optional[AsyncHTTPHandler] = None,
):

    ## LOGGING
    logging_obj.pre_call(
        input=input,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": api_base,
        },
    )
    ## COMPLETION CALL
    if client is None:
        client = AsyncHTTPHandler(concurrent_limit=1)

    response = await client.post(api_base, headers=headers, data=json.dumps(data))

    ## LOGGING
    logging_obj.post_call(
        input=input,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
        original_response=response,
    )

    embeddings = response.json()["embeddings"]

    ## PROCESS RESPONSE ##
    return _process_embedding_response(
        embeddings=embeddings,
        model_response=model_response,
        model=model,
        encoding=encoding,
        input=input,
    )


def embedding(
    model: str,
    input: list,
    model_response: litellm.EmbeddingResponse,
    logging_obj: LiteLLMLoggingObj,
    optional_params: dict,
    headers: dict,
    encoding: Any,
    api_key: Optional[str] = None,
    aembedding: Optional[bool] = None,
    timeout: Union[float, httpx.Timeout] = httpx.Timeout(None),
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
):
    headers = validate_environment(api_key, headers=headers)
    embed_url = "https://api.cohere.ai/v1/embed"
    model = model
    data = {"model": model, "texts": input, **optional_params}

    if "3" in model and "input_type" not in data:
        # cohere v3 embedding models require input_type, if no input_type is provided, default to "search_document"
        data["input_type"] = "search_document"

    ## LOGGING
    logging_obj.pre_call(
        input=input,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )

    ## ROUTING
    if aembedding is True:
        return async_embedding(
            model=model,
            data=data,
            input=input,
            model_response=model_response,
            timeout=timeout,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_base=embed_url,
            api_key=api_key,
            headers=headers,
            encoding=encoding,
        )
    ## COMPLETION CALL
    if client is None or not isinstance(client, HTTPHandler):
        client = HTTPHandler(concurrent_limit=1)
    response = client.post(embed_url, headers=headers, data=json.dumps(data))
    ## LOGGING
    logging_obj.post_call(
        input=input,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
        original_response=response,
    )
    """
        response 
        {
            'object': "list",
            'data': [
            
            ]
            'model', 
            'usage'
        }
    """
    if response.status_code != 200:
        raise CohereError(message=response.text, status_code=response.status_code)
    embeddings = response.json()["embeddings"]

    return _process_embedding_response(
        embeddings=embeddings,
        model_response=model_response,
        model=model,
        encoding=encoding,
        input=input,
    )
