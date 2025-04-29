import json
from typing import Any, Callable, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.bedrock import CohereEmbeddingRequest
from litellm.types.utils import EmbeddingResponse

from .transformation import CohereEmbeddingConfig


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


async def async_embedding(
    model: str,
    data: Union[dict, CohereEmbeddingRequest],
    input: list,
    model_response: litellm.utils.EmbeddingResponse,
    timeout: Optional[Union[float, httpx.Timeout]],
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
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.COHERE,
            params={"timeout": timeout},
        )

    try:
        response = await client.post(api_base, headers=headers, data=json.dumps(data))
    except httpx.HTTPStatusError as e:
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=e.response.text,
        )
        raise e
    except Exception as e:
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=str(e),
        )
        raise e

    ## PROCESS RESPONSE ##
    return CohereEmbeddingConfig()._transform_response(
        response=response,
        api_key=api_key,
        logging_obj=logging_obj,
        data=data,
        model_response=model_response,
        model=model,
        encoding=encoding,
        input=input,
    )


def embedding(
    model: str,
    input: list,
    model_response: EmbeddingResponse,
    logging_obj: LiteLLMLoggingObj,
    optional_params: dict,
    headers: dict,
    encoding: Any,
    data: Optional[Union[dict, CohereEmbeddingRequest]] = None,
    complete_api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    aembedding: Optional[bool] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = httpx.Timeout(None),
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
):
    headers = validate_environment(api_key, headers=headers)
    embed_url = complete_api_base or "https://api.cohere.ai/v1/embed"
    model = model

    data = data or CohereEmbeddingConfig()._transform_request(
        model=model, input=input, inference_params=optional_params
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
            client=(
                client
                if client is not None and isinstance(client, AsyncHTTPHandler)
                else None
            ),
        )

    ## LOGGING
    logging_obj.pre_call(
        input=input,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )

    ## COMPLETION CALL
    if client is None or not isinstance(client, HTTPHandler):
        client = HTTPHandler(concurrent_limit=1)

    response = client.post(embed_url, headers=headers, data=json.dumps(data))

    return CohereEmbeddingConfig()._transform_response(
        response=response,
        api_key=api_key,
        logging_obj=logging_obj,
        data=data,
        model_response=model_response,
        model=model,
        encoding=encoding,
        input=input,
    )
