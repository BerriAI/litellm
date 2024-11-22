# What is this?
## Handler file for OpenAI-like endpoints.
## Allows jina ai embedding calls - which don't allow 'encoding_format' in payload.

import copy
import json
import os
import time
import types
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.utils import EmbeddingResponse

from ..common_utils import OpenAILikeBase, OpenAILikeError


class OpenAILikeEmbeddingHandler(OpenAILikeBase):
    def __init__(self, **kwargs):
        pass

    async def aembedding(
        self,
        input: list,
        data: dict,
        model_response: EmbeddingResponse,
        timeout: float,
        api_key: str,
        api_base: str,
        logging_obj,
        headers: dict,
        client=None,
    ) -> EmbeddingResponse:
        response = None
        try:
            if client is None or isinstance(client, AsyncHTTPHandler):
                self.async_client = get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.OPENAI,
                    params={"timeout": timeout},
                )
            else:
                self.async_client = client

            try:
                response = await self.async_client.post(
                    api_base,
                    headers=headers,
                    data=json.dumps(data),
                )  # type: ignore

                response.raise_for_status()

                response_json = response.json()
            except httpx.HTTPStatusError as e:
                raise OpenAILikeError(
                    status_code=e.response.status_code,
                    message=e.response.text if e.response else str(e),
                )
            except httpx.TimeoutException:
                raise OpenAILikeError(
                    status_code=408, message="Timeout error occurred."
                )
            except Exception as e:
                raise OpenAILikeError(status_code=500, message=str(e))

            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response_json,
            )
            return EmbeddingResponse(**response_json)
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                original_response=str(e),
            )
            raise e

    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        logging_obj,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        model_response: Optional[litellm.utils.EmbeddingResponse] = None,
        client=None,
        aembedding=None,
        custom_endpoint: Optional[bool] = None,
        headers: Optional[dict] = None,
    ) -> EmbeddingResponse:
        api_base, headers = self._validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="embeddings",
            headers=headers,
            custom_endpoint=custom_endpoint,
        )
        model = model
        data = {"model": model, "input": input, **optional_params}

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data, "api_base": api_base},
        )

        if aembedding is True:
            return self.aembedding(data=data, input=input, logging_obj=logging_obj, model_response=model_response, api_base=api_base, api_key=api_key, timeout=timeout, client=client, headers=headers)  # type: ignore
        if client is None or isinstance(client, AsyncHTTPHandler):
            self.client = HTTPHandler(timeout=timeout)  # type: ignore
        else:
            self.client = client

        ## EMBEDDING CALL
        try:
            response = self.client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
            )  # type: ignore

            response.raise_for_status()  # type: ignore

            response_json = response.json()  # type: ignore
        except httpx.HTTPStatusError as e:
            raise OpenAILikeError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except httpx.TimeoutException:
            raise OpenAILikeError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise OpenAILikeError(status_code=500, message=str(e))

        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response_json,
        )

        return litellm.EmbeddingResponse(**response_json)
