import asyncio
import copy
import json
import os
from copy import deepcopy
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.cohere.embed import embedding as cohere_embedding
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.OpenAI.openai import OpenAIChatCompletion
from litellm.types.llms.azure_ai import ImageEmbeddingRequest
from litellm.types.utils import EmbeddingResponse

from .cohere_transformation import AzureAICohereConfig


class AzureAIEmbedding(OpenAIChatCompletion):

    def image_embedding(
        self,
        model: str,
        data: ImageEmbeddingRequest,
        timeout: float,
        logging_obj,
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        api_key: str,
        api_base: str,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        aembedding=None,
    ) -> EmbeddingResponse:
        if client is None or not isinstance(client, HTTPHandler):
            client = HTTPHandler(timeout=timeout, concurrent_limit=1)

        url = "{}/images/embeddings".format(api_base)

        response = client.post(url, data=json.dumps(data))

        embedding_response = response.json()
        return EmbeddingResponse(**embedding_response)

    async def async_embedding(
        self,
        model: str,
        input: List,
        timeout: float,
        logging_obj,
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
    ) -> EmbeddingResponse:
        response: EmbeddingResponse = await super().embedding(  # type: ignore
            model=model,
            input=input,
            timeout=timeout,
            logging_obj=logging_obj,
            model_response=model_response,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            client=client,
            aembedding=True,
        )
        response = AzureAICohereConfig()._transform_response(response=response)  # type: ignore

        return response

    def embedding(
        self,
        model: str,
        input: List,
        timeout: float,
        logging_obj,
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        aembedding=None,
    ):
        """
        - Separate image url from text
        -> route image url call to `/image/embeddings`
        -> route text call to `/v1/embeddings` (OpenAI route)

        assemble result in-order, and return
        """
        if aembedding is True:
            return self.async_embedding(
                model,
                input,
                timeout,
                logging_obj,
                model_response,
                optional_params,
                api_key,
                api_base,
                client,
            )
        response = super().embedding(
            model,
            input,
            timeout,
            logging_obj,
            model_response,
            optional_params,
            api_key,
            api_base,
            client,
            aembedding,
        )

        response = AzureAICohereConfig()._transform_response(response=response)  # type: ignore

        return response
