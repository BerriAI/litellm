import asyncio
import copy
import json
import os
from copy import deepcopy
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import httpx
from openai import OpenAI

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
from litellm.types.utils import Embedding, EmbeddingResponse
from litellm.utils import convert_to_model_response_object, is_base64_encoded

from .cohere_transformation import AzureAICohereConfig


class AzureAIEmbedding(OpenAIChatCompletion):

    def _process_response(
        self,
        image_embedding_responses: Optional[List],
        text_embedding_responses: Optional[List],
        image_embeddings_idx: List[int],
        model_response: EmbeddingResponse,
        input: List,
    ):
        combined_responses = []
        if (
            image_embedding_responses is not None
            and text_embedding_responses is not None
        ):
            # Combine and order the results
            text_idx = 0
            image_idx = 0

            for idx in range(len(input)):
                if idx in image_embeddings_idx:
                    combined_responses.append(image_embedding_responses[image_idx])
                    image_idx += 1
                else:
                    combined_responses.append(text_embedding_responses[text_idx])
                    text_idx += 1

            model_response.data = combined_responses
        elif image_embedding_responses is not None:
            model_response.data = image_embedding_responses
        elif text_embedding_responses is not None:
            model_response.data = text_embedding_responses

        response = AzureAICohereConfig()._transform_response(response=model_response)  # type: ignore

        return response

    async def async_image_embedding(
        self,
        model: str,
        data: ImageEmbeddingRequest,
        timeout: float,
        logging_obj,
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> EmbeddingResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = AsyncHTTPHandler(timeout=timeout, concurrent_limit=1)

        url = "{}/images/embeddings".format(api_base)

        response = await client.post(
            url=url,
            json=data,  # type: ignore
            headers={"Authorization": "Bearer {}".format(api_key)},
        )

        embedding_response = response.json()
        embedding_headers = dict(response.headers)
        returned_response: litellm.EmbeddingResponse = convert_to_model_response_object(  # type: ignore
            response_object=embedding_response,
            model_response_object=model_response,
            response_type="embedding",
            stream=False,
            _response_headers=embedding_headers,
        )
        return returned_response

    def image_embedding(
        self,
        model: str,
        data: ImageEmbeddingRequest,
        timeout: float,
        logging_obj,
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ):
        if api_base is None:
            raise ValueError(
                "api_base is None. Please set AZURE_AI_API_BASE or dynamically via `api_base` param, to make the request."
            )
        if api_key is None:
            raise ValueError(
                "api_key is None. Please set AZURE_AI_API_KEY or dynamically via `api_key` param, to make the request."
            )

        if client is None or not isinstance(client, HTTPHandler):
            client = HTTPHandler(timeout=timeout, concurrent_limit=1)

        url = "{}/images/embeddings".format(api_base)

        response = client.post(
            url=url,
            json=data,  # type: ignore
            headers={"Authorization": "Bearer {}".format(api_key)},
        )

        embedding_response = response.json()
        embedding_headers = dict(response.headers)
        returned_response: litellm.EmbeddingResponse = convert_to_model_response_object(  # type: ignore
            response_object=embedding_response,
            model_response_object=model_response,
            response_type="embedding",
            stream=False,
            _response_headers=embedding_headers,
        )
        return returned_response

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

        (
            image_embeddings_request,
            v1_embeddings_request,
            image_embeddings_idx,
        ) = AzureAICohereConfig()._transform_request(
            input=input, optional_params=optional_params, model=model
        )

        image_embedding_responses: Optional[List] = None
        text_embedding_responses: Optional[List] = None

        if image_embeddings_request["input"]:
            image_response = await self.async_image_embedding(
                model=model,
                data=image_embeddings_request,
                timeout=timeout,
                logging_obj=logging_obj,
                model_response=model_response,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                client=client,
            )

            image_embedding_responses = image_response.data
            if image_embedding_responses is None:
                raise Exception("/image/embeddings route returned None Embeddings.")

        if v1_embeddings_request["input"]:
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
            text_embedding_responses = response.data
            if text_embedding_responses is None:
                raise Exception("/v1/embeddings route returned None Embeddings.")

        return self._process_response(
            image_embedding_responses=image_embedding_responses,
            text_embedding_responses=text_embedding_responses,
            image_embeddings_idx=image_embeddings_idx,
            model_response=model_response,
            input=input,
        )

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

        (
            image_embeddings_request,
            v1_embeddings_request,
            image_embeddings_idx,
        ) = AzureAICohereConfig()._transform_request(
            input=input, optional_params=optional_params, model=model
        )

        image_embedding_responses: Optional[List] = None
        text_embedding_responses: Optional[List] = None

        if image_embeddings_request["input"]:
            image_response = self.image_embedding(
                model=model,
                data=image_embeddings_request,
                timeout=timeout,
                logging_obj=logging_obj,
                model_response=model_response,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                client=client,
            )

            image_embedding_responses = image_response.data
            if image_embedding_responses is None:
                raise Exception("/image/embeddings route returned None Embeddings.")

        if v1_embeddings_request["input"]:
            response: EmbeddingResponse = super().embedding(  # type: ignore
                model,
                input,
                timeout,
                logging_obj,
                model_response,
                optional_params,
                api_key,
                api_base,
                client=(
                    client
                    if client is not None and isinstance(client, OpenAI)
                    else None
                ),
                aembedding=aembedding,
            )

            text_embedding_responses = response.data
            if text_embedding_responses is None:
                raise Exception("/v1/embeddings route returned None Embeddings.")

        return self._process_response(
            image_embedding_responses=image_embedding_responses,
            text_embedding_responses=text_embedding_responses,
            image_embeddings_idx=image_embeddings_idx,
            model_response=model_response,
            input=input,
        )
