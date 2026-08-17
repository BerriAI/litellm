from typing import Final

from openai import OpenAI

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.types.llms.azure_ai import ImageEmbeddingRequest
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

from .cohere_transformation import AzureAICohereConfig


class AzureAIEmbedding(OpenAIChatCompletion):
    def _process_response(
        self,
        image_embedding_responses: list | None,
        text_embedding_responses: list | None,
        image_embeddings_idx: list[int],
        model_response: EmbeddingResponse,
        input: list,
    ):
        combined_responses: Final = []
        if image_embedding_responses is not None and text_embedding_responses is not None:
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

        response: Final = AzureAICohereConfig()._transform_response(response=model_response)

        return response

    async def async_image_embedding(
        self,
        model: str,
        data: ImageEmbeddingRequest,
        timeout: float,
        logging_obj,
        model_response: EmbeddingResponse,
        optional_params: dict,
        api_key: str | None,
        api_base: str | None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
    ) -> EmbeddingResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.AZURE_AI,
                params={"timeout": timeout},
            )

        url: Final = f"{api_base}/images/embeddings"

        response: Final = await client.post(
            url=url,
            json=data,
            headers={"Authorization": f"Bearer {api_key}"},
        )

        embedding_response: Final = response.json()
        embedding_headers: Final = dict(response.headers)
        returned_response: Final[EmbeddingResponse] = convert_to_model_response_object(
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
        model_response: EmbeddingResponse,
        optional_params: dict,
        api_key: str | None,
        api_base: str | None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
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

        url: Final = f"{api_base}/images/embeddings"

        response: Final = client.post(
            url=url,
            json=data,
            headers={"Authorization": f"Bearer {api_key}"},
        )

        embedding_response: Final = response.json()
        embedding_headers: Final = dict(response.headers)
        returned_response: Final[EmbeddingResponse] = convert_to_model_response_object(
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
        input: list,
        timeout: float,
        logging_obj,
        model_response: EmbeddingResponse,
        optional_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        client=None,
    ) -> EmbeddingResponse:
        (
            image_embeddings_request,
            v1_embeddings_request,
            image_embeddings_idx,
        ) = AzureAICohereConfig()._transform_request(input=input, optional_params=optional_params, model=model)

        image_embedding_responses: list | None = None
        text_embedding_responses: list | None = None

        if image_embeddings_request["input"]:
            image_response: Final = await self.async_image_embedding(
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
            response: Final[EmbeddingResponse] = await super().embedding(
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
        input: list,
        timeout: float,
        logging_obj,
        model_response: EmbeddingResponse,
        optional_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        client=None,
        aembedding=None,
        max_retries: int | None = None,
        shared_session=None,
    ) -> EmbeddingResponse:
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
        ) = AzureAICohereConfig()._transform_request(input=input, optional_params=optional_params, model=model)

        image_embedding_responses: list | None = None
        text_embedding_responses: list | None = None

        if image_embeddings_request["input"]:
            image_response: Final = self.image_embedding(
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
            response: Final[EmbeddingResponse] = super().embedding(
                model,
                input,
                timeout,
                logging_obj,
                model_response,
                optional_params,
                api_key,
                api_base,
                client=(client if client is not None and isinstance(client, OpenAI) else None),
                aembedding=aembedding,
                shared_session=shared_session,
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
