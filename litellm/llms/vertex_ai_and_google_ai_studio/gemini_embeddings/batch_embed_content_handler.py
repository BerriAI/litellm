"""
Google AI Studio /batchEmbedContents Embeddings Endpoint
"""

import json
from typing import Any, List, Literal, Optional, Union

import httpx

from litellm import EmbeddingResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import EmbeddingInput
from litellm.types.llms.vertex_ai import (
    VertexAIBatchEmbeddingsRequestBody,
    VertexAIBatchEmbeddingsResponseObject,
)

from ..gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from .batch_embed_content_transformation import (
    process_response,
    transform_openai_input_gemini_content,
)


class GoogleBatchEmbeddings(VertexLLM):
    def batch_embeddings(
        self,
        model: str,
        input: EmbeddingInput,
        print_verbose,
        model_response: EmbeddingResponse,
        custom_llm_provider: Literal["gemini", "vertex_ai"],
        optional_params: dict,
        logging_obj: Any,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        encoding=None,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        aembedding=False,
        timeout=300,
        client=None,
    ) -> EmbeddingResponse:

        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, url = self._get_token_and_url(
            model=model,
            auth_header=_auth_header,
            gemini_api_key=api_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=None,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="batch_embedding",
        )

        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        optional_params = optional_params or {}

        ### TRANSFORMATION ###
        request_data = transform_openai_input_gemini_content(
            input=input, model=model, optional_params=optional_params
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": url,
                "headers": headers,
            },
        )

        if aembedding is True:
            return self.async_batch_embeddings(  # type: ignore
                model=model,
                api_base=api_base,
                url=url,
                data=request_data,
                model_response=model_response,
                timeout=timeout,
                headers=headers,
                input=input,
            )

        response = sync_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        _predictions = VertexAIBatchEmbeddingsResponseObject(**_json_response)  # type: ignore

        return process_response(
            model=model,
            model_response=model_response,
            _predictions=_predictions,
            input=input,
        )

    async def async_batch_embeddings(
        self,
        model: str,
        api_base: Optional[str],
        url: str,
        data: VertexAIBatchEmbeddingsRequestBody,
        model_response: EmbeddingResponse,
        input: EmbeddingInput,
        timeout: Optional[Union[float, httpx.Timeout]],
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> EmbeddingResponse:
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            async_handler: AsyncHTTPHandler = AsyncHTTPHandler(**_params)  # type: ignore
        else:
            async_handler = client  # type: ignore

        response = await async_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        _predictions = VertexAIBatchEmbeddingsResponseObject(**_json_response)  # type: ignore

        return process_response(
            model=model,
            model_response=model_response,
            _predictions=_predictions,
            input=input,
        )
